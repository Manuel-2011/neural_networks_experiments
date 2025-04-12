# %%
from transformers import AutoModelForCausalLM, AutoTokenizer
from logging import getLogger
import logging
import os
import time

start_time = time.time()

logger = getLogger(__name__)
logfile_name = 'grpo-DrGrpo11.log'
if os.path.exists(logfile_name):
  os.remove(logfile_name)
logging.basicConfig(filename=logfile_name, encoding='utf-8', level=logging.DEBUG)

def get_policy_model():
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def generate_batch_completion(model, tokenizer, prompts: list):
    batch = [[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ] for prompt in prompts]
    texts = tokenizer.apply_chat_template(
        batch,
        tokenize=False,
        add_generation_prompt=True,
    )

    model_inputs = tokenizer(texts, padding='longest', padding_side='left', \
        return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=256,
        temperature = 0.8,
        top_p = 0.95,
    )

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return response

import re

def extract_answer(response, transform_fn = lambda x: x, nan_val = None)->str|None:
    ans = re.match('.*?<answer>(.*?)</answer>', response, re.DOTALL|re.MULTILINE)
    if ans:
        try:
            return transform_fn(ans[1])
        except:
            return nan_val
    return nan_val

import numpy as np
from tqdm import tqdm

def eval_multiplication(model, tokenizer, epochs=10, batch_size=128, generate_fn=generate_batch_completion):
    matches = 0
    tries = 0
    format_errors = 0
    for i in tqdm(range(epochs)):
        numbers = np.random.randint(0,101, (batch_size, 2))
        correct_result = numbers[:,0] * numbers[:,1]
        prompt = "What is the result of {} times {}. Provide the final result in an answer tag <answer>final answer</answer>"
        prompts = [prompt.format(*nums) for nums in numbers]
        responses = generate_fn(model, tokenizer, prompts)
        answer = np.array([extract_answer(response, lambda x: int(x) if x.isnumeric() else x) for response in responses])
        format_errors += (answer == None).sum()
        matches += (correct_result == answer).sum()
        tries += len(correct_result)

        del responses
        del answer
        torch.cuda.empty_cache()
    
    acc = matches/tries
    format_errors /= tries
    wrong_answer = 1 - acc - format_errors
    return {
        'acc': acc,
        'format_errors': format_errors,
        'wrong_answer': wrong_answer
    }


import torch
from torch import nn


# %%
def make_rollouts(model, simulations, initial_prompt: str, max_size = 256, temperature=1.0):
    input_tokens = tokenizer(initial_prompt, return_tensors='pt').input_ids
    # Use a single prompt to generate different responses
    input_tokens = input_tokens.repeat(simulations, 1).to(model.device)
    
    prompt_length = input_tokens.shape[1]
    is_terminal = torch.zeros((simulations, max_size + prompt_length))
    # generate each response token in a batch and calculate the value for each of them
    for i in range(max_size):
        # TODO: Only calculate the response logits and values one and use those to do the optimization step
        with torch.no_grad():
            # Calculate the next token logits
            logits = model(input_tokens).logits[:,-1,:].to(torch.float) # [sims, vocab]
            logits /= temperature

            # sample next action (next token)
            probs = torch.nn.functional.softmax(logits, dim=1)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=1)
            # topk=50
            indices = sorted_indices[:,:50]
            probs = sorted_probs[:,:50]
            probs = probs / probs.sum(dim=1, keepdim=True)
            distribution = torch.distributions.Categorical(probs)
            action = distribution.sample() # [sims]
            action = sorted_indices[torch.arange(len(action)), action]
            # Concatenate to the prompt
            input_tokens = torch.cat((input_tokens, action.unsqueeze(1)), dim=1)

            # Check if the new token is the eos token
            if i > 0:
                is_terminal[:,prompt_length+i] = torch.max(is_terminal[:,prompt_length+i-1], (action.cpu() == tokenizer.eos_token_id))

            # Check if all the generations finished early
            if (is_terminal[:,prompt_length+i] == 1).all():
                is_terminal = is_terminal[:,:prompt_length+i+1]
                break

    # Pad the generation with the eos token to the right
    generations = input_tokens.masked_fill(is_terminal.to(input_tokens.device) == 1, tokenizer.eos_token_id)
    return generations[:,prompt_length:], is_terminal[:,prompt_length:], generations, prompt_length

# %%
def get_rewards(samples, is_terminal, correct_result):
    samples = samples.cpu()
    is_terminal = is_terminal.cpu()
    rewards = torch.zeros_like(samples, dtype=torch.float)
    # Extract the answer from the answer tag if any on each response
    samples = tokenizer.batch_decode(samples, skip_special_tokens=True)
    logger.debug(f'samples: {samples}')
    answer = torch.tensor([extract_answer(response, lambda x: int(x) if x.isnumeric() else correct_result+1, torch.nan) for response in samples])

    eos_index = (is_terminal == 0).sum(dim=1)
    logger.info(f'Response length mean: {eos_index.to(torch.float16).mean():,.2f}')
    eos_index = torch.min(eos_index, torch.tensor(is_terminal.shape[1]-1))

    answer_is_correct = (answer == correct_result)
    answer_is_not_correct = (answer != correct_result)
    wrong_format = answer.isnan()
    answer_is_correct_count = answer_is_correct.sum()
    answer_is_not_correct_count = answer_is_not_correct.sum()
    wrong_format_count = wrong_format.sum()
    logger.debug(f'Correct answer: {correct_result} Extracted: {answer}')
    logger.debug(f'Correct: {answer_is_correct_count}, Wrong_format: {wrong_format_count}, Wrong_anser: {answer_is_not_correct_count-wrong_format_count}')

    # 0.5 reward point if the response has an answer tag
    rewards[torch.arange(len(samples)), eos_index] = (1-wrong_format.to(torch.float32))*0.5
    # An additional 1 point of reward if the answer is correct
    rewards[torch.arange(len(samples)), eos_index] += answer_is_correct.to(torch.float32)
    logger.debug(f'Rewards: {rewards[torch.arange(len(samples)), eos_index]}')
    return rewards

# %%
def run_one_mul_simulation(model, generations_num, temperature=1.0):
    # Generate the random numbers to be multiplied to create the prompt
    numbers = torch.randint(0,101, (1, 2))
    # numbers = torch.tensor([[23, 37]])
    correct_result = numbers[:,0] * numbers[:,1]
    prompt = "What is the result of {} times {}. Provide the final result in an answer tag <answer>final answer</answer>"
    prompts = [prompt.format(*nums) for nums in numbers]
    batch = [[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ] for prompt in prompts]
    texts = tokenizer.apply_chat_template(
        batch,
        tokenize=False,
        add_generation_prompt=True
    )

    # Generate the responses for the prompt
    inputs, is_terminal, complete_prompts, prompt_length = make_rollouts(model, generations_num, texts[0], temperature=temperature)
    # Calculate the rewards for each response
    rewards = get_rewards(inputs, is_terminal, correct_result)
    return inputs, rewards, is_terminal, complete_prompts, prompt_length


# %%
def compute_advantages(rewards, is_terminal, gamma=1.0, gae_lambda=0.2, dr_grpo=False):
    # Find the longest response in the batch
    num_rollout_steps = torch.max((is_terminal==0).sum(1))
    num_rollout_steps = torch.min(num_rollout_steps, torch.tensor(rewards.shape[1])-1)
    advantages = torch.zeros((len(rewards), torch.tensor(rewards.shape[1])))

    eos_index = (is_terminal == 0).sum(dim=1)
    eos_index = torch.min(num_rollout_steps, eos_index)
    rewards_of_outputs = rewards[torch.arange(len(rewards)), eos_index]
    norm_rewards = (rewards_of_outputs - rewards_of_outputs.mean()) 
    if not dr_grpo:
        norm_rewards /= (rewards_of_outputs.std() + 1e-8)

    for i in range(len(rewards)):
        advantages[i,:eos_index[i]+1] = norm_rewards[i]

    logger.debug(f'advantages: {advantages}')
    return advantages

# %%
import numpy as np


# %%
def update_policy(model, ref_model, old_model, optimizer, is_terminal, advantanges, complete_prompts, prompt_length, generations, minibatch_size, update_epochs, scheduler=None, normalize_advantage=False, lower_clip=None, upper_clip=None, kl_penalty_coef=0.04, dr_grpo=False):
    lower_clipped_threshold = lower_clip
    upper_clipped_threshold = upper_clip

    for epoch in range(update_epochs):
        batch_indices = np.arange(len(advantanges))
        np.random.shuffle(batch_indices)
        is_terminal = is_terminal.to(model.device)
        advantanges = advantanges.to(model.device)
        minibatches= len(advantanges) // minibatch_size
        batch_size = len(batch_indices)

        ## Iterate in minibatches (random minibatch_size responses)
        for start in range(0, len(advantanges), minibatch_size):
            end = start + minibatch_size
            minibatch_indices = batch_indices[start:end]
            # Calculate the logits for the generated responses with the policy model (the logits of the previous token represents the distribution of the current token, that's why the -1)
            logits = model(complete_prompts[minibatch_indices,:prompt_length+advantanges.shape[1]]).logits[:,prompt_length-1:-1]
            if old_model is not None:
                with torch.no_grad():
                    old_logits  = old_model(complete_prompts[minibatch_indices,:prompt_length+advantanges.shape[1]]).logits[:,prompt_length-1:-1]
            # Calculate the logits for the generated responses with the ref model
            with torch.no_grad():
                ref_logits  = ref_model(complete_prompts[minibatch_indices,:prompt_length+advantanges.shape[1]]).logits[:,prompt_length-1:-1]
            # Get the ids of the actual generated tokens
            completion_ids = generations[minibatch_indices]
            completion_ids = generations[minibatch_indices,:advantanges.shape[1]].reshape(-1)
            actual_minibatch_size = len(logits)
            max_tokens = advantanges.shape[1]
            # Calculate the probabilities of each token generated with the policy and ref model
            probs = nn.functional.softmax(logits.reshape(actual_minibatch_size*max_tokens,-1), dim=1)
            probs_tokens = probs[torch.arange(len(completion_ids)), completion_ids]
            log_probs_sum = torch.log(probs_tokens).reshape(advantanges.shape)
            if old_model:
                old_probs = nn.functional.softmax(old_logits.reshape(actual_minibatch_size*max_tokens,-1), dim=1)
                old_probs_tokens = old_probs[torch.arange(len(completion_ids)), completion_ids]
                log_old_probs_sum = torch.log(old_probs_tokens).reshape(advantanges.shape)
            else:
                log_old_probs_sum = log_probs_sum.detach()
            ref_probs = nn.functional.softmax(ref_logits.reshape(actual_minibatch_size*max_tokens,-1), dim=1)
            ref_probs_tokens = ref_probs[torch.arange(len(completion_ids)), completion_ids]
            log_ref_probs_sum = torch.log(ref_probs_tokens).reshape(advantanges.shape)

            # terminal state is shift to the right so the eos token that has the reward is taken in account
            minibatch_is_not_terminal = torch.cat((torch.ones(actual_minibatch_size,1, device=model.device), 1-is_terminal[minibatch_indices]), dim=1)[:,:advantanges.shape[1]]

            # Track KL divergence
            log_prob_ratio = log_probs_sum - log_ref_probs_sum
            probability_ratio = log_prob_ratio.exp()
            minibatch_approx_kl = ((probability_ratio - 1) - log_prob_ratio) * minibatch_is_not_terminal
            minibatch_approx_kl_by_generation = (minibatch_approx_kl.sum(dim=1) / minibatch_is_not_terminal.count_nonzero(dim=1))
            minibatch_approx_kl_mean = minibatch_approx_kl_by_generation.mean()
            logger.info(f'minibatch_approx_kl_by_generation: {minibatch_approx_kl_by_generation}')
            logger.debug(f'Approx KL divergence of minibatch: {minibatch_approx_kl_mean:.6f}')

            minibatch_advantages = advantanges[minibatch_indices,:advantanges.shape[1]] * minibatch_is_not_terminal

            # The policy loss is to maximize the probability_ratio times the advantages
            new_old_prob_ratio = (log_probs_sum-log_old_probs_sum).exp()
            # Verification in case the old model is the same as the current one
            logger.debug(f'new_old_prob_ratio. This should be 1, {(new_old_prob_ratio*minibatch_is_not_terminal).sum()/minibatch_is_not_terminal.count_nonzero()}')
            loss = new_old_prob_ratio * minibatch_advantages
            logger.debug(f'probability ratio: mean - {probability_ratio.mean()}, min - {probability_ratio.min()}, max: {probability_ratio.max()}')
            # Clipped loss: Only considers the probability_ratio change between a reasonable range
            if lower_clipped_threshold != None and upper_clipped_threshold != None:
                clipped_loss = torch.clamp(new_old_prob_ratio, lower_clipped_threshold, upper_clipped_threshold) * minibatch_advantages
            else:
                clipped_loss = loss

            logger.debug(f'generation_length: {minibatch_is_not_terminal.count_nonzero(dim=1)}')
            # Take the min to be pessimistic
            if dr_grpo:
                loss_mean = torch.min(loss, clipped_loss).sum(dim=1).mean()
            else:
                loss_avg_by_generation = torch.min(loss, clipped_loss).sum(dim=1) / minibatch_is_not_terminal.count_nonzero(dim=1)
                logger.debug(f'loss_avg_by_generation: {loss_avg_by_generation}')
                loss_mean = loss_avg_by_generation.mean()
            logger.debug(f'loss_mean: {loss_mean}')
            # Add the minus to perform optimization minimizing the loss
            if dr_grpo:
                loss = -loss_mean
            else:
                loss_with_kl_penalty = loss_mean - kl_penalty_coef*minibatch_approx_kl_mean
                loss = -loss_with_kl_penalty

            logger.debug(f'loss: {loss.item()}')

            # Update the policy weights
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.1) # Avoid large gradients
            optimizer.step()

    # Update the scheduler every rl step no matter the epochs
    if scheduler:
        scheduler.step()


# Cosine scheduler with warmup
from torch.optim.lr_scheduler import CosineAnnealingLR
class CosineAnnealingWithWarmup:
    def __init__(self, optimizer, warmup_epochs, total_epochs, warmup_start_lr=0.0, eta_min=0):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.warmup_start_lr = warmup_start_lr
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.cosine_scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs, eta_min=eta_min)
        self.current_epoch = 0
    
    def step(self):
        if self.current_epoch < self.warmup_epochs:
            warmup_factor = self.current_epoch / self.warmup_epochs
            for i, param_group in enumerate(self.optimizer.param_groups):
                param_group['lr'] = self.warmup_start_lr + warmup_factor * (self.base_lrs[i] - self.warmup_start_lr)
        else:
            self.cosine_scheduler.step()
        self.current_epoch += 1
    
    def get_last_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]
    
    def get_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]

# %% [markdown]
# ## Putting everything together

# %%
from peft import LoraConfig, get_peft_model 

model, tokenizer = get_policy_model()
ref_model, _ = get_policy_model()

config = LoraConfig(
    r=64, # Rank de las matrices A y B
    lora_alpha=64, # Factor de regularización de las matrices A y B
    target_modules= [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    # lora_dropout=0.05, # Dropout de las matrices A y B
    # bias="none", # No se añade bias a las capas lineales
    task_type="CAUSAL_LM" # Tipo de tarea
)

# Use LoRA to finetune the policy model
model = get_peft_model(model, config)

ref_model.eval()

# %%
max_steps = 250*8
sims_per_prompt = 8
rl_steps = max_steps // sims_per_prompt
minibatch_size = 8
update_epochs = 1

model.train()


policy_lr = 5e-6
kl_penalty_coef = 0.04
warmup_steps = 25
optimizer = torch.optim.AdamW(model.parameters(), lr=policy_lr, betas=(0.9, 0.99), weight_decay=0.1)
# scheduler = CosineAnnealingLR(optimizer, T_max=rl_steps)
scheduler = CosineAnnealingWithWarmup(optimizer, warmup_steps, rl_steps)
scheduler.step()
update_ref_model_steps = None
gae_lambda = 1.0
normalize_advantage=False
temperature=1.0 # Temperature for the generations
lower_clip=0.8
upper_clip=1.2
dr_grpo = True
logger.info(f'Hyperparameters:\nupdate_epochs:{update_epochs}\nrl_steps:{rl_steps}\nsims_per_prompt:{sims_per_prompt}\nminibatch_size:{minibatch_size}\npolicy_lr:{policy_lr}\nwarmup_steps:{warmup_steps}\ngae_lambda: {gae_lambda}\nnormalize advantage:{normalize_advantage}\nlower_clip:{lower_clip}\nupper_clip:{upper_clip}\nkl_penalty_coef:{kl_penalty_coef}\ntemperature:{temperature}\ndr_grpo:{dr_grpo}')


import copy
# Training loop
try:
    # model.eval()
    # with torch.no_grad():
    #     acc = eval_multiplication(model, tokenizer, epochs=40, batch_size=64)
    # logger.info(f'Evaluation before training: {acc}')
    model.train()
    old_model = None

    rl_step = 0
    while rl_step < rl_steps:
        logger.info(f'rl_step: {rl_step+1:,}')
        generations, rewards, is_terminal, complete_prompts, prompt_length = run_one_mul_simulation(model, sims_per_prompt, temperature=temperature)
        advantanges = compute_advantages(rewards, is_terminal, gae_lambda=gae_lambda, dr_grpo=dr_grpo)
        if (advantanges == 0).all().item():
            continue

        logger.info('Updating policy')
        logger.debug(f'Learning rate: {scheduler.get_lr()}')
        update_policy(model, ref_model, old_model, optimizer, is_terminal, advantanges, complete_prompts, prompt_length, generations, minibatch_size, update_epochs, scheduler=scheduler, normalize_advantage=normalize_advantage, lower_clip=lower_clip, upper_clip=upper_clip, dr_grpo=dr_grpo)

        # Track progress on specific task
        if (rl_step+1)%10 == 0:
            model.eval()
            with torch.no_grad():
                acc = eval_multiplication(model, tokenizer, epochs=20, batch_size=64)
            logger.info(f'Evaluation on rl step {rl_step+1:,}: {acc}')
            model.train()

        if update_ref_model_steps is not None and (rl_step+1)%update_ref_model_steps == 0:
            ref_model = copy.deepcopy(model).eval() # Update the ref model

        rl_step += 1

except KeyboardInterrupt:
    pass

model.save_pretrained('grpo_policy_model')
model.eval()
with torch.no_grad():
    acc = eval_multiplication(model, tokenizer, epochs=40, batch_size=64)
logger.info(f'Evaluation after training: {acc}')


end_time = time.time()
execution_time = end_time - start_time
logger.info(f'Total execution time in minutes: {execution_time/60:.2f}')
# %%

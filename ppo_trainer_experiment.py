# %%
from transformers import AutoModelForCausalLM, AutoTokenizer
from logging import getLogger
import logging
import os
import time

start_time = time.time()

logger = getLogger(__name__)
logfile_name = 'ppo-mul7.log'
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

# %%


# %%
def generate_completion(model, tokenizer, prompt):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

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
        max_new_tokens=256
    )

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return response

# %%
import re

def extract_answer(response, transform_fn = lambda x: x, nan_val = None)->str|None:
    ans = re.search('<answer>(.+)</answer>', response)
    if ans:
        try:
            return transform_fn(ans[1])
        except:
            return nan_val
    return nan_val

# %%
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
    
    acc = matches/tries
    format_errors /= tries
    wrong_answer = 1 - acc - format_errors
    return {
        'acc': acc,
        'format_errors': format_errors,
        'wrong_answer': wrong_answer
    }


# %% [markdown]
# ## Create the critic model
# 
# The same Qwen 0.5B model but with a different head to output a single value that will represent the expected value (reward and future rewards) of a given sequence of tokens.

# %%
import torch
from torch import nn

# %%
def get_critic_model():
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    critic = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    critic.lm_head = nn.Linear(896, 1, device=critic.device, dtype=torch.bfloat16)
    # Make the linear layer values close to zero so the initial values predictions are very close to zero too
    critic.lm_head.weight.data /= 896
    return critic


# %%
def make_rollouts(model, critic_model, simulations, initial_prompt: str, max_size = 256):
    input_tokens = tokenizer(initial_prompt, return_tensors='pt').input_ids
    # Use a single prompt to generate different responses
    input_tokens = input_tokens.repeat(simulations, 1).to(model.device)
    
    prompt_length = input_tokens.shape[1]
    is_terminal = torch.zeros((simulations, max_size + prompt_length))
    values = None
    # generate each response token in a batch and calculate the value for each of them
    for i in range(max_size):
        # TODO: Only calculate the response logits and values one and use those to do the optimization step
        with torch.no_grad():
            # Calculate the next token logits
            logits = model(input_tokens).logits[:,-1,:].to(torch.float) # [sims, vocab]

            # sample next action (next token)
            probs = torch.nn.functional.softmax(logits, dim=1)
            distribution = torch.distributions.Categorical(probs)
            action = distribution.sample() # [sims]
            # Concatenate to the prompt
            input_tokens = torch.cat((input_tokens, action.unsqueeze(1)), dim=1)

            # Check if the new token is the eos token
            if i > 0:
                is_terminal[:,prompt_length+i] = torch.max(is_terminal[:,prompt_length+i-1], (action.cpu() == tokenizer.eos_token_id))

            # Check if all the generations finished early
            if (is_terminal[:,prompt_length+i] == 1).all():
                is_terminal[:,prompt_length+i:] = 1
                values = critic_model(input_tokens[:,:-1]).logits[:,:,0] # Use the critic model to calculate the value for each state
                input_tokens = torch.cat((input_tokens, input_tokens[:,[prompt_length+i]].repeat(1, max_size-i-1)), dim=1)
                break

    if values == None:
        with torch.no_grad():
            values = critic_model(input_tokens[:,:-1]).logits[:,:,0] # Use the critic model to calculate the value for each state
    # Pad the generation with the eos token to the right
    generations = input_tokens.masked_fill(is_terminal.to(input_tokens.device) == 1, tokenizer.eos_token_id)
    # Values are shift to the right because the last token is seen as the action and not part of the state (tokens before last)
    return generations[:,prompt_length:], is_terminal[:,prompt_length:], values[:,prompt_length-1:], generations, prompt_length

# %%
def get_rewards(samples, is_terminal, correct_result):
    samples = samples.cpu()
    is_terminal = is_terminal.cpu()
    rewards = torch.zeros_like(samples, dtype=torch.float)
    # Extract the answer from the answer tag if any on each response
    samples = tokenizer.batch_decode(samples, skip_special_tokens=True)
    logger.debug(f'samples: {samples}')
    answer = torch.tensor([extract_answer(response, lambda x: int(x) if x.isnumeric() else torch.nan, torch.nan) for response in samples])

    eos_index = (is_terminal == 0).sum(dim=1)
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
    rewards[torch.arange(len(samples)), eos_index] += (1-wrong_format.to(torch.float32))*0.5
    # An additional 1 point of reward if the answer is correct
    rewards[torch.arange(len(samples)), eos_index] = answer_is_correct.to(torch.float32)
    logger.debug(f'Rewards: {rewards[torch.arange(len(samples)), eos_index]}')
    return rewards

# %%
def run_one_mul_simulation(model, critic_model, generations_num):
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
    inputs, is_terminal, values, complete_prompts, prompt_length = make_rollouts(model, critic_model, generations_num, texts[0])
    # Calculate the rewards for each response
    rewards = get_rewards(inputs, is_terminal, correct_result)
    return inputs, rewards, is_terminal, values, complete_prompts, prompt_length


# %%
def compute_advantages(rewards, is_terminal, values, gamma=1.0, gae_lambda=0.2):
    advantages = torch.zeros_like(rewards)
    # Find the longest response in the batch
    num_rollout_steps = torch.max((is_terminal==0).sum(1))
    num_rollout_steps = torch.min(num_rollout_steps, torch.tensor(rewards.shape[1]-1))

    # Initialize the GAE accumulator
    gae_running_value = 0
    # Accumulate the GAE values in reverse order, which allows us to efficiently
    # compute multi-step advantages
    is_not_terminal = 1 - is_terminal
    for t in reversed(range(num_rollout_steps+1)):
        # Get the value of the next step as predicted by the critic model
        if t == num_rollout_steps:
            next_value = 0
        else:
            next_value = values[:,t+1]
        # Check if the episode in current step has not ended yet
        if t == 0:
            episode_has_not_ended = 1
        else:
            episode_has_not_ended = is_not_terminal[:,t-1]
        episode_continues = is_not_terminal[:,t] # the episode ends a token after the eos token, so the eos token is a valid step

        # Calculate the temporal difference error:
        # Which is the difference between the current value of the state: values[:,t]
        # and a little more accurate estimate: rewards[:,t] + gamma*values[:,t+1]
        delta = rewards[:,t] + gamma * next_value * episode_continues - values[:,t]*episode_has_not_ended

        # The advantage is equal to the the temporal differece error plus the discounted gae_running_value.
        # The temporal differece error recovers the difference in values added by the actual action taken,
        # and the discounted gae_running_value represents the discounted advantages of future tokens.
        advantages[:,t] = delta + gamma * gae_lambda * episode_continues * gae_running_value
        # The GAE helps us balance between considering only the current TD error (gae_lambda=0),
        # or considering all the future TD errors equally (gae_lambda=1)
        gae_running_value = advantages[:,t]
    logger.debug(f'advantages: {advantages[:values.shape[0],:values.shape[1]]}')
    # Returns are an updated estimate of the value of the state, that considers the impact of the action taken (advatange)
    returns = advantages[:values.shape[0],:values.shape[1]] + values
    return advantages, returns

# %%
import numpy as np

def update_critic(critic, optimizer, update_epochs, minibatch_size, returns, is_terminal, complete_prompts, prompt_length, old_values=None, scheduler=None):
    value_clip = 0.2
    for epoch in range(update_epochs):
        batch_indices = np.arange(len(returns))
        np.random.shuffle(batch_indices)
        returns = returns.to(critic.device)
        is_terminal = is_terminal.to(critic.device)
        minibatches = len(returns) // minibatch_size

        ## Iterate in minibatches (random minibatch_size responses)
        for start in range(0, len(returns), minibatch_size):
            end = start + minibatch_size
            minibatch_indices = batch_indices[start:end]
            # Use the prompt+response to calculate the value
            # Shift the values so the value of a state only considers the previous tokens and not the actions (last token)
            minibatch_prompts = complete_prompts[minibatch_indices,:]
            minibatch_is_not_terminal = torch.cat((torch.ones(len(minibatch_prompts),1, device=critic.device), 1-is_terminal[minibatch_indices]), dim=1)[:,:returns.shape[1]]
            minibatch_values = (critic(minibatch_prompts).logits[:,:,0])[:,prompt_length-1:prompt_length-1+returns.shape[1]]
            if epoch == update_epochs-1:
                logger.debug(f'Values: {minibatch_values*minibatch_is_not_terminal}')
            minibatch_returns = returns[minibatch_indices]
            minibatch_old_values = old_values[minibatch_indices]
            clipped_value_diff = torch.clamp(
                minibatch_values - minibatch_old_values,
                -value_clip,
                value_clip,
            )
            clipped_value = minibatch_old_values + clipped_value_diff
            # The critic loss in the difference between tha values and the returns using MSE
            clipped_loss = ((clipped_value - minibatch_returns)**2*minibatch_is_not_terminal).sum() / minibatch_is_not_terminal.count_nonzero()
            logger.debug(f'loss: {clipped_loss.item()}')

            # Update the critic's weights
            optimizer.zero_grad()
            clipped_loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), 0.1) # Avoid large gradients
            optimizer.step()

    # Update the scheduler every epoch
    if scheduler:
        scheduler.step()

# %%
def update_policy(model, ref_model, optimizer, returns, is_terminal, advatanges, complete_prompts, prompt_length, generations, minibatch_size, update_epochs, scheduler=None):
    lower_clipped_threshold = 0.4
    upper_clipped_threshold = 3.0
    # lower_clipped_threshold = 0.8 # Safer update if the ref model is updated at very rl step
    # upper_clipped_threshold = 1.5 # Safer update if the ref model is updated at very rl step
    entropy_clip = 35.
    entro_loss_weight = 0.
    for epoch in range(update_epochs):
        batch_indices = np.arange(len(returns))
        np.random.shuffle(batch_indices)
        returns = returns.to(model.device)
        is_terminal = is_terminal.to(model.device)
        advatanges = advatanges.to(model.device)
        minibatches= len(returns) // minibatch_size

        ## Iterate in minibatches (random minibatch_size responses)
        for start in range(0, len(returns), minibatch_size):
            end = start + minibatch_size
            minibatch_indices = batch_indices[start:end]
            # Calculate the logits for the generated responses with the policy model
            logits = model(complete_prompts[minibatch_indices,:prompt_length+returns.shape[1]]).logits[:,prompt_length:]
            # Calculate the logits for the generated responses with the ref model
            with torch.no_grad():
                ref_logits  = ref_model(complete_prompts[minibatch_indices,:prompt_length+returns.shape[1]]).logits[:,prompt_length:]
            # Get the ids of the actual generated tokens
            completion_ids = generations[minibatch_indices,:returns.shape[1]].reshape(-1)
            actual_minibatch_size = len(logits)
            max_tokens = returns.shape[1]
            # Calculate the probabilities of each token generated with the policy and ref model
            probs = nn.functional.softmax(logits.reshape(actual_minibatch_size*max_tokens,-1), dim=1)
            probs_tokens = probs[torch.arange(len(completion_ids)), completion_ids]
            log_probs_sum = torch.log(probs_tokens)
            ref_probs = nn.functional.softmax(ref_logits.reshape(actual_minibatch_size*max_tokens,-1), dim=1)
            ref_probs_tokens = ref_probs[torch.arange(len(completion_ids)), completion_ids]
            log_ref_probs_sum = torch.log(ref_probs_tokens)
            # calculate the probability ratio as probs_policy_model/probs_ref_model
            probability_ratio = (log_probs_sum - log_ref_probs_sum).exp()

            # terminal state is shift to the right so the eos token that has the reward is taken in account
            minibatch_is_not_terminal = torch.cat((torch.ones(actual_minibatch_size,1, device=critic.device), 1-is_terminal[start:end]), dim=1)[:,:returns.shape[1]]
            minibatch_is_not_terminal = minibatch_is_not_terminal.reshape(-1)

            minibatch_advantages = advatanges[minibatch_indices,:returns.shape[1]].reshape(-1) * minibatch_is_not_terminal

            # Normalize advantages
            minibatch_advantages = (minibatch_advantages - minibatch_advantages.mean()) / (minibatch_advantages.std() + 1e-8)

            # The policy loss is to maximize the probability_ratio times the advantages
            loss = probability_ratio * minibatch_advantages
            logger.debug(f'probability ratio: mean - {probability_ratio.mean()}, min - {probability_ratio.min()}, max: {probability_ratio.max()}')
            # Clipped loss: Only considers the probability_ration change between a reasonable range
            clipped_loss = torch.clamp(probability_ratio, 1-lower_clipped_threshold, 1+upper_clipped_threshold) * minibatch_advantages

            # Print tokens to be prioritized and tokens to be deprioritized
            prioritized_tokens, priotized_indices = torch.sort(clipped_loss, dim=-1)
            logger.debug(f'prioritized_tokens: {prioritized_tokens.shape, prioritized_tokens[-3:]} {tokenizer.decode(completion_ids[priotized_indices][-3:])}')
            logger.debug(f'deprioritized_tokens: {prioritized_tokens.shape, prioritized_tokens[:3]} {tokenizer.decode(completion_ids[priotized_indices][:3])}')

            # Be pessimistic in the loss calulation, and add the minus to perform optimization minimizing the loss
            loss = -torch.min(loss, clipped_loss).sum() /  minibatch_is_not_terminal.count_nonzero()

            # Add entropy loss
            # logits_distribution = torch.distributions.categorical.Categorical(probs=probs_tokens.reshape(-1, probs_tokens.size(-1)))
            # ent = (logits_distribution.entropy().reshape(len(probs_tokens), -1) * minibatch_is_not_terminal).sum() /  minibatch_is_not_terminal.count_nonzero()
            # entro_loss = torch.abs(ent - entropy_clip)
            # loss += entro_loss_weight * entro_loss 
            logger.debug(f'loss: {loss.item()}')

            # Update the policy weights
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.1) # Avoid large gradients
            optimizer.step()

    # Update the scheduler every epoch
    if scheduler:
        scheduler.step()

# %% [markdown]
# ## Putting everything together

# %%
from peft import LoraConfig, get_peft_model 

model, tokenizer = get_policy_model()
ref_model, _ = get_policy_model()
critic = get_critic_model()

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

# Use LoRA to finetune the policy and critic model
model = get_peft_model(model, config)
critic = get_peft_model(critic, config)
# Update the weights on the new added linear layer of the critic model
critic.lm_head.weight.requires_grad = True
critic.lm_head.bias.requires_grad = True

ref_model.eval()

# %%
max_steps = 250*8
sims_per_prompt = 8
rl_steps = max_steps // sims_per_prompt
minibatch_size = 4
update_epochs = 4

model.train()
critic.train()

from torch.optim.lr_scheduler import LinearLR
optimizer1 = torch.optim.Adam(critic.parameters(), lr=1e-7, betas=(0.9, 0.999))
optimizer2 = torch.optim.Adam(model.parameters(), lr=5e-7, betas=(0.9, 0.999))
scheduler1 = LinearLR(optimizer1, total_iters=25)
scheduler2 = LinearLR(optimizer2, total_iters=25)

import copy
# Training loop
try:
    for rl_step in range(rl_steps):
        generations, rewards, is_terminal, values, complete_prompts, prompt_length = run_one_mul_simulation(model, critic, sims_per_prompt)
        advatanges, returns = compute_advantages(rewards, is_terminal, values.cpu())

        logger.debug(f'rl_step: {rl_step+1:,}')
        logger.debug(f'Learning rate: {scheduler1.get_lr()}')
        logger.debug('Updating critic')
        update_critic(critic, optimizer1, update_epochs, minibatch_size, returns, is_terminal, complete_prompts, prompt_length, old_values=values, scheduler=scheduler1)
        logger.debug('Updating policy')
        update_policy(model, ref_model, optimizer2, returns, is_terminal, advatanges, complete_prompts, prompt_length, generations, minibatch_size, update_epochs, scheduler=scheduler2)
        # ref_model = copy.deepcopy(model) # Update the ref model at every iteration
except KeyboardInterrupt:
    pass

model.save_pretrained('policy_model')
critic.save_pretrained('critic_model')
model.eval()
with torch.no_grad():
    acc = eval_multiplication(model, tokenizer, epochs=40, batch_size=64)
logger.debug(f'Evaluation: {acc}')


end_time = time.time()
execution_time = end_time - start_time
logger.info(f'Total execution time in minutes: {execution_time/60:.2f}')
# %%

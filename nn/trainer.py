import numpy as np
import concurrent.futures

def get_grads(model, X, y, loss) -> list[float]:
    grads = []

    y_pred = [model(x) for x in X]
    loss = loss(y, y_pred)

    loss.backward()

    for p in model.parameters():
        grads.append(p.grad)
    
    for p in model.parameters():
        p.grad = 0.0

    return grads, loss.data

def train_GD(loss_func, model, X, y, epochs, lr, batch_size: int = None) -> list[float]:
    loss_history = []

    for epoch in range(epochs):
        if batch_size is not None:
            p = np.random.permutation(len(X))
            X_shuffled, y_shuffled = X[p], y[p]

            with concurrent.futures.ProcessPoolExecutor() as executor:
                X_batched = [X_shuffled[start:start + batch_size] for start in range(0, len(X_shuffled), batch_size)]
                y_batched = [y_shuffled[start:start + batch_size] for start in range(0, len(y_shuffled), batch_size)]
                results = executor.map(get_grads, [model] * len(y_batched), X_batched, y_batched, [loss_func] * len(y_batched))
            
            # Grads average
            grads = []
            losses = []
            for result in results:
                grad, loss = result
                grads.append(grad)
                losses.append(loss)

            grads = np.mean(grads, axis=0)
            loss = np.mean(losses)

            for p, g in zip(model.parameters(), grads):
                p.data -= lr * g
        else:
            y_pred = [model(x) for x in X]
            loss = loss_func(y, y_pred)

            # Calculate the grads
            loss.backward()

            loss = loss.data

            # Gradient descent
            for p in model.parameters():
                p.data -= lr * p.grad
                
            # Reset the grads
            for p in model.parameters():
                p.grad = 0.0

        print(f'Epoch {epoch+1}/{epochs} | Loss: {loss:.5f}')
        loss_history.append(loss)
    
    return loss_history
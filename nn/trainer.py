def train_GD(loss_func, model, X, y, epochs, lr) -> list[float]:
    loss_history = []

    for epoch in range(epochs):
        y_pred = [model(x) for x in X]
        loss = loss_func(y, y_pred)

        # Calculate the grads
        loss.backward()

        # Gradient descent
        for p in model.parameters():
            p.data -= lr * p.grad
            
        # Reset the grads
        for p in model.parameters():
            p.grad = 0.0

        print(f'Epoch {epoch+1}/{epochs} | Loss: {loss.data:.5f}')
        loss_history.append(loss.data)
    
    return loss_history
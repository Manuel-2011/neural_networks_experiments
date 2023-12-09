import numpy as np

def mse_loss(y: list[float], y_pred: list[float]) -> float:
    return np.mean([(yt - yp)**2 for yt, yp in zip(y, y_pred)])
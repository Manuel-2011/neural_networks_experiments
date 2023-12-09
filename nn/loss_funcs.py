import numpy as np
import sys

sys.path.append('..')
from nn.engine import Value

def mse_loss(y: list[float], y_pred: list[Value]) -> Value:
    return np.mean([(yt - yp)**2 for yt, yp in zip(y, y_pred)])

def binary_cross_entropy_loss(y: list[float], y_pred: list[Value]) -> Value:
    def cross_entropy(y: float, yp: Value) -> Value:
        return - ((y * yp.log()) + (1 - y) * (1 - yp).log())
    
    return np.mean([cross_entropy(yt, yp) for yt, yp in zip(y, y_pred)])
import numpy as np
from typing import Literal
import math
from abc import ABC, abstractmethod

from nn.engine import Value

Activations = Literal['tanh', 'relu', 'sigm'] | None

class Neuron:

    def __init__(self, nin, activation: Activations = None):
        self.init_strategy =  lambda : np.random.uniform(-1, 1)
        self.nin = nin

        if activation == 'relu':
            self.init_strategy = self._he_init_strategy
        elif activation == 'sigm' or activation == 'tanh':
            self.init_strategy = self._xavier_init_strategy

        self.w = [Value(self.init_strategy(nin)) for _ in range(nin)]
        self.b = Value(self.init_strategy(nin))
        self.activation = activation

    def __call__(self, x):
        output: Value = sum(w * x for w, x in zip(self.w, x)) + self.b

        if self.activation == 'tanh':
            return output.tanh()
        elif self.activation == 'relu':
            return output.relu()
        elif self.activation == 'sigm':
            return output.sigmoid()
        elif self.activation == None:
            return output

    def _he_init_strategy(self, input_size: int):
        return np.random.normal(0, math.sqrt(2.0 / input_size))
    
    def _xavier_init_strategy(self, input_size: int):
        return np.random.uniform(-1.0 / math.sqrt(input_size), 1.0 / math.sqrt(input_size))
    
    def parameters(self):
        return self.w + [self.b]
    
    def __repr__(self) -> str:
        return  f'Neuron(nin={self.nin},init_strategy={self.init_strategy},activation={self.activation})'

    
class BaseLayer(ABC):
    @abstractmethod
    def __call__(self, x) -> list[float] | float:
        pass

    @abstractmethod
    def parameters(self) -> list[float]:
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass

    
class Layer(BaseLayer):

    def __init__(self, nin, nout, activation: Activations, **kwargs):
        self.nin = nin
        self.nout = nout
        self.activation = activation
        self.neurons = [Neuron(nin, activation=activation, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        outs =  [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]
    
    def __repr__(self) -> str:
        return f'Layer(nin={self.nin},nout={self.nout},activation={self.activation})'
    
class MLP:

    def __init__(self, nin, nouts: list[int], activation: Activations = 'relu', out_activation: Activations = None, **kwargs):
        sizes = [nin] + nouts
        self.layers = [Layer(sizes[i], sizes[i + 1], activation=activation if i < len(nouts) - 1 else out_activation, **kwargs) for i in  range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)     
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    
class DropoutLayer(BaseLayer):

    def __init__(self, proportion = 0.2, **kwargs):
        self.proportion = proportion
        self.training = True

    def __call__(self, x):
        if not self.training:
            return x
        
        mask = np.random.choice(range(len(x)), size=round(len(x) * (1 - self.proportion)), replace=False)
        outs =  [x[i] if i in mask else 0.0 for i in range(len(x))]
        return outs
    
    def parameters(self):
        return []
    
    def __repr__(self) -> str:
        return f'Dropout(proportion={self.proportion})'
    
class Sequential:
    def __init__(self, layers: list[BaseLayer]):
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)     
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    
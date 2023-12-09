import numpy as np
from typing import Literal

from nn.engine import Value

Activations = Literal['tanh', 'relu', 'sigm'] | None

class Neuron:

    def __init__(self, nin, activation: Activations = None):
        self.w = [Value(np.random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(np.random.uniform(-1, 1))
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
    
    def parameters(self):
        return self.w + [self.b]
    
class Layer:

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
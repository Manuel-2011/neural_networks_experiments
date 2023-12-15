import math

class Value:
  
  def __init__(self, data, _children=(), _op='', label=''):
    self.data = data
    self.grad = 0.0
    self._prev = set(_children)
    self._op = _op
    self.label = label
    self.grad = 0.0
    self._backward = self._default_backward

  def _default_backward(self):
    return None

  def __repr__(self):
    return f"Value(data={self.data})"
  
  def __add__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data + other.data, (self, other), '+')

    def _backward():
      self.grad += 1.0 * out.grad
      other.grad += 1.0 * out.grad
    out._backward = _backward
    return out
  
  def __radd__(self, other):
    return self + other

  def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data * other.data, (self, other), '*')

    def _backward():
      self.grad += other.data * out.grad
      other.grad += self.data * out.grad
    out._backward = _backward
    return out
  
  def __rmul__(self, other):
    return self * other
  
  def __truediv__(self, other):
    other = other if isinstance(other, Value) else float(other)
    return self * (other ** -1)
  
  def __rtruediv__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    return other * (self ** -1)
  
  def __neg__(self):
    return self * -1
  
  def __sub__(self, other):
    return self + (-other)
  
  def __rsub__(self, other):
    return other + (-self)
  
  def __pow__(self, other):
    assert isinstance(other, (int, float)), "only supporting int/float powers for now"
    out = Value(self.data ** other, (self,), f'**{other}')

    def _backward():
      self.grad += (other * self.data ** (other - 1)) * out.grad
    out._backward = _backward
    return out
  
  def exp(self):
    out = Value(math.exp(self.data), (self,), 'exp')

    def _backward():
      self.grad += out.data * out.grad
    out._backward = _backward
    return out
  
  def log(self, base = 2):
    out = Value(math.log(self.data, base), (self,), 'log')

    def _backward():
      self.grad += (1 / (self.data * math.log(base))) * out.grad
    out._backward = _backward
    return out
  
  def tanh(self):
    out = Value(math.tanh(self.data), (self,), 'tanh')

    def _backward():
      self.grad += (1 - out.data ** 2) * out.grad
    out._backward = _backward
    return out
  
  def relu(self):
    out = Value(max(0, self.data), (self,), 'relu')

    def _backward():
      self.grad += (out.data > 0) * out.grad
    out._backward = _backward
    return out
  
  def sigmoid(self):
    out = Value(1 / (1 + math.exp(-self.data)), (self,), 'sigm')

    def _backward():
      self.grad += out.data * (1 - out.data) * out.grad
    out._backward = _backward
    return out
  
  def backward(self):
    topo = []
    visited = set()
    def topological_sort(v):
      if v not in visited:
        visited.add(v)
        for child in v._prev:
          topological_sort(child)
        topo.append(v)

    topological_sort(self)

    self.grad = 1.0
    for node in reversed(topo):
      node._backward()
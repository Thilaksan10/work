from activation import Activation
import numpy as np

class Tanh(Activation):
    def __init__(self):
        tanh = lambda x: np.tanh(x)
        tanh_prime = lambda x: 1 - np.tanh(x) ** 2
        super().__init__(tanh, tanh_prime)

class ReLU(Activation):
    def __init__(self):
        relu = lambda x: np.maximum(x, 0)
        relu_prime = lambda x: x > 0
        super().__init__(relu, relu_prime)

'''
class Softmax(Activation):
    def __init__(self):
        softmax = lambda x: (np.exp(x - np.max(x, axis=1, keepdims=True))) / (np.sum(np.exp(x - np.max(x, axis=1, keepdims=True)), axis=1, keepdims=True))
        softmax_prime = 0
        super().__init__(softmax, softmax_prime)
'''
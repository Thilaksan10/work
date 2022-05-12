from dense import Dense
from activations import ReLU, Tanh
from losses import mean_squared_error ,mean_squared_error_prime
import numpy as np

X = np.reshape(
    [
        [0, 0], 
        [0, 1], 
        [1, 0], 
        [1, 1]
    ], 
    (4, 2, 1)
)

Y = np.reshape(
    [
        [0], 
        [1], 
        [1], 
        [0]
    ], 
    (4, 1, 1)
)

network = [
    Dense(2, 3),
    ReLU(),
    Dense(3, 5),
    ReLU(),
    Dense(5, 3),
    ReLU(),
    Dense(3, 1),
    Tanh()
]

epochs = 10000
learning_rate = 0.001

#train
for e in range(epochs):
    error = 0
    for x, y in zip(X, Y):
        #forward
        output = x
        for layer in network:
            output = layer.forward(output)

        #error
        error += mean_squared_error(y, output)

        #backward
        gradient = mean_squared_error_prime(y, output)
        for layer in reversed(network):
            gradient = layer.backward(gradient, learning_rate)

    error /= len(X)
    print('%d/%d, error=%f' % (e + 1, epochs, error))
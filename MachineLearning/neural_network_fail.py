import numpy as np
from numpy.ma.core import exp

class Layer():
    def __init__(self,n_inputs,n_neurons,activation):
        np.random.seed(0)
        self.weights = 0.1 * np.random.randn(n_inputs,n_neurons)
        self.biases = np.random.rand(1,n_neurons)-0.5
        self.activation = activation

    def forward(self, inputs):
        temp = np.dot(inputs,self.weights) + self.biases
        if(self.activation == 'binary_step'):
            self.output = self.binary_step(temp)
        elif(self.activation == 'sigmoid'):
            self.output = self.sigmoid(temp)
        elif(self.activation == 'tanh'):
            self.output = self.tanh(temp)
        elif(self.activation == 'relu'):
            self.output = self.relu(temp)
        elif(self.activation == 'softmax'):
            self.output = self.softmax(temp)
        elif(self.activation == 'maxout'):
            self.output = self.maxout(temp)
        else:
            self.output = temp

    def back_propagation(self,inputs,prev_input,Y,alpha):
        m = Y.size
        if(self.activation == 'binary_step'):
            dZ = self.derivative_binary_step(inputs)
        elif(self.activation == 'sigmoid'):
            dZ = self.derivative_sigmoid(inputs)
        elif(self.activation == 'tanh'):
            dZ = self.derivative_tanh(inputs)
        elif(self.activation == 'relu'):
            dZ = self.derivative_relu(inputs)
        elif(self.activation == 'softmax'):
            dZ = self.derivative_softmax(inputs,Y)
        elif(self.activation == 'maxout'):
            pass
        else:
            dZ = self.derivative_identity(inputs)

        #print("dZ: ",dZ.shape)
        #print("input: ",inputs.shape)
        #print("prev_input: ",prev_input.shape)
        dW = 1 / m * np.dot(dZ.T,prev_input)
        #print("dW: ",dW.shape)
        db = 1 / m * np.sum(dZ)
        #print("db: ",dW.shape)

        self.weights = self.weights - alpha * dW.T
        self.biases = self.biases - alpha * db
    
    def identity(self,inputs):
        return inputs

    def derivative_identity(self,inputs):
        return 1

    def binary_step(self,inputs):
        return np.heaviside(inputs,1)

    def derivative_binary_step(self,inputs):
        return 0

    def sigmoid(self,inputs):
        return 1 / (1 + np.exp(-1 * inputs))

    def derivative_sigmoid(self,inputs):
        return self.sigmoid(inputs) * (1 - self.sigmoid(inputs))

    def tanh(self, inputs):
        return (np.exp(inputs) - np.exp(-1 * inputs)) / (np.exp(inputs) + np.exp(-1 * inputs))

    def derivative_tanh(self,inputs):
        return 1 - self.tanh(inputs) ** 2

    def relu(self,inputs):
        return np.maximum(0,inputs)

    def derivative_relu(self,inputs):
        return inputs > 0

    def softmax(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        return exp_values / np.sum(exp_values, axis=1, keepdims=True)

    def derivative_softmax(self, inputs, Y):
        one_hot_Y = np.zeros((Y.size, Y.max() + 1))
        one_hot_Y[np.arange(Y.size), Y] = 1
        one_hot_Y = one_hot_Y.T
        return (inputs.T - one_hot_Y).T

    def maxout(self, inputs):
        return np.max(inputs)

class NeuralNetwork():
    def __init__(self):
        self.layers = []

    def add(self,layer):
        self.layers.append(layer)

    def calculate_loss(self, output, y):
        sample_losses = self.categorical_cross_entropy(output,y)
        data_loss = np.mean(sample_losses)
        return data_loss

    def categorical_cross_entropy(self, y_prediction, y_true):
        samples = len(y_prediction)
        y_prediction_clipped = np.clip(y_prediction, 1e-7, 1-1e-7)
        if len(y_true.shape) == 1:
            correct_confidences = y_prediction_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_prediction_clipped * y_true, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
    
    def get_predictions(self,output):
        print(output)
        return np.argmax(output,0)

    def get_accuracy(self,predictions, Y):
        print(predictions, Y)
        return np.sum(predictions == Y) / Y.size

    def gradient_descent(self, X, Y, iterations, alpha):
        for i in range(iterations):
            x = X
            for layer in self.layers:
                #print(layer.weights)
                layer.forward(x)
                x = layer.output 
            #print(self.layers[-1].output)
            if i % 5 == 0:
                print("Iteration: ", i)
                print('Loss: ', self.calculate_loss(self.layers[-1].output,Y))
                prediction = self.get_predictions(self.layers[-1].output[0:2])
                print('prediction: ', prediction)
                print("Accuracy: ", round(self.get_accuracy(self.get_predictions(self.layers[-1].output), Y)*100,8), '%')
            for prev, layer in zip(reversed(self.layers[:-1]),reversed(self.layers)):
                #print(i)
                layer.back_propagation(layer.output,prev.output,Y,alpha)
        



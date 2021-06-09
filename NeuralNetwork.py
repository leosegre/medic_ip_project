import numpy as np
import cv2
import main

class NeuralNetwork():

    def __init__(self, learning_rate, f1, f2):
        self.learning_rate = learning_rate
        self.W1 = np.random.normal(0, 1, (1024, main.nn_hdim))
        self.b1 = np.random.normal(0, 1, (1, main.nn_hdim))
        self.W2 = np.random.normal(0, 1, (main.nn_hdim, 1))
        self.b2 = np.random.normal(0, 1, (1, 1))
        self.f1 = f1
        self.f2 = f2
        self.z1 = None
        self.a1 = None
        self.z2 = None
        self.a2 = None
        self.loss = None

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def sigmoid_derivative(self, z):
        return self.sigmoid(z)* (1-self.sigmoid(z))


    def tanh_derivative(self, z):
       return 1/(np.cosh(z)**2)

    def tanh(self, x):
        return np.tanh(x)

    def relu(self, x):
        return np.maximum(0, x)

    # def apply_tanh(self, z):
        # preform_vectorized_tanh = np.vectorize(self.tanh)
        # result = preform_vectorized_tanh(z)
        # return result

    # def apply_sigmoid(self, z):
    #     preform_vectorized_sigmoid = np.vectorize(self.sigmoid)
    #     result = preform_vectorized_sigmoid(z)
    #     return result

    def calculate_linear_combination(self, input, weights, bias):
        weights = np.tile(weights, (input.shape[0], 1, 1))
        w_x = np.sum(weights*np.expand_dims(input, axis=2), axis=1)
        z = w_x + bias
        return z

    def apply_activation(self, activation_function, x):
        if activation_function == "tanh":
            return self.tanh(x)
        elif activation_function == "sigmoid":
            return self.sigmoid(x)
        elif activation_function == "relu":
            return self.relu(x)

    def loss_function(self, predicted, labels):
        return (((labels-predicted)**2)/2).mean()

    def forward_pass(self, input, labels):
        self.z1 = self.calculate_linear_combination(input, self.W1, self.b1)
        self.a1 = self.apply_activation(self.f1, self.z1)
        self.z2 = self.calculate_linear_combination(self.a1, self.W2, self.b2)
        self.a2 = self.apply_activation(self.f2, self.z2)
        self.loss = self.loss_function(self.a2, labels)









import numpy as np
import cv2 as cv

class NeuralNetwork():

    def __init__(self, learningrate):
        self.learningrate = learningrate

    def InitilaizeWeigths(self, dimensions, neuron):
        #for the first layer - dimensions = (unknownx1024)
        #for the second layer - dimensions = (1xunknown)
        W1 = np.random.normal(0, 1/neuron, (dimensions[0], dimensions[1]))
        return W1

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def tanh(self, x):
        return np.tanh(x)

    def apply_tanh(self, z):
        preform_vectorized_tanh = np.vectorize(self.tanh)
        result = preform_vectorized_tanh(z)
        return result

    def apply_sigmoid(self, z):
        preform_vectorized_sigmoid = np.vectorize(self.sigmoid)
        result = preform_vectorized_sigmoid(z)
        return result

    def calculate_linear_combination(self, input, weights, bias):
        w_x = np.dot(weights, input)
        z = w_x + bias
        return z





import numpy as np
import cv2
import main

class NeuralNetwork():
##W1 = 1024X nn_hdim
## w2 = nn.hdim X 1
##b1= 1Xnn_hdim
##b2=1X1
##z1 = 1X nn_hdim
##z2= 1X1

    def __init__(self, learning_rate, f1, f2, sd_init, sd_init_w2):
        self.learning_rate = learning_rate
        self.W1 = np.random.normal(0, sd_init, (1024, main.nn_hdim))
        self.b1 = np.random.normal(0, sd_init, (1, main.nn_hdim))
        self.W2 = np.random.normal(0, sd_init_w2, (main.nn_hdim, 1))
        self.b2 = np.random.normal(0, sd_init, (1, 1))
        self.f1 = f1
        self.f2 = f2
        self.z1 = None
        self.a1 = None
        self.z2 = None
        self.a2 = None
        self.loss = None
        self.delta_1= None
        self.delta_2 = None
        self.accuracy = 0

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def sigmoid_derivative(self, z):
        return self.sigmoid(z)* (1-self.sigmoid(z))

    def tanh_derivative(self, z):
       return 1/(np.cosh(z)**2)

    def relu_derivative(self,z):
        y = (z > 0) * 1
        return y

    def tanh(self, z):
        return np.tanh(z)

    def relu(self, z):
        return np.maximum(0,z)

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

    def apply_activation_derivative(self, activation_function, x):
        if activation_function == "tanh":
            return self.tanh_derivative(x)
        elif activation_function == "sigmoid":
            return self.sigmoid_derivative(x)
        elif activation_function == "relu":
            return self.relu_derivative(x)

    def loss_function(self, predicted, labels):
        predicted = predicted.squeeze()
        loss = (((labels-predicted)**2)/2).mean()
        return loss

    def loss_function_derivative(self, predicted, labels):
        return predicted-labels

    def calculate_accuracy(self, labels):
        self.accuracy = (np.round(self.a2).squeeze() == labels).mean()


    def forward_pass(self, input, labels):
        self.z1 = self.calculate_linear_combination(input, self.W1, self.b1)
        self.a1 = self.apply_activation(self.f1, self.z1)
        self.z2 = self.calculate_linear_combination(self.a1, self.W2, self.b2)
        self.a2 = self.apply_activation(self.f2, self.z2)
        self.loss = self.loss_function(self.a2, labels)

    def backward_pass(self, labels):
        labels = np.expand_dims(labels, axis=1)
        self.delta_2 = self.loss_function_derivative(self.a2, labels) *\
                       self.apply_activation_derivative(self.f2, self.z2)
        self.delta_1 = self.calculate_linear_combination(self.delta_2, self.W2.transpose(), 0) *\
                       self.apply_activation_derivative(self.f1, self.z1)

    def compute_gradient(self, input):
            self.b1 -= self.delta_1.mean(axis=0) * self.learning_rate
            self.b2 -= self.delta_2.mean(axis=0) * self.learning_rate
            self.W1 -= self.learning_rate * np.mean(np.expand_dims(input, 2) * \
                        np.expand_dims(self.delta_1, 1), axis=0)
            # print((np.expand_dims(input.mean(axis = 0),1) * np.expand_dims(self.delta_1.mean(axis = 0),0)).shape)
            self.W2 -= self.learning_rate * np.mean(np.expand_dims(self.a1, 2) * \
                        np.expand_dims(self.delta_2, 1), axis=0)
            # print((np.expand_dims(self.a1.mean(axis=0),1)*np.expand_dims(self.delta_2.mean(axis=0),0)).shape)

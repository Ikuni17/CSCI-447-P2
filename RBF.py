import random
import math
import numpy as np

class RBF:
    sigma = 0.1

    def __init__(self, num_basis, train_in):
        self.train_in = []
        self.train_out = []
        self.weights = []
        self.centers = []
        self.sigmas = []
        # print(train_in)
        for x in train_in:
            self.train_in.append(x[:len(x) - 1])
            self.train_out.append(x[len(x) - 1])

        # initialize weight and centers array with random values
        for i in range(num_basis):
            # Put random point from training data as center
            self.centers.append(self.train_in[random.randint(0, len(self.train_in) - 1)][:len(self.train_in[0])-1])
            self.weights.append(random.uniform(0, 100))
            self.sigmas.append(random.uniform(0, 0.3))

    def train(self):
        out = self.gradient_descent(RBF.get_outputs(self.train_in, self.weights, self.centers), self.train_out, self.weights, 0.001, 100)
        self.weights = out[0]
        # print(self.weights)
        output = self.hypothesis(self.train_in)

        # for i in range(len(output)):
        # print('Expected out: {0}, Actual out: {1}'.format(self.train_out[i], output[i]))

    def hypothesis(self, data_in):
        output = []
        for datapoint in data_in:
            value = 0
            for i in range(len(self.weights)):
                value += self.weights[i] * RBF.gaussian_function(datapoint, self.centers[i], RBF.sigma)
            output.append(value)
        return output

    @staticmethod
    def gaussian_function(datapoint, center, sigma):
        dist = np.linalg.norm(np.array(datapoint) - np.array(center))
        return math.exp(-(dist ** 2) / 2 * sigma ** 2)

    @staticmethod
    def get_outputs(data_in, weights, centers):
        output = []
        for datapoint in data_in:
            value = []
            for i in range(len(weights)):
                value.append(RBF.gaussian_function(datapoint, centers[i], RBF.sigma))
            output.append(value)
        return output

    @staticmethod
    def gradient_descent(inputs, output, weights, alpha, num_iter):
        y = np.array(output)
        theta = np.array(weights)
        x = np.array(inputs)
        # print(theta)
        x_trans = x.transpose()
        costs = []
        for i in range(num_iter):
            hypothesis = np.dot(x, theta)
            # print(hypothesis[2])
            loss = hypothesis - y
            cost = np.sum(loss ** 2)
            costs.append(cost)
            gradient = np.dot(x_trans, loss) / (np.shape(loss))
            theta = theta - alpha * gradient
        return (theta.tolist(), costs)

'''
CSCI 447: Project 2
October 5, 2017
'''

import numpy as np
import random


class node():
    number = 0
    use_linear_act = True

    def __init__(self, number, use_linear):
        self.number = number
        self.use_linear_act = use_linear
        self.value = 0
        self.output = 0
        self.weights = []
        self.historical_weights = []

    def activation_function(self, input_num):
        c = 1
        if self.use_linear_act is True:
            # Linear Activation Function
            return c * input_num
        else:
            # Tanh Activation Function
            return np.tanh(input_num)

    def gaussian_function(self, input_num):
        # return math.e **
        pass

    def initialize_weights(self, size):
        for i in range(size):
            self.weights.append(random.random())

    def append_zero_weights(self, number):
        for i in range(number):
            self.weights.append(0)


def node_test():
    network = []
    for i in range(10):
        network.append(node(i, True))

    for i in len(network):
        for j in range(i + 1, len(network)):
            network[i].set_connections(network[j].number)

    print(network[0].connections)
    print(network[1].connections)

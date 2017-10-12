'''
CSCI 447: Project 2
October 5, 2017
'''

import math

class node():
    number = 0
    use_linear_act = True

    def __init__(self, number, use_linear, center):
        self.center = center
        self.number = number
        self.use_linear_act = use_linear

    def activation_function(self, input_num):
        c = 2
        if self.use_linear_act is True:
            # Linear Activation Function
            return c * input_num
        else:
            # Tanh Activation Function
            return math.tanh(input_num)

    def gaussian_function(self, input_num):
        #return math.e **
        pass




def node_test():
    network = []
    for i in range(10):
        network.append(node(i, True))

    for i in len(network):
        for j in range(i + 1, len(network)):
            network[i].set_connections(network[j].number)

    print(network[0].connections)
    print(network[1].connections)

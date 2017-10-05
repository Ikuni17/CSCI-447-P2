'''
CSCI 447: Project 2
October 5, 2017
'''

class Node():
    number = 0
    use_linear_act = True

    def __init__(self, number, use_linear):
        self.number = number
        self.use_linear_act = use_linear

    def activation_function(self, input):
        output = None

        if self.use_linear_act is True:
            # Linear Activation Function
            pass
        else:
            # Sigmoid Activation Function
            pass

        return output

    def gaussian_function(self, input):
        output = None

        return output

def node_test():
    network = []
    for i in range(10):
        network.append(Node(i, True))

    for i in len(network):
        for j in range(i+1, len(network)):
            network[i].set_connections(network[j].number)

    print(network[0].connections)
    print(network[1].connections)
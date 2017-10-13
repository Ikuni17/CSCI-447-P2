import node
import random
import numpy as np
import rosen_generator


class MLP:
    # List of input node objects
    input_layer = []
    # List of lists which contain nodes for a hidden layer
    hidden_layers = [[]]
    # List of output nodes
    output_layer = []
    momentum = None
    converged = False
    # Matrix of weights where the connect between nodes i and j will be at index [i][j]. The third list contains
    # historical weights for that connection, where the 0th index is the current weight.
    weights = [[[]]]

    def __init__(self, num_inputs, num_layers, nodes_per_layer, num_outputs, momentum, input_vectors, outputs):
        node_id_counter = 0
        self.input_vectors = input_vectors
        self.outputs = outputs

        # Initialize input layer
        for i in range(num_inputs):
            self.input_layer.append(node.node(node_id_counter, True, 0))
            self.input_layer[i].value = input_vectors[i]
            node_id_counter += 1

        '''for i in range(len(self.input_layer)):
            print(self.input_layer[i].value)'''

        # Initialize any hidden layers
        for i in range(num_layers):
            if i != 0:
                self.hidden_layers.append([])
            for j in range(nodes_per_layer[i]):
                self.hidden_layers[i].append(node.node(node_id_counter, True, 0))
                node_id_counter += 1

        # Initialize output layer
        output_vector = outputs[:num_inputs]
        for i in range(num_outputs):
            self.output_layer.append(node.node(node_id_counter, True, 0))
            self.output_layer[i].value = output_vector
            node_id_counter += 1

        # Initialize all weights to zero
        self.weights = [[[0] for i in range(node_id_counter)] for j in range(node_id_counter)]
        # Add a random weight between 0 and 1 if it isn't the connection to itself
        for i in range(node_id_counter):
            for j in range(node_id_counter):
                if i == j:
                    continue
                else:
                    self.weights[i][j][0] = random.random()

    def train(self):
        # Train the first layer from the input layer
        for hidden_node in self.hidden_layers[0]:
            for input_node in self.input_layer:
                hidden_node.value = np.dot(input_node.value, self.weights[hidden_node.number][input_node.number][0])
                print(hidden_node.value)

    def sum_weights(self):
        pass

    def backprop(self):
        pass

    def hypothesis_of(self, testing_data):
        pass

    def calculate_error(self):
        pass

    def output_results(self):
        pass

    def print_network(self):
        print("Inputs: ", end="")
        for i in range(len(self.input_layer)):
            print("{0}".format(self.input_layer[i].number), end=', ')

        print("\n\nHidden Layers: ")
        for i in range(len(self.hidden_layers)):
            print("\nLayer {0}: ".format(i), end="")
            for j in range(len(self.hidden_layers[i])):
                print("{0}".format(self.hidden_layers[i][j].number), end=', ')

        print("\n\nOutputs: ", end="")
        for i in range(len(self.output_layer)):
            print("{0}".format(self.output_layer[i].number), end=', ')


def main():
    rosen_in = rosen_generator.generate(input_type=0, dimensions=2, num_data_points=1000)

    input_vectors = []
    outputs = []
    dimension = len(rosen_in[0]) - 1
    for i in range(len(rosen_in)):
        input_vectors.append(rosen_in[i][:dimension])
        outputs.append(rosen_in[i][dimension])

    # print(rosen_in)
    # print(input_vectors)
    # print(outputs)
    mlp_network = MLP(5, 1, [5, 5], 1, 0, input_vectors, outputs)
    mlp_network.train()



main()

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
    converged = False
    # Matrix of weights where the connect between nodes i and j will be at index [i][j]. The third list contains
    # historical weights for that connection, where the 0th index is the current weight.
    weights = [[[]]]

    def __init__(self, num_inputs, num_layers, nodes_per_layer, num_outputs, input_vectors, outputs):
        # Keep track of the total amount of nodes to construct the weight matrix
        node_id_counter = 0
        self.input_vectors = input_vectors
        self.outputs = outputs

        # Initialize input layer
        for i in range(num_inputs):
            self.input_layer.append(node.node(node_id_counter, True, 0))
            # Add an input vector to the node
            self.input_layer[i].value = input_vectors[i]
            node_id_counter += 1

        # Initialize any hidden layers
        for i in range(num_layers):
            if i != 0:
                # Add a new list to represent a layer
                self.hidden_layers.append([])
            # Add the amount of nodes specified for this layer
            for j in range(nodes_per_layer[i]):
                self.hidden_layers[i].append(node.node(node_id_counter, False, 0))
                node_id_counter += 1

        # Initialize output layer
        output_vector = outputs[:num_inputs]
        for i in range(num_outputs):
            self.output_layer.append(node.node(node_id_counter, True, 0))
            # Add the output vector
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

    # Function to forward propagate through the network then determine if backprop is needed again
    def train(self):
        # Train the first hidden layer from the input layer
        for hidden_node in self.hidden_layers[0]:
            for input_node in self.input_layer:
                # Do the dot product between the input vector and the edge weight and store it
                hidden_node.value.append(
                    np.dot(input_node.value, self.weights[hidden_node.number][input_node.number][0]))
            # Run each weighted input vector through an activation function
            for vector in hidden_node.value:
                hidden_node.output.append(hidden_node.activation_function(vector))

        # Calculate the MSE
        self.calc_mse()

    def calc_mse(self):
        n = len(self.output_layer[0].value)
        temp_sum = [0 for i in range(n)]

        for hidden_node in self.hidden_layers[0]:
            for i in range(len(hidden_node.output)):
                final_value = np.dot(hidden_node.output[i],
                                     self.weights[hidden_node.number][self.output_layer[0].number][0])
                temp_sum[i] += (final_value - self.output_layer[0].value[i]) ** 2

        total_sum = 0
        for i in range(n):
            total_sum += temp_sum[i][0]

        print((1/n) * total_sum)

    def backprop(self):
        pass

    def hypothesis_of(self, testing_data):
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
    mlp_network = MLP(1000, 1, [5, 5], 1, input_vectors, outputs)
    mlp_network.train()


main()

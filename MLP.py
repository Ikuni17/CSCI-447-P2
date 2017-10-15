import node
import random
import numpy as np
import rosen_generator


class MLP:
    def __init__(self, num_inputs, num_hidden_layers, nodes_per_layer, num_outputs, input_vectors, outputs):
        # List of input node objects
        self.input_layer = []
        # List of lists which contain nodes for a hidden layer
        self.hidden_layers = []
        # List of output nodes
        self.output_layer = []
        self.converged = False
        # Matrix of weights where the connect between nodes i and j will be at index [i][j]. The third list contains
        # historical weights for that connection, where the 0th index is the current weight.
        # weights = [[]]
        # historical_weights = [[]]
        # Vector of MSE with the newest value in the 0th index
        self.error = 0
        self.current_input = 0
        self.node_id_counter = 0
        # Keep track of the total amount of nodes to construct the weight matrix
        self.input_vectors = input_vectors
        self.outputs = outputs
        self.problem_dimension = len(self.input_vectors[0])

        # Initialize input layer
        for i in range(num_inputs):
            self.input_layer.append(node.node(self.node_id_counter, True))
            self.node_id_counter += 1

        # Initialize any hidden layers
        # TODO Fix connecting two hidden layers
        for i in range(num_hidden_layers):
            # Add a new list to represent a layer
            self.hidden_layers.append([])
            # Add the amount of nodes specified for this layer
            for j in range(nodes_per_layer[i]):
                self.hidden_layers[i].append(node.node(self.node_id_counter, False))
                self.hidden_layers[i][j].initialize_weights(self.problem_dimension)
                if num_inputs > self.problem_dimension:
                    self.hidden_layers[i][j].append_zero_weights(num_inputs - self.problem_dimension)
                self.node_id_counter += 1

        # Initialize output layer
        # output_vector = outputs[:self.problem_dimension]
        for i in range(num_outputs):
            self.output_layer.append(node.node(self.node_id_counter, True))
            # TODO Generalize outputs
            self.output_layer[i].initialize_weights(len(self.hidden_layers[0]))
            self.node_id_counter += 1

        self.update_input()

        # Initialize all weights to zero
        '''self.weights = [[0 for i in range(node_id_counter)] for j in range(node_id_counter)]
        # Add a random weight between 0 and 1 if it isn't the connection to itself
        for i in range(node_id_counter):
            for j in range(node_id_counter):
                if i == j:
                    continue
                else:
                    self.weights[i][j] = random.random()

        print(self.weights)'''

    def update_input(self):
        for i in range(len(self.input_vectors[self.current_input])):
            self.input_layer[i].value = self.input_vectors[self.current_input][i]

        # TODO Generalize to multiple output nodes
        # for i in range(len(self.outputs[self.current_input])):
        self.output_layer[0].value = self.outputs[self.current_input]
        self.current_input += 1

    # Function to forward propagate through the network then determine if backprop is needed again
    def train(self):
        self.forward_prop()
        self.calc_error()

    def forward_prop(self):
        # Iterate through all hidden layers
        for i in range(len(self.hidden_layers)):
            # Iterate through all nodes within the hidden layer
            for hidden_node in self.hidden_layers[i]:
                # Special case for the first hidden layer
                if i == 0:
                    # Create a vector of the inputs from the input layer
                    temp_vector = []
                    for input_node in self.input_layer:
                        temp_vector.append(input_node.value)
                    # Do the dot product between the input vector and the weight vector, then input it into the
                    # activation function
                    hidden_node.output = hidden_node.activation_function(np.dot(temp_vector, hidden_node.weights))
                else:
                    # Create a vector of the outputs from the previous hidden layer
                    temp_vector = []
                    for previous_node in self.hidden_layers[i - 1]:
                        temp_vector.append(previous_node.output)
                    # Do the dot product between the input vector and the weight vector, then input it into the
                    # activation function
                    hidden_node.output = hidden_node.activation_function(np.dot(temp_vector, hidden_node.weights))

        # TODO Generalize to multiple output nodes
        # Handle the output layer
        # Check if we have any hidden layers
        if len(self.hidden_layers) == 0:
            temp_vector = []
            for input_node in self.input_layer:
                temp_vector.append(input_node.value)
            self.output_layer[0].output = self.output_layer[0].activation_function(
                np.dot(temp_vector, self.output_layer[0].weights))
        # Otherwise calculate off the last hidden layer
        else:
            temp_vector = []
            for hidden_node in self.hidden_layers[len(self.hidden_layers) - 1]:
                temp_vector.append(hidden_node.output)
            self.output_layer[0].output = self.output_layer[0].activation_function(
                np.dot(temp_vector, self.output_layer[0].weights))

    # Calculates the weighted inputs from the last hidden layer and then calculate the Means Squared Error for this
    # iteration.
    def calc_error(self):
        self.error += (self.output_layer[0].value - self.output_layer[0].output) ** 2

    def backprop(self):
        pass

    def hypothesis_of(self, testing_data):
        pass

    def output_results(self):
        pass

    def print_network(self):
        print("Inputs: ", end="")
        for i in range(len(self.input_layer)):
            print("{0}".format(self.input_layer[i].value), end=', ')

        print("\n\nHidden Layers: ")
        for i in range(len(self.hidden_layers)):
            print("\nLayer {0}: ".format(i), end="")
            for j in range(len(self.hidden_layers[i])):
                print("{0}".format(self.hidden_layers[i][j].value), end=', ')

        print("\n\nOutputs: ", end="")
        for i in range(len(self.output_layer)):
            print("{0}".format(self.output_layer[i].value), end=', ')


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
    mlp_network = MLP(2, 1, [5, 5], 1, input_vectors, outputs)
    mlp_network.train()
    mlp_network.print_network()


main()

import node
import random
import numpy as np
import rosen_generator


class MLP:
    def __init__(self, num_inputs, num_hidden_layers, nodes_per_layer, num_outputs, training_data, learning_rate=0.1,
                 epoch=1):

        # Make sure we have nodes per layer defined for all hidden layers
        # Can probably be moved to experiment.py after receiving user input
        assert num_hidden_layers <= len(nodes_per_layer), "Please specify the nodes per layer, for all hidden layers"

        # List of input node objects
        self.input_layer = []
        # List of lists which contain nodes for a hidden layer, if hidden layers exist
        self.hidden_layers = []
        # List of output nodes
        self.output_layer = []
        # Boolean to determine if the training has converged
        self.converged = False
        # Summation of the error during batch processing for each output node
        self.error = [0] * num_outputs
        # Keeps track of the current input vector
        self.current_input = 0
        # Creates a unique ID for each node within the network during initialization, unused currently
        self.node_id_counter = 0
        # List of vectors which contain the inputs, the ith input vector will have its corresponding output
        # at the ith index of self.output_vectors
        self.input_vectors = []
        # List of output vectors of the function
        self.output_vectors = []
        # Call helper to split the data into input and output vectors
        self.process_data(training_data)
        # The dimension of the function being approximated
        self.function_dimension = len(self.input_vectors[0])
        # The dimension of the output of the function being approximated
        self.output_dimension = len(self.output_vectors[0])
        # The learning rate of the network, default value of 0.1
        self.learning_rate = learning_rate
        # Number of iterations to train the network, default value of 10000
        self.epoch = epoch

        # Initialize input layer
        for i in range(num_inputs):
            self.input_layer.append(node.node(self.node_id_counter, True))
            self.node_id_counter += 1

        # Initialize any hidden layers
        for i in range(num_hidden_layers):
            # Add a new list to represent a layer
            self.hidden_layers.append([])
            # Add the amount of nodes specified for this layer
            for j in range(nodes_per_layer[i]):
                self.hidden_layers[i].append(node.node(self.node_id_counter, False))
                # If this is the first hidden layer, connect to the input layer
                if i == 0:
                    # Create the edges connecting the input layer
                    self.hidden_layers[i][j].initialize_weights(self.function_dimension)
                    # If we have extra input nodes with no value, set their weights to zero
                    if num_inputs > self.function_dimension:
                        self.hidden_layers[i][j].append_zero_weights(num_inputs - self.function_dimension)
                # Otherwise connect to the previous hidden layer
                else:
                    # Create the edges connecting the previous hidden layer
                    self.hidden_layers[i][j].initialize_weights(len(self.hidden_layers[i - 1]))

                self.node_id_counter += 1

        # Initialize output layer
        for i in range(num_outputs):
            self.output_layer.append(node.node(self.node_id_counter, True))
            if i < self.output_dimension:
                # Check if we have hidden layers
                if len(self.hidden_layers) == 0:
                    # Create the edges connecting the input layer
                    self.output_layer[i].initialize_weights(self.function_dimension)
                    # If we have extra input nodes with no value, set their weights to zero
                    if num_inputs > self.function_dimension:
                        self.output_layer[i].append_zero_weights(num_inputs - self.function_dimension)
                # Otherwise connect to the last hidden layer
                else:
                    self.output_layer[i].initialize_weights(nodes_per_layer[-1])
            else:
                if len(self.hidden_layers) == 0:
                    self.output_layer[i].append_zero_weights(num_inputs)
                else:
                    self.output_layer[i].append_zero_weights(nodes_per_layer[-1])

            self.node_id_counter += 1

        self.update_input()

    # Initialize the inputs and output from the function
    def update_input(self):
        # Use a single input value per input node
        for i in range(self.function_dimension):
            self.input_layer[i].value = self.input_vectors[self.current_input][i]

        # Use a single output value per output node
        for i in range(self.output_dimension):
            self.output_layer[i].value = self.output_vectors[self.current_input][i]

        # Move to the next vectors
        self.current_input += 1

    # Function to forward propagate through the network then determine if backprop is needed again
    def train(self):
        for i in range(self.epoch):
            for j in range(len(self.input_vectors)):
                self.forward_prop()
                self.calc_error()
                if j != len(self.input_vectors) - 1:
                    self.update_input()
            # self.backprop()
            # Reset parameters before next iteration
            self.current_input = 0
            self.update_input()
            # print(self.error)
            self.error = [0] * len(self.output_layer)
            # print(self.error/len(self.input_vectors))

    # Forward propagation through the network calculating weighted sums
    def forward_prop(self):
        # Iterate through all hidden layers, if any
        for i in range(len(self.hidden_layers)):
            # Iterate through all nodes within the hidden layer
            for hidden_node in self.hidden_layers[i]:
                # Create a temporary vector to store the inputs from the previous layer
                temp_vector = []
                # Special case for the first hidden layer
                if i == 0:
                    # Create a vector of the inputs from the input layer
                    for input_node in self.input_layer:
                        temp_vector.append(input_node.value)
                else:
                    # Create a vector of the output_vectors from the previous hidden layer
                    for previous_node in self.hidden_layers[i - 1]:
                        temp_vector.append(previous_node.output[0])

                # Do the dot product between the input vector and the weight vector, then input it into the
                # activation function. Then insert it into the 0th index of this nodes outputs.
                hidden_node.output.insert(0,
                                          hidden_node.activation_function(np.dot(temp_vector, hidden_node.weights)))

        # Handle the output layer
        for i in range(len(self.output_layer)):
            temp_vector = []
            # Check if we have any hidden layers
            if len(self.hidden_layers) == 0:
                for input_node in self.input_layer:
                    temp_vector.append(input_node.value)
            # Otherwise calculate off the last hidden layer
            else:
                for hidden_node in self.hidden_layers[len(self.hidden_layers) - 1]:
                    temp_vector.append(hidden_node.output[0])
            self.output_layer[i].output.insert(0, self.output_layer[i].activation_function(
                np.dot(temp_vector, self.output_layer[i].weights)))

    # Calculates the weighted inputs from the last hidden layer and then calculate the Means Squared Error for this
    # iteration.
    def calc_error(self):
        for i in range(len(self.output_layer)):
            self.error[i] += (self.output_layer[i].value - self.output_layer[i].output[0]) ** 2

    def backprop(self, err):
        #err is total error on output layer (deltai )
        counter = 0
        for j in range(len(self.output_layer)):
            counter = 0
            for i in range(self.hidden_layers[len(self.hidden_layers)]):
                counter +=self.hidden_layers[len(self.hidden_layers) - 1][i].value[j]

            modifier = (self.learning_rate * counter * err)
            self.output_layer.weights += modifier

        for hidden_layer, j  in reversed(list(enumerate(self.hidden_layers))):
            for i in range(len(hidden_layer)):
                counter = 0
                if j!= 0:
                    for node in hidden_layers[j-1]:
                        counter += node.value[i]
                else:
                    for node in self.input_layer:
                        counter += node.value[i]
                modifier = (self.learning_rate * counter * err)
                self.output_layer.weights += modifier






    def hypothesis_of(self, testing_data):
        # Reset parameters before testing the network
        self.input_vectors = []
        self.output_vectors = []
        self.current_input = 0
        self.error = [0] * len(self.output_layer)
        # Process the testing data in the vectors
        self.process_data(testing_data)
        n = len(self.input_vectors)
        self.update_input()

        # Forward prop for each input then calculate the error for that input
        for i in range(n):
            self.forward_prop()
            self.calc_error()
            if i != n - 1:
                self.update_input()

        # Return the MSE for the testing data
        return np.dot((1/n), self.error)

    # Split the data into input and output vectors
    def process_data(self, dataset):
        dimension = len(dataset[0]) - 1
        for i in range(len(dataset)):
            self.input_vectors.append(dataset[i][:dimension])
            self.output_vectors.append([dataset[i][dimension]])

    def output_results(self):
        pass

    def print_network(self):
        print("Inputs: ", end="")
        for i in range(len(self.input_layer)):
            print("{0}".format(self.input_layer[i].weights), end=', ')

        print("\n\nHidden Layers: ")
        for i in range(len(self.hidden_layers)):
            print("\nLayer {0}: ".format(i), end="")
            for j in range(len(self.hidden_layers[i])):
                print("{0}".format(self.hidden_layers[i][j].weights), end=', ')

        print("\n\nOutputs: ", end="")
        for i in range(len(self.output_layer)):
            print("{0}".format(self.output_layer[i].weights), end=', ')


def main():
    rosen_in = rosen_generator.generate(input_type=0, dimensions=2)
    mlp_network = MLP(num_inputs=2, num_hidden_layers=1, nodes_per_layer=[5, 5], num_outputs=1, training_data=rosen_in)
    mlp_network.train()
    mlp_network.print_network()
    rosen_test = rosen_generator.generate(input_type=0, dimensions=2)
    print(mlp_network.hypothesis_of(rosen_test))


#main()

import node 

class MLP:

    input_layer = []
    hidden_layers = [[]]
    output_layer = []
    momentum = None
    converged = False
    weights = [[[]]]

    def __init__(self, num_inputs, num_layers, nodes_per_layer, num_outputs, momentum):
        node_id_counter = 0

        for i in range(num_inputs):
            self.input_layer.append(node.node(node_id_counter, True, 0))
            node_id_counter += 1

        for j in range(num_layers):
            if j != 0:
                self.hidden_layers.append([])
            for k in range(nodes_per_layer[j]):
                self.hidden_layers[j].append(node.node(node_id_counter, True, 0))
                node_id_counter += 1

        for m in range(num_outputs):
            self.output_layer.append(node.node(node_id_counter, True, 0))
            node_id_counter += 1

    def train(self, training_data):
        # Train the first layer from the input layer

        # Train any hidden layers based on previous layer

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
    mlp_network = MLP(2, 2, [4, 6], 1, 0)
    mlp_network.print_network()

main()
import node
import copy
import random
import numpy as np


class RBF:
    input_layer = []
    clusters = []
    output_layer = []
    num_vectors = []
    sigma = None
    converged = False
    training_data = None
    hidden_layer = []

    def __init__(self, num_inputs, num_basis_functions, num_output, training_data):
        self.training_data = copy.copy(training_data)
        # Initialize k-centers using number of basis functions by randomly picking points from the data set.
        for i in range(0, num_basis_functions):
            index = random.randint(0, len(training_data))
            self.hidden_layer.append(node.node(2, True, self.training_data[index]))
            self.clusters.append([0] * (num_inputs + 1))

    def train(self):
        #training_data = self.training_data
        #hidden_layer = self.hidden_layer
        #clusters = self.clusters

        converged = False
        while not converged:
            converged = True
            # initialize clusters
            for i in range(0, len(self.training_data) - 1):
                temp_index = None
                temp_min = None
                for j in range(len(self.hidden_layer) - 1):
                    # to not include outputs in distance calculation
                    # if temp_min == None or np.linalg.norm(training_data[i][:len(training_data[i]) -1] - hidden_layer[j][:len(hidden_layer[j]) - 1]
                    if temp_min is None or np.linalg.norm(
                                    np.array(self.training_data[i]) - np.array(self.hidden_layer[j].center)) < temp_min:
                        temp_min = np.linalg.norm(np.array(self.training_data[i]) - np.array(self.hidden_layer[j].center))
                        temp_index = j
                self.clusters[temp_index].append(i)

            # Calculates new centers
            average = [0] * len(self.clusters[0])
            for i in range(len(self.clusters) - 1):
                for item in average:
                    item = 0
                if i == 1:
                    print(average)
                # For every index in cluster
                for j in range(len(self.clusters[i]) - 1):
                    average = [x + y for x, y in zip(average, self.training_data[self.clusters[i][j]])]

                average = [x / len(self.clusters[i]) for x in average]
                if average != self.hidden_layer[i].center:
                    converged = False
                    self.hidden_layer[i].center = average

    def hypothesis_of(self, testing_data):
        pass

    def calculate_error(self):
        pass

    def output_results(self):
        pass

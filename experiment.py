import RBF
import MLP
import rosen_generator
import threading
import time
import csv
import numpy as np


# Class to handle an RBF network in a thread, used for experimentation
class RBFThread(threading.Thread):
    def __init__(self, thread_ID, dataset, num_basis):
        threading.Thread.__init__(self)
        self.thread_ID = thread_ID
        self.name = "RBF{0}".format(self.thread_ID)
        self.num_basis = num_basis
        self.num_dim = len(dataset[0]) - 1
        self.training_data = dataset[:int(len(dataset) * 0.8)]
        self.testing_data = dataset[int(len(dataset) * 0.8):]

    def run(self):
        # Train an RBF network based on its input parameters and dataset
        print("Thread {0}: starting {1} TRAINING with {2} dimensions and {3} basis functions at {4}".format(
            self.thread_ID, self.name, self.num_dim, self.num_basis, time.ctime(time.time())))
        rbf = RBF.RBF(self.num_basis, self.training_data)
        loss_set = rbf.train()
        with open('RBF_{0}.csv'.format(self.thread_ID), 'wb') as csvfile:
            results_writ = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            results_writ.writerow(loss_set)

        # Test the same RBF network on a portion of the dataset
        print("Thread {0}: starting {1} TESTING with {2} dimensions and {3} basis functions at {4}".format(
            self.thread_ID, self.name, self.num_dim, self.num_basis, time.ctime(time.time())))
        result = rbf.hypothesis_of(self.testing_data)
        print("Thread {0}: {1} result {5} with {2} dimensions and {3} basis functions at {4}".format(self.thread_ID,
                                                                                                     self.name,
                                                                                                     self.num_dim,
                                                                                                     self.num_basis,
                                                                                                     time.ctime(
                                                                                                         time.time()),
                                                                                                     '?'))


# Class to handle an MLP network in a thread, used for experimentation
class MLPThread(threading.Thread):
    def __init__(self, thread_ID, dataset, num_inputs, num_hidden_layers, num_nodes_per_layer, num_outputs=1):
        threading.Thread.__init__(self)
        self.thread_ID = thread_ID
        self.name = "MLP".format(self.thread_ID)
        self.num_inputs = num_inputs
        self.num_dim = len(dataset[0]) - 1
        self.num_hidden_layers = num_hidden_layers
        self.num_nodes_per_layer = num_nodes_per_layer
        self.num_outputs = num_outputs
        self.training_data = dataset[:int(len(dataset) * 0.8)]
        self.testing_data = dataset[int(len(dataset) * 0.8):]

    def run(self):
        print("Thread {0}: starting {1} TRAINING with {2} dimensions and {3} hidden layers at {4}".format(
            self.thread_ID, self.name, self.num_dim, self.num_hidden_layers, time.ctime(time.time())))
        mlp = MLP.MLP(self.num_inputs, self.num_hidden_layers, self.num_nodes_per_layer, self.num_outputs,
                      self.training_data)
        temp = []
        with open('MLP{0} Learning Curve.csv'.format(self.thread_ID), 'w', newline='') as csvfile:

            for i in range(len(mlp.overall_error)):
                temp.append(mlp.overall_error[i][0])
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(temp)
            print(temp)

        print("Thread {0}: starting {1} TESTING with {2} dimensions and {3} hidden layers at {4}".format(
            self.thread_ID, self.name, self.num_dim, self.num_hidden_layers, time.ctime(time.time())))

        result = mlp.hypothesis_of(self.testing_data)
        print("Thread {0}: {1} result {5} with {2} dimensions and {3} hidden layers at {4}".format(
            self.thread_ID, self.name, self.num_dim, self.num_hidden_layers, time.ctime(time.time()), result))
        with open('MLP{0} Test.csv'.format(self.thread_ID), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(result)


def create_folds(data, num_folds):
    data_length = len(data)
    fold_length = int(data_length / num_folds)
    folded_data = []
    for i in range(num_folds):
        folded_data.append(data[:fold_length])
        data = data[fold_length:]
    return folded_data


def fold_training(network, data):
    num_folds = len(data)
    for i in range(num_folds):
        current_data_set = []
        for j in data:
            if j != i:
                current_data_set.append(data[i])
                # print('\n' + str(current_data_set) + '\n')
                # network.train(current_data_set)


def perform_experiment():
    rosen_datasets = []
    for i in range(2, 7):
        rosen_datasets.append(rosen_generator.generate(0, i))

    '''rbf_threads = []
    thread_counter = 0

    for i in range(len(rosen_datasets)):
        rbf_threads.append(RBFThread(thread_counter, rosen_datasets[i], 3))
        thread_counter += 1
        rbf_threads.append(RBFThread(thread_counter, rosen_datasets[i], 5))
        thread_counter += 1
        rbf_threads.append(RBFThread(thread_counter, rosen_datasets[i], 7))
        thread_counter += 1'''

    print("Overall and RBF starting time: {0}".format(time.ctime(time.time())))
    '''for i in range(len(rbf_threads)):
        rbf_threads[i].start()

    for i in range(len(rbf_threads)):
        rbf_threads[i].join()
    print("RBF ending time: {0}".format(time.ctime(time.time())))'''

    mlp_threads = []
    thread_counter = 0

    #for i in range(len(rosen_datasets)):
    i = 0
    current_dim = len(rosen_datasets[i]) - 1

    mlp_threads.append(MLPThread(thread_counter, rosen_datasets[i], current_dim, 0, [0], 1))
    thread_counter += 1
    mlp_threads.append(MLPThread(thread_counter, rosen_datasets[i], current_dim, 1, [current_dim + 1], 1))
    thread_counter += 1
    mlp_threads.append(
        MLPThread(thread_counter, rosen_datasets[i], current_dim, 2, [current_dim + 1, current_dim + 1], 1))
    thread_counter += 1

    print("MLP starting time: {0}".format(time.ctime(time.time())))
    for i in range(len(mlp_threads)):
        mlp_threads[i].start()

    for i in range(len(mlp_threads)):
        mlp_threads[i].join()

    print("Overall and MLP ending time: {0}".format(time.ctime(time.time())))


def perform_comparison(rosen_dim, num_basis_functions, num_inputs, num_hidden_layers, num_nodes_per_layer,
                       num_outputs, input_type):
    rosen_dataset = rosen_generator.generate(input_type, rosen_dim)

    rbf_thread = RBFThread(0, rosen_dataset, num_basis_functions)
    mlp_thread = MLPThread(1, rosen_dataset, num_inputs, num_hidden_layers, num_nodes_per_layer, num_outputs)

    rbf_thread.start()
    mlp_thread.start()

    rbf_thread.join()
    mlp_thread.join()


def main():
    valid_response1 = False
    while not valid_response1:
        mode1 = input("Perform the experiment or create neural networks? (e/n) > ")

        if mode1 == "e":
            valid_response1 = True
            print("Starting experiment")
            perform_experiment()

        elif mode1 == "n":
            valid_response1 = True
            valid_response2 = False
            while not valid_response2:
                mode2 = input("Use default inputs? (y/n) > ")

                if mode2 == "y":
                    valid_response2 = True
                    rosen_dim = 2
                    num_basis_functions = 7
                    num_inputs = 2
                    num_hidden_layers = 1
                    num_nodes_per_layer = [7]
                    num_outputs = 1
                    input_type = 0
                    print("Using the following parameters:\n"
                          "Dimension of Rosenbrock: {0}\n"
                          "Randomly generated Rosenbrock input values\n"
                          "Number of basis functions (RBF): {1}\n"
                          "Number of input nodes (MLP): {2}\n"
                          "Number of hidden layers (MLP): {3}\n"
                          "Number of hidden nodes per layer (MLP): {4}\n"
                          "Number of output nodes (MLP): {5}\n".format(
                        rosen_dim, num_basis_functions, num_inputs, num_hidden_layers, num_nodes_per_layer,
                        num_outputs))

                    perform_comparison(rosen_dim, num_basis_functions, num_inputs, num_hidden_layers,
                                       num_nodes_per_layer, num_outputs, input_type)

                elif mode2 == "n":
                    valid_response2 = True
                    rosen_dim = int(input("Dimension of Rosenbrock > "))
                    input_type = int(input(
                        "Randomly generated (0) or sequential (1) input values for the Rosenbrock function (0/1) > "))
                    num_basis_functions = int(input("Number of basis functions (RBF) > "))
                    num_inputs = int(input("Number of input nodes (MLP) > "))
                    num_hidden_layers = int(input("Number of hidden layers (MLP) > "))
                    if num_hidden_layers > 0:
                        s = input("Space separated number of nodes per hidden layer (MLP) > ")
                        num_nodes_per_layer = list(map(int, s.split()))
                    else:
                        num_nodes_per_layer = [0]
                    num_outputs = int(input("Number of output nodes (MLP) > "))

                    perform_comparison(rosen_dim, num_basis_functions, num_inputs, num_hidden_layers,
                                       num_nodes_per_layer, num_outputs, input_type)

                else:
                    print("Please enter a valid response")

        else:
            print("Please enter a valid response")


if __name__ == '__main__':
    main()
    # test = create_folds(rosen_generator.generate(1, 2), 3)
    # fold_training(5, test)

import RBF
import MLP
import rosen_generator
import threading
import time


class RBFThread(threading.Thread):
    def __init__(self, thread_ID, dataset, num_basis):
        threading.Thread.__init__(self)
        self.thread_ID = thread_ID
        self.name = "RBF"
        self.num_basis = num_basis
        self.num_dim = len(dataset[0]) - 1
        self.training_data = dataset[:int(len(dataset) * 0.8)]
        self.testing_data = dataset[int(len(dataset) * 0.8):]

    def run(self):
        print("Thread {0}: starting {1} TRAINING with {2} dimensions and {3} basis functions at {4}".format(
            self.thread_ID, self.name, self.num_dim, self.num_basis, time.ctime(time.time())))
        rbf = RBF.RBF(self.num_basis, self.training_data)
        rbf.train()
        print("Thread {0}: starting {1} TESTING with {2} dimensions and {3} basis functions at {4}".format(
            self.thread_ID, self.name, self.num_dim, self.num_basis, time.ctime(time.time())))
        #rbf.hypothesis(self.testing_data)

'''def __init__(self, num_inputs, num_hidden_layers, nodes_per_layer, num_outputs, training_data, learning_rate=0.1,
                 epoch=1):'''
class MLPThread(threading.Thread):
    def __init__(self, thread_ID, dataset):
        threading.Thread.__init__(self)
        self.thread_ID = thread_ID
        self.name = "MLP"
        self.num_dim = len(dataset[0]) - 1
        self.training_data = dataset[:int(len(dataset) * 0.8)]
        self.testing_data = dataset[int(len(dataset) * 0.8):]


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
        print('\n' + str(current_data_set) + '\n')
        # network.train(current_data_set)


def perform_experiment():
    rosen_datasets = []
    for i in range(2, 7):
        rosen_datasets.append(rosen_generator.generate(0, i))

    '''print(len(rosen_datasets))

    for i in range(len(rosen_datasets)):
        print(len(rosen_datasets[i]))'''

    '''training_data = rosen_datasets[0][:int(len(rosen_datasets[0]) * 0.8)]
    testing_data = rosen_datasets[0][int(len(rosen_datasets[0]) * 0.8):]

    rbf2 = RBF.RBF(3, training_data)
    print("Training RBF with 2 dimensions and 3 basis functions")
    rbf2.train()
    print("Testing RBF with 2 dimensions and 3 basis functions")
    print(rbf2.hypothesis(testing_data))
    # thing = input("Waiting")'''

    rbf_threads = []
    thread_counter = 0

    for i in range(len(rosen_datasets)):
        rbf_threads.append(RBFThread(thread_counter, rosen_datasets[i], 3))
        thread_counter += 1
        rbf_threads.append(RBFThread(thread_counter, rosen_datasets[i], 5))
        thread_counter += 1
        rbf_threads.append(RBFThread(thread_counter, rosen_datasets[i], 7))
        thread_counter += 1

    print("Overall starting time: {0}".format(time.ctime(time.time())))
    for i in range(len(rbf_threads)):
        rbf_threads[i].start()

    for i in range(len(rbf_threads)):
        rbf_threads[i].join()
    print("Overall ending time: {0}".format(time.ctime(time.time())))


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
                    num_basis_functions = 40
                    num_inputs = 2
                    num_hidden_layers = 1
                    num_hidden_nodes = 5
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
                        rosen_dim, num_basis_functions, num_inputs, num_hidden_layers, num_hidden_nodes, num_outputs))

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
                        num_hidden_nodes = list(map(int, s.split()))
                    else:
                        num_hidden_nodes = [0]
                    num_outputs = int(input("Number of output nodes (MLP) > "))

                else:
                    print("Please enter a valid response")

        else:
            print("Please enter a valid response")

            # rbf_nn = RBF(num_inputs, num_basis_functions, num_outputs)
            # mlp_nn = MLP(num_inputs, nodes_per_layer, num_outputs, momentum)

            # rosen_in = rosen_generator.generate(input_type, rosen_dim)
            # rbf_nn = RBF.RBF(num_basis_functions, rosen_in)

            # rbf_nn.train()
            # mlp_nn.train(rosen_in)

            # rosen_test = rosen_generator.generate(input_type, num_data_points)
            # print('testing:\n' + str(rosen_test) + '\n')

            # results_rbf = rbf_nn.hypothesis_of(rosen_tests)
            # results_mlp = mlp_nn.hypothesis_of(rosen_tests)


if __name__ == '__main__':
    main()
    # test = create_folds(rosen_generator.generate(1, 2), 3)
    # fold_training(5, test)

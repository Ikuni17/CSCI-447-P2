# import RBF
# import MLP
import rosen_generator


def create_folds(data, num_folds):
    data_length = len(data)
    fold_length = int(data_length / num_folds)
    folded_data = []
    for i in range(num_folds):
        folded_data.append(data[:fold_length])
        data = data[fold_length:]
    return folded_data


def fold_training(data, folds):
    for i in range(folds):
        current_data_set = []


def main():
    mode = input('Type anything to run the default test: ')
    num_inputs = 2  # int(input('Number of inputs(dimensionality): '))
    num_basis_functions = 40  # int(input('Number of basis functions: '))
    nodes_per_layer = 8  # int(input('Number of nodes per layer: '))
    num_data_points = 1000  # int(input('Number of data points: '))
    num_outputs = 1
    input_type = 0
    # rbf_nn = RBF(num_inputs, num_basis_functions, num_outputs)
    # mlp_nn = MLP(num_inputs, nodes_per_layer, num_outputs, momentum)

    rosen_in = rosen_generator.generate(input_type, num_inputs, num_data_points)
    rbf_nn = RBF.RBF(num_basis_functions, rosen_in)

    rbf_nn.train()
    # mlp_nn.train(rosen_in)

    rosen_test = rosen_generator.generate(input_type, num_data_points)
    # print('testing:\n' + str(rosen_test) + '\n')

    # results_rbf = rbf_nn.hypothesis_of(rosen_tests)
    # results_mlp = mlp_nn.hypothesis_of(rosen_tests)


if __name__ == '__main__':
    # main()
    testing_data = rosen_generator.generate(1, 2)
    folded_data = create_folds(testing_data, 10)
    print(str(folded_data))

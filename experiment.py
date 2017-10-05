import RBF
import MLP
import rosen_generator

def main():
    mode = int(input('Type anything to run the default test: '))
    num_inputs = int(input('Number of inputs: '))
    num_basis_functions = int(input('Number of basis functions: '))
    nodes_per_layer = int(input('Number if nodes per layer: '))
    num_outputs = 1
    rbf_nn = RBF(num_inputs, num_basis_functions, num_outputs) 
    mlp_nn = MLP(num_inputs, nodes_per_layer, num_outputs, momentum)
    
    rosen_in = rosen_generator(num_inputs)
    rbf_nn.train(rosen_in)
    mlp_nn.train(rosen_in)

    rosen_test = rosen_generator(num_inputs)

    results_rbf = rbf_nn.hypothesis_of(rosen_tests)
    results_mlp = rbf_nn.hypothesis_of(rosen_tests)

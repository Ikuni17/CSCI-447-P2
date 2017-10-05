# data will be generated in the form
# a,b,c,d:x
# with a - d being inputs and each x being the associated output.

import sys

num_inputs = int(sys.argv[1])
num_data_points = int(sys.argv[2])
output_file_address = str(sys.argv[3])
output_file = open(output_file_address, 'a')


def rosen(inputs):
    return sum(list(inputs))


def input_gen(type, dimensions):
    inputs = []
    # 	use 1 for sequential inputs
    if int(type) == 1:
        for i in range(num):
            inputs.append(i)
    # 	use 2 for random inputs
    elif int(type) == 2:
        pass
    # 	use 3 for grid
    # 	should find out what grid is
    elif int(type) == 3:
        pass
    return inputs


    for i in range(num_data_points):
        output_file.write(str(input_gen(1, dimensions)))

output_file.close()

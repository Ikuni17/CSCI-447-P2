# generates input vectors and
# calculates the rosenbrock function
# on each one.

# input parameters
# 0,1,2		random,sequential,grid input vectors
# 2,3,4,5,6	dimension of the input vectors
# integer		number of data points
# string		output file

import sys
import random

MAX_RANDOM_SIZE = 100

if __name__ == "__main__":
	input_type = int(sys.argv[1])
	dimensions = int(sys.argv[2])
	num_data_points = int(sys.argv[3])
	output_file_address = str(sys.argv[4])
	output_file = open(output_file_address, 'a')
	MAX_RANDOM_SIZE = num_data_points

	generate(input_type, dimensions, num_data_points)

	output_file.close()

# takes a list of inputs and returns the rosenbrock function output on those inputs
def rosen(inputs):
	totals = []
	for i in range(len(inputs)-1):
		temp = (((1 - inputs[i])**2) + (100 * (inputs[i+1] - inputs[i]**2)**2))
		totals.append(temp)
	result = sum(totals)
	return result

# generates inputs for rosenbrock function
# 0 - random numbers
# 1 - sequential numbers
# 2 - grid numbers??? NEED TO LOOK THIS UP/ASK BIG SHEP
def input_gen(type, dimensions, num_data_points):
	generated = []
	if type == 0:
		for i in range(num_data_points):
			temp = []
			for j in range(dimensions):
				temp.append(random.random() * float(MAX_RANDOM_SIZE))
			generated.append(temp)
	elif type == 1:
		counter = 1
		for i in range(num_data_points):
			temp = []
			for j in range(dimensions):
				temp.append(counter)
				counter += 1
			generated.append(temp)
	elif type == 2:
		pass
	return generated

# appends the result of the rosenbrock function to each input and records the result in output_file
def apply_rosen(inputs):
	for i in inputs:
		dimension = len(i)
		i.append(rosen(i))
		if __name__ == '__main__':
			output_file.write(str(i) + '\n')
	return inputs

# generates a data set based on the provided parameters
# input parameters
# 0,1,2		random,sequential,grid input vectors
# 2,3,4,5,6	dimension of the input vectors
# integer		number of data points
# string		output file
def generate(input_type, dimensions, num_data_points):
	return apply_rosen(input_gen(input_type, dimensions, num_data_points))

import sys
import random

dimensions = int(sys.argv[1])
num_data_points = int(sys.argv[2])
output_file_address = str(sys.argv[3])
output_file = open(output_file_address, 'a')

# takes a list of inputs and returns the rosenbrock function output on those inputs
def rosen(inputs):
	totals = []
	for i in range(len(inputs)):
		if i < (len(inputs)-1):
			temp = (((1 - inputs[i])**2) + (100 * (inputs[i+1] - inputs[i]**2)**2))
			totals.append(temp)
	result = sum(totals)
	print(str(result))
	return 

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
				temp.append(random.random() * num_data_points)
			generated.append(temp)
	elif type == 1:
		counter = 0
		for i in range(num_data_points):
			temp = []
			for j in range(dimensions):
				temp.append(counter)
				counter += 1
			generated.append(temp)
	elif type == 2:
		pass
	output_file.write(str(generated))
	return generated

input_gen(1, dimensions, num_data_points)
output_file.close()
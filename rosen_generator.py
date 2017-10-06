# data will be generated in the form
# a,b,c,d:x
# with a - d being inputs and each x being the associated output.

import sys
import random

num_inputs = int(sys.argv[1])
num_data_points = int(sys.argv[2])
output_file_address = str(sys.argv[3])
output_file = open(output_file_address, 'a')

def rosen(inputs):
	totals = []
	for i in range(len(inputs)):
		if i < (len(inputs)-1):
			temp = (((1 - inputs[i])**2) + (100 * (inputs[i+1] - inputs[i]**2)**2))
			totals.append(temp)
	result = sum(totals)
	print(str(result))

def input_gen(type, dimensions, num_data_points):
	inputs = []
	next_input = []
	if type == 0:
		for i in range(num_data_points):
			next_input = []
			for j in range(dimensions):
				next_input.append(random.random() * num_data_points)
# 			print(str(i) + ': ' + str(next_input) + '\n')
	elif type == 1:
		pass
	elif type == 2:
		pass
	output_file.write(str(inputs))

output_file.write(str(rosen([0,1,8])))
# output_file.write(str(input_gen(0, 2, num_data_points)))
output_file.close()
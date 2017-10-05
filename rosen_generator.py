import sys

num_inputs = int(sys.argv[1])
num_data_points = int(sys.argv[2])
output_file_address = str(sys.argv[3])
output_file = open(output_file_address, 'a')

for i in range(0, num_data_points):
	
	output_file.write('\ntest')
	

def rosen(inputs):
	for i in inputs:
		print(str(i))

def input_gen(type, num):
	inputs = 0
# 	use 1 for sequential inputs
	if int(type) == 1: 
		for i in range(num):
			list.append(i)
# 	use 2 for random inputs
	elif int(type) == 2:
		pass
# 	use 3 for grid 
# 	should find out what grid is
	elif int(type) == 3:
		pass
	return inputs
		

output_file.close()
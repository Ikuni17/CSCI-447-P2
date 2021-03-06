import random
import numpy as np
import rosen_generator as rosen

class MLP2:
	def __init__(self, num_inputs, num_hidden_layers, nodes_per_layer, training_data, learning_rate=0.01, iterations=1000):
		self.weights = [] # Each numpy array in this represents the weights coming into a node
		self.inputs = []
		self.train_in = []
		self.train_out = []
		self.activation = [] # Each numpy array in this represents the activation leaving a node for every input
		self.iterations = iterations
		self.learning_rate = learning_rate
		for x in training_data:
			self.train_in.append(x[:num_inputs])
			self.train_out.append(x[num_inputs:])
		print('Setting up the network with {0} inputs and {1} output(s)'.format(len(self.train_in[0]),len(self.train_out[0])))
		# print('Train_in: {0}'.format(train_in))
		# Initilize the NN with random weights and populate the activation matrix
		for i in range(num_hidden_layers+1):
			self.weights.append([]) # append a matrix to represent a layer in the NN
			if i == 0:
				# this transposes the input to the right format and adds as the first activation layer
				self.activation.append(np.array(self.train_in).transpose())
				num_nodes = nodes_per_layer
				num_weights = num_inputs
			# use the number of outputs for the last layer
			elif i == num_hidden_layers:
				num_weights = num_nodes
				num_nodes = len(self.train_out[0])
				self.activation.append(np.array([]))
			# Otherwise it's a hidden_layer
			else:
				num_weights = num_nodes
				num_nodes = nodes_per_layer
				self.activation.append(np.array([]))
				
			# Each matrix in a layer represents a node and holds all the weights coming into the node
			for j in range(num_nodes):
				temp = []
				for k in range(num_weights):
					temp.append(random.uniform(0, 10))
				self.weights[i].append(np.array(temp))
		self.activation.append(np.array([]))
		# print('Input_layer: {0}'.format(self.activation[0]))

	def train(self):
		for i in range(self.iterations):
			self.feedforward()
			error = self.backprop()
			if i%100 == 0:
				print('Error at iteration {0}: {1}'.format(i, error))
				#if error < 0.1:
					#print('Actual: {0}, Predicted: {1}'.format(self.train_out[:3], self.activation[len(self.activation)-1][:3][:3]))

	# updates the activation arrays and the output
	def feedforward(self):
		for i in range(len(self.weights)):
			temp = np.zeros((len(self.weights[i]), len(self.activation[0][0])))
			for j in range(len(self.weights[i])):
				# append all the activations to list to be converted to an np array as an actiaviton layer
				temp[j] = self.activation[i].transpose().dot(self.weights[i][j])
			# don't run the activation on the tanh function
			if(i == len(self.weights)-1):
				self.activation[i+1] = np.array(temp)
			else:
				self.activation[i+1] = np.tanh(temp)

	def backprop(self):
		error = np.average(np.subtract(np.array(self.train_out), self.activation[len(self.activation)-1][0]))
		#print('output: {0} actual: {1}, error: {2}'.format(self.activation[len(self.activation)-1][0], self.train_out, error))
		for i, layer in reversed(list(enumerate(self.weights))):
			for j in range(len(layer)):
				activ_out = self.activation[i+1][j]
				modifiers = activ_out*error
				#print('activ_out: {0}, error: {1}'.format(activ_out, error))
				self.weights[i][j] = np.add(self.weights[i][j], self.learning_rate*np.average(modifiers))
		return error

	def print_nn(self):
		print('Dimensions of weights \n --------------------------')
		for layers in self.weights:
			for weight in layers:
				print(weight.shape, end='')
			print('\n')
		print('Dimensions of activations \n --------------------------')
		for layer in self.activation:
				print(layer.shape)

	def change_training(self, training_data):
		for x in training_data:
			self.train_in.append(x[:num_inputs])
			self.train_out.append(x[num_inputs:])
		self.actiaviton[0] = np.array(self.train_in).transpose()
	
	def calc_error(self, testing_data)
		self.change_training(training_data)
		self.feedforward()
		return np.average(np.subtract(np.array(self.train_out), self.activation[len(self.activation)-1][0]))
	
	# Calcualte the acitvations of a node given it's wieghts and the values coming into the node
	@staticmethod
	def activ_fun(activ, weights):
		x = np.array(activ).transpose()
		y = np.array(weights)
		z = np.tanh(y.dot(x))
		return z

def main():
	num_inputs = 2
	training_data = rosen.generate(0, num_inputs)
	MLP = MLP2(num_inputs, 2, 8, training_data)
	MLP.train()


if __name__ == "__main__":
	main()

import numpy as np
import torch
import math

dtype = torch.double
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Convolution:
	def __init__(self, in_neurons, out_neurons):
		self.out_neurons = out_neurons
		self.in_neurons = in_neurons

		self.output = None # batch_size X out_neurons
		self.W = torch.randn(out_neurons, in_neurons, dtype=dtype, device=device)*math.sqrt(2.0/self.in_neurons) # out_neurons X in_neurons
		self.B = torch.randn(out_neurons, 1, dtype=dtype, device=device)*math.sqrt(2.0/self.in_neurons) # out_neurons X 1
		self.gradW = torch.zeros(out_neurons, in_neurons, dtype=dtype, device=device) # out_neurons X in_neurons
		self.gradB = torch.zeros(out_neurons, 1, dtype=dtype, device=device) # out_neurons X 1
		self.gradItorchut = None # batch_size X in_neurons


	def __init__(self, in_channels, filter_size, numfilters, stride):

		self.in_depth, self.in_row, self.in_col = in_channels
		self.filter_row, self.filter_col = filter_size
		self.stride = stride

		self.out_depth = numfilters
		self.out_row = int((self.in_row - self.filter_row)/self.stride + 1)
		self.out_col = int((self.in_col - self.filter_col)/self.stride + 1)

		# Stores the outgoing summation of weights * feautres 
		self.data = None
		
		# Initializes the Weights and Biases using a Normal Distribution with Mean 0 and Standard Deviation 0.1
		self.weights = torch.random.normal(0,0.1, (self.out_depth, self.in_depth, self.filter_row, self.filter_col))	
		self.biases = torch.random.normal(0,0.1,self.out_depth)
		

	def forwardpass(self, X):
		n = X.size()[0]  # batch size
		
		# self.data = torch.zeros(n,self.out_depth, self.out_row, self.out_col,dtype=dtype, device=device)
		# for t in range(n):
		# 	for j in range(0,self.out_row):
		# 		for k in range(0,self.out_col):
		# 			j1 = self.stride*j
		# 			k1 = self.stride*k
		# 			itorch = torch.sum(torch.sum(torch.sum(self.weights*(X[t,:,j1:j1+self.filter_row,k1:k1+self.filter_col]),dim=3),dims=2),dim=1)
		# 			self.data[t,:,j,k] = (itorch+self.biases)

		self.data = F.conv2d(X, self.weights, padding=None)

		return sigmoid(self.data)


	def backwardpass(self, lr, activation_prev, delta):
		n = activation_prev.shape[0] # batch size
		j=0
		k=0
		new_delta = torch.zeros((n,self.in_depth,self.in_row,self.in_col))

		grad_in = delta*derivative_sigmoid(self.data)

		for t in range(n):
			for i in range(self.out_depth):
				for j in range(self.out_row):
					for k in range(self.out_col):
						new_delta[t,:,j*self.stride:j*self.stride+self.filter_row,
						k*self.stride:k*self.stride+self.filter_col] += (grad_in[t][i][j][k]*self.weights[i]) 

		for i in range(self.out_depth):
			self.biases[i] -= lr*sum(sum(sum(grad_in[:,i,:,:])))
			for j in range(self.out_row):
				for k in range(self.out_col):
					self.weights[i] -= lr*torch.transpose(torch.mm(torch.transpose(activation_prev[:,:,j*self.stride:j*self.stride+self.filter_row,k*self.stride:k*self.stride+self.filter_col])
						,grad_in[:,i,j,k])) 

		return new_delta


	def clearGradParam(self):
		self.gradW = torch.zeros(out_neurons, in_neurons) # out_neurons X in_neurons
		self.gradB = torch.zeros(out_neurons, 1) # out_neurons X 1

	def dispGradParam(self):
		print("Linear Layer")

	def updateParam(self, learningRate):
		self.W -= learningRate*self.gradW
		self.B -= learningRate*self.gradB
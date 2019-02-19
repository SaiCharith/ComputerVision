import numpy as np
import torch
import math

dtype = torch.double
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Linear:
	def __init__(self, in_neurons, out_neurons):
		self.out_neurons = out_neurons
		self.in_neurons = in_neurons
		self.layerName = 'linear'

		self.output = None # batch_size X out_neurons
		self.W = torch.randn(out_neurons, in_neurons, dtype=dtype, device=device)*math.sqrt(2.0/self.in_neurons) # out_neurons X in_neurons
		self.B = torch.randn(out_neurons, 1, dtype=dtype, device=device)*math.sqrt(2.0/self.in_neurons) # out_neurons X 1
		self.gradW = torch.zeros(out_neurons, in_neurons, dtype=dtype, device=device) # out_neurons X in_neurons
		self.gradB = torch.zeros(out_neurons, 1, dtype=dtype, device=device) # out_neurons X 1
		self.gradInput = None # batch_size X in_neurons
		self.momentumW = torch.zeros(out_neurons, in_neurons, dtype=dtype, device=device)
		self.momentumB = torch.zeros(out_neurons, 1, dtype=dtype, device=device)
	
	def forward(self, input):
		self.output = input.mm(self.W.transpose(0,1)).add(self.B.transpose(0,1))
		return self.output

	def backward(self, input, gradOutput):
		self.gradInput = gradOutput.mm(self.W)
		self.gradB = gradOutput.sum(dim = 0).reshape(self.out_neurons,1)
		self.gradW = gradOutput.transpose(0,1).mm(input)
		return self.gradInput

	def clearGradParam(self):
		self.gradW = torch.zeros(out_neurons, in_neurons) # out_neurons X in_neurons
		self.gradB = torch.zeros(out_neurons, 1) # out_neurons X 1

	def dispGradParam(self):
		print("Linear Layer")

	def updateParam(self, learningRate, alpha=0):
		self.W += self.momentumW
		self.B += self.momentumB
		self.momentumW = alpha*self.momentumW - learningRate*self.gradW
		self.momentumB = alpha*self.momentumB - learningRate*self.gradB

	def set_W(self,W):
		self.W=W
	def set_B(self,b):
		self.B=b


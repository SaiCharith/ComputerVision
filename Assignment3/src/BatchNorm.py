import numpy as np
import torch
dtype = torch.double
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class BatchNorm:
	def __init__(self):
		self.output = None
		self.gradInput = None
		self.layerName = 'BatchNorm'
		self.batch_mean=None
		self.batch_std=None
	def forward(self, input,isTrain=False):
		self.batch_mean=input.mean(dim=0)
		self.batch_std=input.std(dim=0)
		self.output = (input-self.batch_mean)/self.batch_std
		# self.output[input<0] = 0
		return self.output
		# print("BatchNorm Layer Forward")
	
	def backward(self, input, gradOutput):
		myB=input.size()[0]
		self.gradInput = ((1-(1/myB))/self.batch_std - ((input-self.batch_mean)**2)/((self.batch_std**3)*myB))* gradOutput
		# self.gradInput[input <= 0] = 0
		# print("BatchNorm Layer backward")
		return self.gradInput

	def clearGradParam(self):
		return

	def dispGradParam(self):
		print("BatchNorm Layer")

	def updateParam(self, learningRate, alpha,regularizer=0):
		# print("BatchNorm Layer Update Weights & Biases: ")
		return
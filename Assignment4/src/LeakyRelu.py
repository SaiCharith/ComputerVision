import numpy as np
import torch
dtype = torch.double
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class LeakyRelu:
	def __init__(self,leak=1e-2):
		self.output = None
		self.gradInput = None
		self.leak=leak
		self.layerName = 'LeakyRelu'

	def forward(self, input,isTrain=False):
		self.output = input
		self.output[input<0] = self.leak*self.output[input<0]
		return self.output
		# print("LeakyRelu Layer Forward")
	
	def backward(self, input, gradOutput):
		self.gradInput = gradOutput
		self.gradInput[input <= 0] = self.leak*gradOutput[input<=0]
		# print("LeakyRelu Layer backward")
		return self.gradInput

	def clearGradParam(self):
		return

	def dispGradParam(self):
		print("LeakyRelu Layer")

	def updateParam(self, learningRate, alpha,regularizer=0):
		# print("LeakyRelu Layer Update Weights & Biases: ")
		return
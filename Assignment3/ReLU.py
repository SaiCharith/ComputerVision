import numpy as np
import torch
dtype = torch.double
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReLU:
	def __init__(self):
		self.output = None
		self.gradInput = None

	def forward(self, input):
		self.output = input
		self.output[input<0] = 0.0
	
	def backward(self, input, gradOutput):
		self.gradInput = gradOutput
		self.gradInput[input < 0] = 0.0
		return self.gradInput

	def clearGradParam(self):
		return

	def dispGradParam(self):
		print("ReLU Layer")

	def updateParam(self, learningRate):
		return
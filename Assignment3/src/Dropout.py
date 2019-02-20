import numpy as np
import torch
dtype = torch.double
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Dropout:
	def __init__(self,drop_rate=0.5):
		self.output = None
		self.gradInput = None
		self.drop_rate=drop_rate
		self.layerName = 'Dropout'
		self.pass_ind=None
		self.istrain = False
	def forward(self, input,istrain=False):

		self.output= input
		self.istrain = istrain
		if istrain:
			self.pass_ind=torch.rand(input.size()[1],dtype=dtype,device=device)
			self.pass_ind[self.pass_ind>self.drop_rate]=-1
			self.output[:,(self.pass_ind==-1).type(torch.ByteTensor)] = 0
			self.output *= (1/self.drop_rate)

		return self.output

	
	def backward(self, input, gradOutput):
		self.gradInput= gradOutput
		if self.istrain:
			self.gradInput*=(1/self.drop_rate)
			self.gradInput[:,(self.pass_ind==-1).type(torch.ByteTensor)] = 0
		return self.gradInput

	def clearGradParam(self):
		return

	def dispGradParam(self):
		print("Dropout Layer")

	def updateParam(self, learningRate, alpha,regularizer=0):
		# print("Dropout Layer Update Weights & Biases: ")
		return
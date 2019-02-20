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
	def forward(self, input,istrain=False):

		self.output= input*(1/self.drop_rate)
		if istrain:
			self.pass_ind=torch.rand(input.size()[1],dtype=dtype,device=device)
			self.pass_ind[self.pass_ind>self.drop_rate]=0
			# self.pass_ind[self.pass_ind>self.drop_rate]=1
			self.output[:,(self.pass_ind==0).type(torch.ByteTensor)] = 0

		# self.output[input<0] = 0
		return self.output
		# print("Dropout Layer Forward")
	
	def backward(self, input, gradOutput):
		self.gradInput= gradOutput*(1/self.drop_rate)
		self.gradInput[:,(self.pass_ind==0).type(torch.ByteTensor)] = 0
		# self.gradInput[:, <= 0] = 0
		# print("Dropout Layer backward")
		return self.gradInput

	def clearGradParam(self):
		return

	def dispGradParam(self):
		print("Dropout Layer")

	def updateParam(self, learningRate, alpha,regularizer=0):
		# print("Dropout Layer Update Weights & Biases: ")
		return
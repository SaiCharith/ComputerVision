import numpy as np
import torch
import math
import ReLU

dtype = torch.double
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class RNN:
	def __init__(self, input_dim, hidden_dim,output_dim):
		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.output_dim = output_dim

		self.weights_hh = torch.randn(hidden_dim, hidden_dim, dtype=dtype, device=device)*math.sqrt(2.0/self.hidden_dim)
		self.weights_hx = torch.randn(hidden_dim, input_dim, dtype=dtype, device=device)*math.sqrt(2.0/self.hidden_dim)
		self.weights_hy = torch.randn(output_dim, hidden_dim, dtype=dtype, device=device)*math.sqrt(2.0/self.hidden_dim)
		self.bias_h = torch.randn(hidden_dim, 1, dtype=dtype, device=device)*math.sqrt(2.0/self.hidden_dim) # hidden_dim X 1
		self.bias_y = torch.randn(output_dim, 1, dtype=dtype, device=device)*math.sqrt(2.0/self.hidden_dim) # output_dim X 1

		self.y = None
		self.h = None
		self.x = None

		self.grad_Whh = None
		self.grad_Whx = None
		self.grad_Why = None
		self.grad_bias_h = None
		self.grad_bias_y = None
		self.grad_inp = None
		self.grad_prev = None
		self.r = ReLU.ReLU()
	
	def forward(self, input,isTrain=False):
		# if istrain:
		self.y =[]
		self.h =[torch.zeros(input[0].size()[0] , self.hidden_dim, dtype=dtype, device=device)]
		self.prev_h = []
		self.x = input
		
		for i in range(len(input)):

			Whh_h = self.h[-1].mm(self.weights_hh.transpose(0,1))        #  batch X hidden
			# print(Whh_h.size())
			Wxh_x = input[i].mm(self.weights_hx.transpose(0,1)) #  batch X hidden
			# print(Wxh_x.size())
			ht = Whh_h.add(Wxh_x)									#  batch X hidden
			ht = ht.add(self.bias_h.transpose(0,1))					#  batch X hidden				
			ht = self.r.forward(ht)										#  batch X hidden
			self.h.append(ht)

			# self.h.append(r.forward(.add(input[:,i,:].mm(self.weights_xh)).add(self.grad_bias_y)))
			yt = ht.mm(self.weights_hy.transpose(0,1)).add(self.bias_y.transpose(0,1)) # batch X output
			self.y.append(yt)
		# print (self.y)
		return self.y


	def backward(self, input, gradOutput):

		grad_ht = torch.zeros((input[0].size()[0]),self.hidden_dim)			# batch X hidden
		grad_x = [None for _ in range(len(input))]

		for i in reversed(range(len(input))):
			grad_y = gradOutput[i]								# batch X output_dim
			self.grad_bias_y = self.grad_bias_y.add(grad_y.sum(dim=0).reshape(self.output_dim,1))
			self.grad_Why = self.grad_Why.add(grad_y.transpose(0,1).mm(self.h[i]))  # output X hidden

			grad_act = self.r.backward(grad_ht,self.h[i]) + grad_y.mm(self.weights_hy)	# batch X hidden
			self.grad_bias_h = self.grad_bias_h.add(grad_act.sum(dim=0).reshape(self.hidden_dim,1)) # hidden X 1
			self.grad_Whh = self.grad_Whh.add(grad_act.transpose(0,1).mm(self.h[i-1]))
			# print(self.grad_Whx.size(),grad_act.size(),input[i].size())
			self.grad_Whx = self.grad_Whx.add(grad_act.transpose(0,1).mm(input[i]))   # hidden X input

			grad_x[i] = grad_act.mm(self.weights_hx)
			grad_ht = grad_act.mm(self.weights_hh)

		return grad_x

	def clearGradParam(self):
		self.grad_Whh = torch.zeros(self.hidden_dim, self.hidden_dim, dtype=dtype, device=device)
		self.grad_Whx = torch.zeros(self.hidden_dim, self.input_dim, dtype=dtype, device=device)
		self.grad_Why = torch.zeros(self.output_dim, self.hidden_dim, dtype=dtype, device=device)
		self.grad_bias_h = torch.zeros(self.hidden_dim, 1, dtype=dtype, device=device)
		self.grad_bias_y = torch.zeros(self.output_dim, 1, dtype=dtype, device=device)

	def updateParam(self, learningRate, alpha=0, regularizer=0):

		self.weights_hh += self.grad_Whh
		self.weights_hx += self.grad_Whx
		self.weights_hy += self.grad_Why
		self.bias_h += self.grad_bias_h
		self.bias_y += self.grad_bias_y

		# self.W += (self.momentumW -2*regularizer*self.W)
		# self.B += (self.momentumB -2*regularizer*self.B)
		# self.momentumW = alpha*self.momentumW - learningRate*self.gradW
		# self.momentumB = alpha*self.momentumB - learningRate*self.gradB



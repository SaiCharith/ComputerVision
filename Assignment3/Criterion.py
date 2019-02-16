import numpy as np
import torch
dtype = torch.double
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def pause():
	print("press enter")
	input()


class Criterion:
	def __init__(self):
		return

	def forward(self, input, target):
		batch_size, num_classes = input.size()
		
		inputExp = input.exp()
		self.probabilities = inputExp/inputExp.sum(dim=1).unsqueeze(1)
		loss = -self.probabilities.log()
		return torch.sum(loss*torch.eye(num_classes, dtype=dtype, device=device)[target])/batch_size

	def backward(self, input, target):
		batch_size, num_classes = input.size()
		inputExp = input.exp()
		probabilities = inputExp/(inputExp.sum(dim=1).unsqueeze(1))
		probabilities[torch.isnan(probabilities)] = 1
		grads = (probabilities - torch.eye(num_classes, dtype=dtype, device=device)[target])/batch_size
		return grads
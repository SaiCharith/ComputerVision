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

		probabilities = inputExp/inputExp.sum(dim=1).unsqueeze(1)
		probabilities[torch.isnan(probabilities)] = 1
		probabilities = probabilities/(probabilities.sum(dim=1).unsqueeze(1))
		loss = -probabilities.log()
		labels = torch.eye(num_classes, device=device, dtype=dtype)[target]
		loss[labels!=1] = 0 
		loss = loss/batch_size
		avgLoss = torch.sum(loss)
		return avgLoss


	def backward(self, input, target):
		batch_size, num_classes = input.size()
		inputExp = input.exp()
		probabilities = inputExp/(inputExp.sum(dim=1).unsqueeze(1))
		probabilities[torch.isnan(probabilities)] = 1
		probabilities = probabilities/(probabilities.sum(dim=1).unsqueeze(1))
		labels = torch.eye(num_classes, device=device, dtype=dtype)[target]
		grads = (probabilities - labels)/batch_size
		return grads
import numpy as np
import torch
import Linear
import ReLU
import Criterion

dtype = torch.double
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def pause():
	print("press enter")
	input()

class Model:
	def __init__(self):
		self.Layers = []
		self.isTrain = True

	def forward(self, input):
		# print("Forwarding: ")
		for layer in self.Layers:
			layer.forward(input)
			input = layer.output
		return input

	def backward(self, input, gradOutput):
		# print("Backpropogation: ")
		for i in range(len(self.Layers) - 1):
			inputPrev = self.Layers[-i-2].output
			gradOutput = self.Layers[-i-1].backward(inputPrev, gradOutput)
		gradOutput = self.Layers[0].backward(input, gradOutput)

	def updateParam(self, learningRate):
		# print("Updating Weights & Biases: ")
		for layer in self.Layers:
			# layer.dispGradParam()
			layer.updateParam(learningRate)
			# layer.dispGradParam()

	def dispGradParam(self):
		for i in range(len(self.Layers)):
			self.Layers[-i-1].dispGradParam()

	def clearGradParam(self):
		for layer in self.Layers:
			layer.clearGradParam()

	def addLayer(self, layer):
		self.Layers.append(layer)

	def trainModel(self, learningRate, batchSize, epochs, trainingData, trainingLabels):
		trainingDataSize = trainingData.size()[0]
		criterion = Criterion.Criterion()
		numBatches = trainingDataSize//batchSize + 1*(trainingDataSize%batchSize!=0)
		for i in range(epochs):
			print("Epoch ", i)
			for j in range(numBatches):
				activations = self.forward(trainingData[batchSize*j:(j+1)*batchSize])
				gradOutput = criterion.backward(activations, trainingLabels[batchSize*j:(j+1)*batchSize])
				# print("BatchLoss: ",criterion.forward(activations, trainingLabels[batchSize*j:(j+1)*batchSize]).item())
				self.backward(trainingData[batchSize*j:(j+1)*batchSize], gradOutput)
				self.updateParam(learningRate/((i+1)**0.7))

			predictions = self.classify(trainingData)
			print(0, torch.sum(predictions == 0).item())
			print(1, torch.sum(predictions == 1).item())
			print(2, torch.sum(predictions == 2).item())
			print(3, torch.sum(predictions == 3).item())
			print(4, torch.sum(predictions == 4).item())
			print(5, torch.sum(predictions == 5).item())
			print(torch.sum(predictions == trainingLabels).item())
			# print(list(zip(predictions,trainingLabels)))
			print(0,torch.sum(predictions == 0).item())
			print(1,torch.sum(predictions == 1).item())
			print(2,torch.sum(predictions == 2).item())
			print(3,torch.sum(predictions == 3).item())
			print(4,torch.sum(predictions == 4).item())
			print(5,torch.sum(predictions == 5).item())
			print(6,torch.sum(predictions == 6).item())
			print("Training Loss",criterion.forward(self.forward(trainingData), trainingLabels).item())
			print("Training Accuracy: ", (torch.sum(predictions == trainingLabels).item()*100.0/trainingLabels.size()[0]))

	def classify(self, data):
		guesses = self.forward(data)
		value, indices = torch.max(guesses,dim=1)

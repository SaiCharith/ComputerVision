import numpy as np
import torch
import Linear
import ReLU
import Criterion

dtype = torch.double
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Model:
	def __init__(self):
		self.Layers = []
		self.isTrain = True

	def forward(self, input):
		for layer in self.Layers:
			layer.forward(input)
			input = layer.output
		return input

	def backward(self, input, gradOutput):
		for i in range(len(self.Layers) - 1):
			inputPrev = self.Layers[-i-2].output
			gradOutput = self.Layers[-i-1].backward(inputPrev, gradOutput)
		gradOutput = self.Layers[0].backward(input, gradOutput)


	def updateParam(self, learningRate):
		for layer in self.Layers:
			layer.updateParam(learningRate)

	def dispGradParam(self):
		for i in range(len(Self.Layers)):
			self.Layers[-i-1].dispGradParam

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
				self.backward(trainingData[batchSize*j:(j+1)*batchSize], gradOutput)
				self.updateParam(learningRate)
			predictions = self.classify(trainingData)
			print("Training Accuracy: ", (tensor.sum(predictions == trainingLabels)/trainingLabels.size()[0])*100)

	def classify(self, data):
		guesses = self.forward(data)
		value,indices = torch.max(guesses,dim=1)
		return indices
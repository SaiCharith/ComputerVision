import numpy as np
import torch
import torchfile
import Model
import Linear
import ReLU
import random

dtype = torch.double
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def loadData():
	TRAINING_DATA = "Train/data.bin"
	TRAINING_LABELS = "Train/labels.bin"
	TESTING_DATA = "Test/test.bin"

	Data = torch.tensor(torchfile.load(TRAINING_DATA), dtype=dtype, device=device)
	Data = Data/256.0
	Labels = torch.tensor(torchfile.load(TRAINING_LABELS), dtype=torch.long, device=device)

	Data = Data/256.0

	SIZE = Data.size()[0]
	HEIGHT = Data.size()[1]
	WIDTH = Data.size()[2]
	TRAINING_SIZE = int(0.7*SIZE)
	VALIDATION_SIZE = SIZE - TRAINING_SIZE

	Data = Data.reshape(SIZE, HEIGHT*WIDTH)
	indices = list(range(SIZE))
	random.shuffle(indices)

	trainingData = Data[indices[0:TRAINING_SIZE]]
	trainingLabels = Labels[indices[0:TRAINING_SIZE]]
	validationData = Data[indices[TRAINING_SIZE:]]
	validationLabels = Labels[indices[TRAINING_SIZE:]]

	return trainingData, trainingLabels, validationData, validationLabels

def createModel():
	trainingData, trainingLabels, validationData, validationLabels = loadData()

	neuralNetwork = Model.Model()
	neuralNetwork.addLayer(Linear.Linear(108*108, 100))
	neuralNetwork.addLayer(ReLU.ReLU())
	neuralNetwork.addLayer(Linear.Linear(100, 6))
	learningRate = 0.01
	batchSize = 30
	epochs = 50

	neuralNetwork.trainModel(learningRate, batchSize, epochs, trainingData, trainingLabels)

	predictions = neuralNetwork.classify(validationData)
	print(torch.sum(predictions == validationLabels).item())
	print("Validation Accuracy: ", (torch.sum(predictions == validationLabels).item()*100.0/validationLabels.size()[0]))

createModel()










# neuralNetwork = Model.Model()
# neuralNetwork.addLayer(Linear.Linear(108*108,356))
# neuralNetwork.addLayer(ReLU.ReLU())
# neuralNetwork.addLayer(Linear.Linear(356, 6))

# learningRate = 0.008
# batchSize = 30
# epochs = 100
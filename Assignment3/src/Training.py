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
	# TRAINING_DATA = "Train/data.bin"
	TRAINING_DATA = "input_sample_1.bin"
	
	# TRAINING_LABELS = "Train/labels.bin"
	TRAINING_LABELS = "target_sample_1.bin"
	TESTING_DATA = "Test/test.bin"

	Data = torch.tensor(torchfile.load(TRAINING_DATA), dtype=dtype, device=device)
	Labels = torch.tensor(torchfile.load(TRAINING_LABELS), dtype=torch.long, device=device)

	Data = Data/(256.0)
	print (Data.size())
	SIZE = Data.size()[0]
	# HEIGHT = Data.size()[1]
	# WIDTH = Data.size()[2]
	# # DEPTH = Data.size()[3]
	TRAINING_SIZE = int(0.7*SIZE)
	VALIDATION_SIZE = int(0.3*SIZE)
	sz=1
	for i in Data.size():
		sz=sz*i

	Data = Data.reshape(SIZE, int(sz/SIZE))
	indices = list(range(SIZE))
	random.shuffle(indices)
  


	trainingData = Data[indices[0:TRAINING_SIZE]]
	trainingMean = trainingData.mean(dim=0)
	trainingLabels = Labels[indices[0:TRAINING_SIZE]]
	validationData = Data[indices[TRAINING_SIZE:]]
	validationLabels = Labels[indices[TRAINING_SIZE:]]

	return trainingData, trainingLabels, validationData, validationLabels, trainingMean


def createModel():
	trainingData, trainingLabels, validationData, validationLabels, trainingMean = loadData()

	trainingData = trainingData - trainingMean
	validationData = validationData - trainingMean

	neuralNetwork = Model.Model()
	# neuralNetwork.addLayer(Linear.Linear(108*108,6))
	# neuralNetwork.addLayer(ReLU.ReLU())
	# neuralNetwork.addLayer(Linear.Linear(1002, 154))
	# neuralNetwork.addLayer(ReLU.ReLU())
	# neuralNetwork.addLayer(Linear.Linear(1002,1002))
	# neuralNetwork.addLayer(ReLU.ReLU())
	# neuralNetwork.addLayer(Linear.Linear(1002,501))
	# neuralNetwork.addLayer(ReLU.ReLU())
	# neuralNetwork.addLayer(Linear.Linear(501, 309))
	# neuralNetwork.addLayer(ReLU.ReLU())
	# neuralNetwork.addLayer(Linear.Linear(309, 102))
	# neuralNetwork.addLayer(ReLU.ReLU())
	# neuralNetwork.addLayer(Linear.Linear(102,54))
	# neuralNetwork.addLayer(ReLU.ReLU())
	# neuralNetwork.addLayer(Linear.Linear(154,6))


	learningRate = 0.01
	batchSize = 20
	epochs = 1
	alpha = 0.5

	# neuralNetwork.trainModel(learningRate, batchSize, epochs, trainingData, trainingLabels, alpha)

	# neuralNetwork.saveModel("Model.txt","ModelWeights.bin","ModelBiases.bin")
	neuralNetwork.loadModel("modelConfig_1.txt")
	# neuralNetwork.trainModel(learningRate, batchSize, epochs, trainingData, trainingLabels, alpha)

	predictions = neuralNetwork.classify(validationData)
	print(torch.sum(predictions == validationLabels).item())
	print("Validation Accuracy: ", (torch.sum(predictions == validationLabels).item()*100.0/validationLabels.size()[0]))

createModel()







# neuralNetwork.addLayer(Linear.Linear(108*108,1002))
# neuralNetwork.addLayer(ReLU.ReLU())
# neuralNetwork.addLayer(Linear.Linear(1002, 1002))
# neuralNetwork.addLayer(ReLU.ReLU())
# neuralNetwork.addLayer(Linear.Linear(1002,1002))
# neuralNetwork.addLayer(ReLU.ReLU())
# neuralNetwork.addLayer(Linear.Linear(1002,501))
# neuralNetwork.addLayer(ReLU.ReLU())
# neuralNetwork.addLayer(Linear.Linear(501, 309))
# neuralNetwork.addLayer(ReLU.ReLU())
# neuralNetwork.addLayer(Linear.Linear(309, 102))
# neuralNetwork.addLayer(ReLU.ReLU())
# neuralNetwork.addLayer(Linear.Linear(102,54))
# neuralNetwork.addLayer(ReLU.ReLU())
# neuralNetwork.addLayer(Linear.Linear(54,6))
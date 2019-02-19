import sys
import os
sys.path.insert(0, './src')

import Linear
import ReLU
import Model
import BatchNorm

import argparse
import torch
import torchfile
import random

dtype = torch.double
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def loadData(dataPath,labelsPath):
	
	TRAINING_DATA = dataPath
	TRAINING_LABELS = labelsPath
	Data = torch.tensor(torchfile.load(TRAINING_DATA), dtype=dtype, device=device)
	Labels = torch.tensor(torchfile.load(TRAINING_LABELS), dtype=torch.long, device=device)
	
	dataSize = Data.size()
	Data = Data/(256.0)
	SIZE = dataSize[0]

	TRAINING_SIZE = int(0.7*SIZE)
	VALIDATION_SIZE = int(0.3*SIZE)

	featureSize = 1
	for i in range(1,len(dataSize)):
		featureSize*=dataSize[i]

	Data = Data.reshape(SIZE, featureSize)
	indices = list(range(SIZE))
	random.shuffle(indices)

	trainingData = Data[indices[0:TRAINING_SIZE]]
	trainingMean = trainingData.mean(dim=0)
	trainingLabels = Labels[indices[0:TRAINING_SIZE]]
	validationData = Data[indices[TRAINING_SIZE:]]
	validationLabels = Labels[indices[TRAINING_SIZE:]]

	return trainingData, trainingLabels, validationData, validationLabels, trainingMean

if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-modelName', help='Give Model Name',dest ="modelName",default='model')
	parser.add_argument('-data', help='Give input.bin path',dest ="dataPath",default='./Train/input.bin')
	parser.add_argument('-target', help='give gradOutput.bin path',dest ="labelsPath",default='./Train/labels.bin')

	args = parser.parse_args()
	trainingData, trainingLabels, validationData, validationLabels, trainingMean = loadData(args.dataPath,args.labelsPath)

	trainingData -= trainingMean
	validationData -= trainingMean

	neuralNetwork = Model.Model()
	neuralNetwork.addLayer(BatchNorm.BatchNorm())
	neuralNetwork.addLayer(Linear.Linear(108*108,6))
	neuralNetwork.addLayer(ReLU.ReLU())
	neuralNetwork.addLayer(Linear.Linear(1024,6))

	learningRate = 0.01
	batchSize = 20
	epochs = 20
	alpha = 0.5

	neuralNetwork.trainModel(learningRate, batchSize, epochs, trainingData, trainingLabels, alpha)

	predictions = neuralNetwork.classify(validationData)
	print(torch.sum(predictions == validationLabels).item())
	print("Validation Accuracy: ", (torch.sum(predictions == validationLabels).item()*100.0/validationLabels.size()[0]))

	directory = "./"+args.modelName+"/"
	if not os.path.exists(directory):
		os.makedirs(directory)

	# neuralNetwork.saveModel(directory+"modelConfig.txt",directory+"ModelWeights.bin",directory+"ModelBiases.bin")
	# torch.save(trainingMean,directory+"trainingMean.bin")
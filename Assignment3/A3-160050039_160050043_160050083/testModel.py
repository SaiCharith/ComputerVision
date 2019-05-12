import sys
import os
sys.path.insert(0, './src')

import Linear
import ReLU
import Model

import argparse
import torch
import torchfile
import random

dtype = torch.double
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def loadData(dataPath):

	Data = torch.tensor(torchfile.load(dataPath), dtype=dtype, device=device)
	
	dataSize = Data.size()
	Data = Data/(256.0)
	SIZE = dataSize[0]


	featureSize = 1
	for i in range(1,len(dataSize)):
		featureSize*=dataSize[i]

	testData = Data.reshape(SIZE, featureSize)

	return testData

if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-modelName', help='Give Model Name',dest ="modelName",default='model')
	parser.add_argument('-data', help='Give input.bin path',dest ="dataPath",default='./Test/test.bin')

	args = parser.parse_args()
	testData = loadData(args.dataPath)

	directory = "./"+args.modelName+"/"
	trainingMean = torch.load(directory+"trainingMean.bin",map_location= {'cuda:0':device.type})
	testData -= trainingMean

	neuralNetwork = Model.Model()
	neuralNetwork.loadModel(directory+"modelConfig.txt")


	predictions = neuralNetwork.classify(testData)
	torch.save(predictions,"testPrediction.bin")


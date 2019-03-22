import sys
import os
sys.path.insert(0, './src')

import Linear
import ReLU
import Model
# import BatchNorm

import argparse
import torch
import torchfile
import random
import Dropout
import LeakyRelu
import RNN
import numpy as np

dtype = torch.double
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def loadData(dataPath,labelsPath):
	
	TRAINING_DATA = dataPath
	TRAINING_LABELS = labelsPath
	Data= []
	with open(TRAINING_DATA) as inputfile:
	    for line in inputfile:
	        Data.append([int(m) for m in line.strip().split(' ')])
	Labels= []
	with open(TRAINING_LABELS) as inputfile:
	    for line in inputfile:
	        Labels.append([int(m) for m in line.strip().split(' ')][0])

	
	SIZE = len(Data)
	flattened = [val for sublist in Data for val in sublist]
	unique_labels=list(np.unique(flattened))

	TRAINING_SIZE = int(0.07*SIZE)
	VALIDATION_SIZE = int(0.3*SIZE)

	indices = list(range(SIZE))
	random.shuffle(indices)
	Data=np.array(Data)
	Labels=np.array(Labels,dtype='int')
	trainingData = list(Data[indices[0:TRAINING_SIZE]])
	trainingLabels = list(Labels[indices[0:TRAINING_SIZE]])
	validationData = list(Data[indices[TRAINING_SIZE:]])
	validationLabels = list(Labels[indices[TRAINING_SIZE:]])


	return trainingData, trainingLabels, validationData, validationLabels,unique_labels

if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-modelName', help='Give Model Name',dest ="modelName",default='model')
	parser.add_argument('-data', help='Give input.bin path',dest ="dataPath",default='./data/train_data.txt')
	parser.add_argument('-target', help='give gradOutput.bin path',dest ="labelsPath",default='./data/train_labels.txt')

	args = parser.parse_args()
	trainingData, trainingLabels, validationData, validationLabels, unique_labels = loadData(args.dataPath,args.labelsPath)



	batchSize = 20
	epochs = 50
	lr = 0.00001
	reg = 0.000001
	al = 0.7
	leak = 0.01
	dropout_rate = 0.75
	
	neuralNetwork = Model.Model()
	# neuralNetwork.addLayer(Linear.Linear(108*108,1024))
	# neuralNetwork.addLayer(Dropout.Dropout(dropout_rate))
	# neuralNetwork.addLayer(LeakyRelu.LeakyRelu(leak))
	# neuralNetwork.addLayer(Linear.Linear(1024, 512))
	# neuralNetwork.addLayer(Dropout.Dropout(dropout_rate))
	# neuralNetwork.addLayer(LeakyRelu.LeakyRelu(leak))
	# neuralNetwork.addLayer(Linear.Linear(512, 512))
	# neuralNetwork.addLayer(Dropout.Dropout(dropout_rate))
	# neuralNetwork.addLayer(LeakyRelu.LeakyRelu(leak))
	# neuralNetwork.addLayer(Linear.Linear(512,6))
	neuralNetwork.addLayer(RNN.RNN(len(unique_labels),64,2))
	neuralNetwork.trainModel(lr, batchSize, epochs, trainingData,unique_labels, trainingLabels, al,reg)


	# directory = "./"+args.modelName+"/"
	# if not os.path.exists(directory):
	# 	os.makedirs(directory)

	# torch.save(trainingMean,directory+"trainingMean.bin")
	# neuralNetwork.saveModel(directory+"bestModalConfig.txt",directory+"ModalWeights.bin",directory+"ModelBais.bin")


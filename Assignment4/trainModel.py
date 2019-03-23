import sys
import os
sys.path.insert(0, './src')

import Model
# import BatchNorm

import argparse
import torch
import random
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

	TRAINING_SIZE = int(0.7*SIZE)
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
	parser.add_argument('-modelName', help='Give Model Name',dest ="modelName",default='bestModel')
	parser.add_argument('-data', help='Give input.bin path',dest ="dataPath",default='./data/train_data.txt')
	parser.add_argument('-target', help='give gradOutput.bin path',dest ="labelsPath",default='./data/train_labels.txt')

	args = parser.parse_args()
	random.seed(0)
	trainingData, trainingLabels, validationData, validationLabels, unique_labels = loadData(args.dataPath,args.labelsPath)

	

	batchSize = 3
	epochs = 50
	lr = 1.0e-1
	reg = 1.0e-4
	al = 0.7
	# leak = 0.01
	# dropout_rate = 0.75
	
	neuralNetwork = Model.Model()
	# neuralNetwork.loadModel('bestModel/bestModalConfig.txt','bestModel/ModalWeights.bin')
	neuralNetwork.addLayer(RNN.RNN(len(unique_labels),16,2))
	# neuralNetwork.addLayer(RNN.RNN(16,16,2))
	# neuralNetwork.addLayer(RNN.RNN(16,64,2))
	neuralNetwork.trainModel(lr, batchSize, epochs, trainingData,unique_labels, trainingLabels,al,reg,validationData, validationLabels)


	directory = "./"+args.modelName+"/"
	if not os.path.exists(directory):
		os.makedirs(directory)

	neuralNetwork.saveModel(directory+"bestModalConfig.txt",directory+"ModalWeights.bin")



	# batchSize = 3
	# epochs = 50
	# lr = 1.0e-1
	# reg = 0
	# al = 0.7
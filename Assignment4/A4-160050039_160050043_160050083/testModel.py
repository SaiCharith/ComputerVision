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


def loadData(dataPath):
	
	TRAINING_DATA = dataPath
	
	Data= []
	with open(TRAINING_DATA) as inputfile:
	    for line in inputfile:
	        Data.append([int(m) for m in line.strip().split(' ')])
	
	SIZE = len(Data)
	flattened = [val for sublist in Data for val in sublist]
	unique_labels=list(np.unique(flattened))

	Data=np.array(Data)

	Data = list(Data)

	return Data

if __name__=='__main__':
	
	parser = argparse.ArgumentParser()
	parser.add_argument('-modelName', help='Give Model Name',dest ="modelName",default='model')
	parser.add_argument('-data', help='Give input.bin path',dest ="dataPath",default='./Test/test.bin')

	args = parser.parse_args()
	testData = loadData(args.dataPath)

	directory = "./"+args.modelName+"/"


	neuralNetwork = Model.Model()
	neuralNetwork.loadModel(directory+'bestModelConfig.txt',directory+'ModelWeights.bin')

	predictions = neuralNetwork.gettestPredictions(testData)
	torch.save(predictions,"testPrediction.bin")


	# f= open("testPrediction.txt","w+")
	# f.write("id,label\n")
	# for i in range(predictions.size()[0]):
	#   f.write(str(i)+","+str(predictions[i].item())+"\n")
	# f.close()



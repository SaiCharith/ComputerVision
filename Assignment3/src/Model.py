import numpy as np
import torch
import Linear
import ReLU
import Criterion
import torchfile

dtype = torch.double
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def pause():
	print("press enter")
	input()

class Model:
	def __init__(self):
		self.Layers = []
		self.isTrain = True

	def forward(self, input,isTrain=False):
		# print("Forwarding: ")
		for layer in self.Layers:
			input = layer.forward(input,isTrain)
		return input

	def backward(self, input, gradOutput):
		# print("Backpropogation: ")
		for i in range(len(self.Layers) - 1):
			inputPrev = self.Layers[-i-2].output
			gradOutput = self.Layers[-i-1].backward(inputPrev, gradOutput)
		gradOutput = self.Layers[0].backward(input, gradOutput)
		return gradOutput

	def updateParam(self, learningRate, alpha, regularizer=0):
		# print("Updating Weights & Biases: ")
		for layer in self.Layers:
			# layer.dispGradParam()
			layer.updateParam(learningRate,alpha,regularizer)
			# layer.dispGradParam()

	def dispGradParam(self):
		for i in range(len(self.Layers)):
			self.Layers[-i-1].dispGradParam()

	def clearGradParam(self):
		for layer in self.Layers:
			layer.clearGradParam()

	def addLayer(self, layer):
		self.Layers.append(layer)

	def trainModel(self, learningRate, batchSize, epochs, trainingData, trainingLabels, alpha=0, regularizer=0):
		trainingDataSize = trainingData.size()[0]
		criterion = Criterion.Criterion()
		numBatches = trainingDataSize//batchSize + 1*(trainingDataSize%batchSize!=0)
		for i in range(epochs):
			print("Epoch ", i)
			for j in range(numBatches):
				activations = self.forward(trainingData[batchSize*j:(j+1)*batchSize],True)
				gradOutput = criterion.backward(activations, trainingLabels[batchSize*j:(j+1)*batchSize])
				# print("BatchLoss: ",criterion.forward(activations, trainingLabels[batchSize*j:(j+1)*batchSize]).item())
				self.backward(trainingData[batchSize*j:(j+1)*batchSize], gradOutput)
				self.updateParam(learningRate/((i+1)**0.7),alpha/((i+1)**0.7),regularizer)

			predictions = self.classify(trainingData)
			# print(torch.sum(predictions == trainingLabels).item())
			print("Training Loss",criterion.forward(self.forward(trainingData), trainingLabels).item())
			print("Training Accuracy: ", (torch.sum(predictions == trainingLabels).item()*100.0/trainingLabels.size()[0]))

	def classify(self, data):
		guesses = self.forward(data)
		value, indices = torch.max(guesses,dim=1)
		return indices

	def saveModel(self, filepath0, filePath1, filePath2):
		lW = []
		lB = []
		f= open(filepath0,"w+")
		f.write(str(len(self.Layers))+"\n")
		for layer in self.Layers:
			if layer.layerName == 'linear':
				f.write(layer.layerName+" "+str(layer.in_neurons)+" "+str(layer.out_neurons)+"\n")
				lW.append(layer.W)
				lB.append(layer.B)
			if layer.layerName == 'relu':
				f.write("relu"+"\n")
			if layer.layerName == 'Dropout':
				f.write("Dropout"+"\n")
			if layer.layerName == 'LeakyRelu':
				f.write("LeakyRelu"+"\n")
			if layer.layerName == 'BatchNorm':
				f.write("BatchNorm"+"\n")
		f.write(filePath1+"\n")
		f.write(filePath2)
		f.close()
		torch.save(lW,filePath1)
		torch.save(lB,filePath2)

	def loadModel(self,path_config):

		with open(path_config) as f:
			content = f.readlines()
			# you may also want to remove whitespace characters like \n at the end of each line
			content = [x.strip() for x in content]
		# print (content)
		no_layers=int(content[0])
		# print (no_layers)
		layer_w_path=content[-2]
		layer_bias_path=content[-1]
		# weights=[]
		# bias =[]
		try:
		 	bias = torch.load(layer_bias_path)
		 	weights = torch.load(layer_w_path)
		except:
			print("exept1")
			try:

				bias=torchfile.load(layer_bias_path)
				weights=torchfile.load(layer_w_path)
			except:
				print("exept2")
				pass
			pass

		indices=[]
		j = 0
		for i in range(1,len(content)-2):
			words=content[i].split()
			# print(words)
			if(words[0]=='linear'):
				in_nodes=int(words[1])
				out_nodes=int(words[2])
				# print("creating linear layer with " + str(in_nodes) +" "+str(out_nodes))
				self.addLayer(Linear.Linear(in_nodes,out_nodes))
				# print(self.Layers[-1].B.size())
				if type(self.Layers[-1].W)==type(weights[j]):
					self.Layers[-1].W = (weights[j])#.clone().detach().requires_grad_(True)
					self.Layers[-1].B = (bias[j]).reshape(self.Layers[-1].B.size())#.clone().detach()
				else:
					self.Layers[-1].W = torch.from_numpy(weights[j])#.clone().detach().requires_grad_(True)
					self.Layers[-1].B = torch.from_numpy(bias[j]).reshape(self.Layers[-1].B.size())#.clone().detach()
				j+=1
				# print(type(self.Layers[-1].B))#,self.Layers[-1].B.size())
				indices.append(i-1)
			elif(words[0]=='relu'):
				# print("creating relu layer")
				self.addLayer(ReLU.ReLU())
			elif(words[0]=='Dropout'):
				self.addLayer(Dropout.Dropout())
			elif(words[0]=='LeakyRelu'):
				self.addLayer(LeakyRelu.LeakyRelu())
			elif(words[0]=='BatchNorm'):
				self.addLayer(BatchNorm.BatchNorm())

			
	def save_Grads(self,path_w,path_b):
		lb=[]
		lw=[]
		for layer in self.Layers:
			if (layer.layerName=='linear'):
				# print(layer.gradB)
				lb.append(layer.gradB.reshape(layer.out_neurons))
				lw.append(layer.gradW)
		torch.save(lw,path_w)
		torch.save(lb,path_b)
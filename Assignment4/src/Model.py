import numpy as np
import torch
import time
import Criterion
import RNN


dtype = torch.double
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def pause():
	print("press enter")
	input()

class Model:
	def __init__(self):
		self.Layers = []
		self.isTrain = True
		self.unique_labels = None

	def forward(self, input,isTrain=False):
		# print("Forwarding: ")
		# print(input)
		for layer in self.Layers:
			input = layer.forward(input,isTrain)
		return input[-1]

	def backward(self, input, gradOutput):
		# print("Backpropogation: ")
		# print(len(gradOutput),len(gradOutput[0]))
		for i in range(len(self.Layers) - 1):
			# print('reached')
			inputPrev = self.Layers[-i-2].y
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

	def convert(self,batch,unique_labels):# convert to one hot encoding and same max length 
			max_num=float('-inf')
			newlist=[]
			for i in range(len(batch)):
				l=len(batch[i])
				if(l>max_num):
					max_num=l
			for i in range(max_num):
				x=torch.zeros(len(batch),len(unique_labels), dtype=dtype, device=device)
				for j in range(len(batch)):
					if len(batch[j])>i:
						# print(unique_labels.index(batch[j][i]))
						x[j,unique_labels.index(batch[j][i])]=1

				newlist.append(x)
			return newlist

	def createBatches(self,trainingData,trainingLabels, batchSize,unique_labels):
		trainingDataSize = len(trainingData) #list of tensors
		numBatches = trainingDataSize//batchSize + 1*(trainingDataSize%batchSize!=0)
		dlist=[[] for _ in range(numBatches)]
		llist=[[] for _ in range(numBatches)]

		

		for j in range(numBatches):
			batch=trainingData[batchSize*j:(j+1)*batchSize]
			labels=trainingLabels[batchSize*j:(j+1)*batchSize]

			batch=self.convert(batch,unique_labels) 
			dlist[j]=batch
			llist[j]=labels
		return dlist,llist

	def trainModel(self, learningRate, batchSize, epochs, trainingData,unique_labels, trainingLabels, alpha=0, regularizer=0,validationData=None,validationLabels=None):
		criterion = Criterion.Criterion()
		self.unique_labels = unique_labels
		batchList,batchLabels = self.createBatches(trainingData,trainingLabels, batchSize,unique_labels)
		DbatchList,DbatchLabels = self.createBatches(trainingData,trainingLabels, 1,unique_labels)
		trainingLabels=torch.tensor(trainingLabels)
		if type(validationData)!=type(None):
			validationData1,validationLabels1=self.createBatches(validationData,validationLabels,1,unique_labels)
			validationLabels=torch.tensor(validationLabels)
		for i in range(epochs):
			print("Epoch ", i)
			t = time.time()
			for j in range(len(batchList)):
				activations = self.forward(batchList[j],True)
				gradOutput=[torch.zeros(len(batchList[j][0]),activations.size()[1],dtype=dtype,device=device) for _ in range(len(batchList[j]))]        # times max_len of sequence of the batch
				gradOutput[-1] = criterion.backward(activations, batchLabels[j])
				self.clearGradParam()
				self.backward(batchList[j], gradOutput)
				self.updateParam(learningRate/((i+1)**0.7),alpha/((i+1)**0.7),regularizer)			
			t = time.time() - t
			print("time elapsed",t)
			if (i+1)%5==0 :
				predictions=[]
				crit_list=[]
				for j in range(len(DbatchList)):
					predictions.append(self.classify(DbatchList[j])[0])
					crit_list.append(criterion.forward(self.forward(DbatchList[j]), DbatchLabels[j]).item())

				predictions=torch.tensor(predictions)
				
				
				print("Training Loss",sum(crit_list)/len(crit_list))
				print("Training Accuracy: ", (torch.sum(predictions == trainingLabels).item()*100.0/trainingLabels.size()[0]))
				
			
				if type(validationData)!=type(None):
					predictions=[]
					crit_list=[]
					for j in range(len(validationData)):
						predictions.append(self.classify(validationData1[j])[0])
						crit_list.append(criterion.forward(self.forward(validationData1[j]), validationLabels1[j]).item())
					predictions=torch.tensor(predictions)
					print("validation Loss",sum(crit_list)/len(crit_list))
					print("Validation Accuracy: ", (torch.sum(predictions == validationLabels).item()*100.0/validationLabels.size()[0]))
						
					


	def classify(self, data):
		guesses = self.forward(data)
		value, indices = torch.max(guesses,dim=1)
		return indices

	def gettestPredictions(self,testData):
		l = [None]*len(testData)
		testData1,l=self.createBatches(validationData,l,1,self.unique_labels)
		predictions=[]
		for j in range(len(testData)):
			predictions.append(self.classify(testData1[j])[0])
		predictions=torch.tensor(predictions)
		return predictions

	def saveModel(self,pathconfig,filePath):
		lW = []
		f= open(pathconfig,"w+")
		f.write(str(len(self.Layers))+"\n")
		for layer in self.Layers:
			if layer.layerName == 'RNN' :
				f.write(layer.layerName+" "+str(layer.input_dim)+" "+str(layer.hidden_dim)+" "+str(layer.output_dim)+" "+str(layer.max)+"\n")
				lW.append(layer.weights_hh)
				lW.append(layer.weights_hx)
				lW.append(layer.weights_hy)
				lW.append(layer.bias_h)
				lW.append(layer.bias_y)
		lW.append(self.unique_labels)		
		torch.save(lW,filePath)

	def loadModel(self,path_config,filepath):
		with open(path_config) as f:
			content = f.readlines()
			content = [x.strip() for x in content]

		no_layers=int(content[0])
		lW = torch.load(filepath)
		# print(len(content))
		for i in range(0,len(content)-1):
			words=content[i+1].split()
			if words[0]=='RNN':
				layer = (RNN.RNN(int(words[1]),int(words[2]),int(words[3]),float(words[4])))
				layer.weights_hh = lW[i*5+0]
				layer.weights_hx = lW[i*5+1]
				layer.weights_hy = lW[i*5+2]
				layer.bias_h = lW[i*5+3]
				layer.bias_y = lW[i*5+4]
				self.addLayer(layer)
		self.unique_labels = lW[-1]

		
		
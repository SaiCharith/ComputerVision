import numpy as np
import torch
import Linear
import ReLU
import Criterion
import torchfile
import Dropout
import LeakyRelu

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
		
		batchList,batchLabels = self.createBatches(trainingData,trainingLabels, batchSize,unique_labels)
		DbatchList,DbatchLabels = self.createBatches(trainingData,trainingLabels, 1,unique_labels)
		trainingLabels=torch.tensor(trainingLabels)
		if type(validationData)!=type(None):
			validationData,validationLabels1=self.createBatches(validationData,validationLabels,1,unique_labels)
		for i in range(epochs):
			print("Epoch ", i)
			for j in range(len(batchList)):
				# print(batchList[j])
				# print(batchList)
				activations = self.forward(batchList[j],True)
				# print("forward done")
				gradOutput=[torch.zeros(len(batchList[j][0]),activations.size()[1],dtype=dtype,device=device) for _ in range(len(batchList[j]))]        # times max_len of sequence of the batch
				# gradOutput[:] =torch.zeros(len(batchList[j][0]),activations.size()[1]) # batchsize,output channels
				# print('max_len=',len(batchList[j]))
				# print('batchsize=',len(batchList[j][0]))
				# print('batchsize=',activations.size())


				gradOutput[-1] = criterion.backward(activations, batchLabels[j])
				# print(len(gradOutput),len(gradOutput[0]))
				self.clearGradParam()
				# print('backward start')

				self.backward(batchList[j], gradOutput)
				# print('backward done')
				self.updateParam(learningRate/((i+1)**0.7),alpha/((i+1)**0.7),regularizer)			
			
			predictions=[]
			crit_list=[]
			for j in range(len(DbatchList)):
				predictions.append(self.classify(DbatchList[j])[0])
				crit_list.append(criterion.forward(self.forward(DbatchList[j]), DbatchLabels[j]).item())

			predictions=torch.tensor(predictions)
			
			print("Training Loss",sum(crit_list))
			print("Training Accuracy: ", (torch.sum(predictions == trainingLabels).item()*100.0/trainingLabels.size()[0]))
			if i%5==0 and i>0:
				if type(validationData)!=type(None):
					predictions=[]
					crit_list=[]
					for j in range(len(validationData)):
						predictions.append(self.classify(validationData[j])[0])
						crit_list.append(criterion.forward(self.forward(validationData[j]), validationLabels[j]).item())
					predictions=torch.tensor(predictions)
					print("validation Loss",sum(crit_list))
					print("Validation Accuracy: ", (torch.sum(predictions == validationLabels).item()*100.0/validationLabels.size()[0]))
						
					


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
				f.write("Dropout "+str(layer.drop_rate)+"\n")
			if layer.layerName == 'LeakyRelu':
				f.write("LeakyRelu "+str(layer.leak)+"\n")
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
		print(layer_bias_path)

		try:
			bias = torch.load(layer_bias_path,map_location= {'cuda:0':'cpu'})
			weights = torch.load(layer_w_path,map_location= {'cuda:0':'cpu'})
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
					self.Layers[-1].W = torch.tensor(self.Layers[-1].W,dtype=dtype,device=device)
					self.Layers[-1].B = (bias[j]).reshape(self.Layers[-1].B.size())
					self.Layers[-1].B = torch.tensor(self.Layers[-1].B,dtype=dtype,device=device)
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
				self.Layers[-1].drop_rate = float(words[1])
			elif(words[0]=='LeakyRelu'):
				self.addLayer(LeakyRelu.LeakyRelu())
				self.Layers[-1].leak = float(words[1])
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
import numpy as np
import torch
dtype = torch.double
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Tanh:
    def __init__(self):
        self.output = None
        self.gradInput = None
        self.layerName = 'tanh'
        self.temp = None

    def forward(self, input, isTrain=False):
        self.output = input
        self.temp = torch.exp(2.0*input)
        self.output = (self.temp - 1.0)/(self.temp + 1.0)
        return self.output
        # print("Tanh Layer Forward")

    def backward(self, input, gradOutput):
        self.temp = torch.exp(2.0*input)
        self.gradInput = gradOutput
        self.gradInput *= (4.0*self.temp)/(self.temp + 1.0)**2.0
        # print("Tanh Layer backward")
        return self.gradInput

    def clearGradParam(self):
        return

    def dispGradParam(self):
        print("Tanh Layer")

    def updateParam(self, learningRate, alpha, regularizer=0):
        # print("Tanh Layer Update Weights & Biases: ")
        return

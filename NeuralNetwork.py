import random as rand
from helper import *


WEIGHT_INIT_MIN = -.1
WEIGHT_INIT_MAX = .1
RANDOM_SEED = 1
LEARNING_RATE = .1

class NeuralNetwork:
	def __init__(self, numInput, numOutput, w, d):
		self.numInputNodes = numInput
		self.numOutputNodes = numOutput
		self.width = w
		self.depth = d
		self.numLayers = d + 2
		self.layerSizes = [self.numInputNodes] + ([w] * d) + [self.numOutputNodes]
		self.initializeWeights()


	def initializeWeights(self):
		self.weights = []
		for layer in range(len(self.layerSizes) - 1):
			self.weights.append([[rand.uniform(WEIGHT_INIT_MIN, WEIGHT_INIT_MAX) for y in range(self.layerSizes[layer+1])] for x in range(self.layerSizes[layer])])
		

	def classify(self, ex):
		return applySigmoid(self.runOneExample(ex)[-1])

	def runOneExample(self, ex):
		currentLayer = ex
		if len(ex) >= self.numInputNodes:
			currentLayer = currentLayer[:-1]
		
		activations = []
		activations.append(ex)
	
		for x in range(self.numLayers - 1):
			try:
				nextLayer = self.getNextLayer(currentLayer, self.layerSizes[x+1], self.weights[x])
			except:
				print ex
				exit(0)
			activations.append(nextLayer)
			currentLayer = applySigmoid(nextLayer)
		return activations

	def getNextLayer(self, currentLayer, sizeNext, w):
		nextLayer = [0] * sizeNext
		for x in range(sizeNext):
			activation = 0 
			for y in range(len(currentLayer)):
				activation += (w[y][x] * currentLayer[y])

			nextLayer[x] = activation

		return nextLayer

	def getError(self, predicted, groundTruth):
		error = 0
		for i in len(predicted):
			error += 0.5 * math.pow(predicted[i] - groundTruth[i], 2)
		return error

	def isCorrect(self, predicted, groundTruth):
		classification = applySigmoid(predicted)
		return classification.index(max(classification)) == groundTruth.index(max(groundTruth))

	def learnOnExample(self, ex, groundTruth):
		activations = self.runOneExample(ex)
		delta = [[0 for x in range(self.layerSizes[y])]
		               for y in range(len(self.layerSizes))]

		for x in range(len(groundTruth)):
			delta[-1][x] = ((-1.0 * (groundTruth[x] - sigmoid(activations[-1][x]))) * sigmoidPrime(activations[-1][x]))

		for layer in range(len(self.layerSizes) - 2, 0, -1):
			for node in range(len(delta[layer])):
				sum = 0
				for x in range(len(delta[layer+1])):
					sum += (self.weights[layer][node][x] * delta[layer + 1][x])
					delta[layer][node] = sigmoidPrime(activations[layer][node]) * sum

		for layer in range(len(self.weights)):
			for x in range(len(self.weights[layer])):
				for y in range(len(self.weights[layer][x])):
					currentWeight = self.weights[layer][x][y]
					newWeight = currentWeight - (LEARNING_RATE * delta[layer+1][y] * sigmoid(activations[layer][x]))
					self.weights[layer][x][y] = newWeight
		return self.isCorrect(activations[-1], groundTruth)


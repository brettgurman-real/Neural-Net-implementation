import math as math

def sigmoid(x):
	return 1/float(1 + math.pow(math.e, -1 * max(-50, x)))

def sigmoidPrime(x):
	return float(sigmoid(x) * (1.0 - sigmoid(x)))


def applySigmoid(arr):
	return [sigmoid(i) for i in arr]



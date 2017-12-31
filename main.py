from NeuralNetwork import *
import sys

ITER_EXP1 = 5000
ITER_EXP2 = 200
ITER_EXP3 = 200         
LOG = False        

def toVector(index, numValues):
	vector = [0 for x in range(numValues)]
	vector[int(index)] = 1
	return vector

def experiment1():
	results = runExperiment(3, 1, "838.arff", "838.arff", ITER_EXP1)

	with open('experiment1.results', 'w') as f:
		f.write("Training accuracies for Experiment 1:\n")
		for line in results[0]:
			f.write(str(line[0]) + ", " + str(line[1]) + '\n')
		f.write('\n\n')

		f.write("Testing accuracies for Experiment 1:\n")
		for line in results[1]:
			f.write(str(line[0]) + ", " + str(line[1]) + '\n')
		f.write('\n\n')

		f.write("Hidden Layer Activations for Experiment 1:\n")
		for line in results[2]:	
			f.write(str(line[0]) + ", " + str(line[1]) + '\n')

def experiment2():
	depth = 3
	width = [5, 10, 15, 20, 30, 40]

	with open('experiment2.results', 'w') as f:
		f.write("Experiment two results. Testing constant depth (d=3) and variable width (w=[5,10,15,20,30,40])\n\n")
		for w in width:
			results = runExperiment(w,depth, "optdigits_train.arff", "optdigits_test.arff", ITER_EXP2)
			f.write("Training accuracies for w=" + str(w) + ":\n")
			for line in results[0]:
				f.write(str(line[0]) + ", " + str(line[1]) + '\n')
			f.write('\n')

			f.write("Testing accuracies for w=" + str(w) + ":\n")
			for line in results[1]:
				f.write(str(line[0]) + ", " + str(line[1]) + '\n')
			f.write('\n\n')

def experiment3():
	depth = [0,1,2,3,4,5]	
	width = 10

	with open('experiment3.results', 'w') as f:
		f.write("Experiment three results. Testing constant width (w=10) and variable width (d=[0,1,2,3,4,5])\n\n")
		for d in depth:
			results = runExperiment(width, d, "optdigits_train.arff", "optdigits_test.arff", ITER_EXP3)
			f.write("Training accuracies for w=" + str(d) + ":\n")
			for line in results[0]:
				f.write(str(line[0]) + ", " + str(line[1]) + '\n')
			f.write('\n')

			f.write("Testing accuracies for w=" + str(d) + ":\n")
			for line in results[1]:
				f.write(str(line[0]) + ", " + str(line[1]) + '\n')
			f.write('\n\n')

def experimentX(name,w,d,train,test, iterations):
	with open('experiment' + name + '.results', 'w') as f:
		f.write("Experiment " + name + " results. Training file = " + train + ", Testing file = " + test + ", w = " + str(w) + ", d = " + str(d) + "\n\n")
		results = runExperiment(w, d, train, test, iterations)
		f.write("Training accuracies for Experiment " + name + ":\n")
		for line in results[0]:
			f.write(str(line[0]) + ", " + str(line[1]) + '\n')
		f.write('\n')

		f.write("Testing accuracies for Experiment " + name + ":\n")
		for line in results[1]:
			f.write(str(line[0]) + ", " + str(line[1]) + '\n')
		f.write('\n\n')

		f.write("Hidden Layer Activations for Experiment 1:\n")
		for line in results[2]:	
			f.write(str(line[0]) + ", " + str(line[1]) + '\n')

def runExperiment(w,d,train,test, epochs):
	trainAccuracy = []
	trainSet = getAttributesAndData(train)

	testAccuracy = []
	testSet = getAttributesAndData(test)

	labels = getLabels(trainSet)
	numInputNodes = len(trainSet[1][0]) - 1
	numOutputNodes = len(labels)

	n = NeuralNetwork(numInputNodes, numOutputNodes,w,d)

	print "Starting..."
	for i in range(epochs):
		errors = []
		for example in trainSet[1]:
			errors.append(n.learnOnExample(example[:-1], toVector(labels.index(int(example[-1])), numOutputNodes)))
		if i%(epochs/5) == (epochs/5)-1:
			print "...still training..."
		trainAccuracy.append((i, sum(errors)/float(len(trainSet[1]))))

		errors = []
		for example in testSet[1]:
			if train == test:
				classification = n.classify(example[:-1])
			else:
				classification = n.classify(example)

			errors.append(classification.index(max(classification)) == labels.index(int(example[-1])))

		testAccuracy.append((i, sum(errors)/float(len(testSet[1]))))

	return (trainAccuracy, testAccuracy, getActivations(n))


def getAttributesAndData(filename):
	with open(filename, 'r') as file:
		attributes = []
		data = []

		atData = False

		for line in file:
			if atData:
				data.append([float(x) for x in line.split(",")])
			else:
				if "@attribute" in line.lower():
					attributes.append((line.split(" ")[1], line.split(" ")[2]))
				elif "@data" in line.lower():
					atData = True


		return (attributes,data)

def getLabels(s):
	return [int(label) for label in s[0][-1][-1][1:-2].split(',')]


def getActivations(net):
	result = []
	for i in range(net.numOutputNodes):
		result.append((i, str(net.runOneExample(toVector(i,net.numOutputNodes))[1])))
	return result


def main():
	if(len(sys.argv) == 1):
		experiment1()
		print "done experiment1"
		experiment2()
		print "done experiment2"
		experiment3()
		print "done experiment3"
	else:
		try:
			experimentX(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5], int(sys.argv[6]))
		except:
			print "Incorrect usage. Correct usage is: python main.py <test name> <width> <depth> <train file> <test file> <num iterations>"
			exit(0)
if __name__ == '__main__':
	main()


from Perceptron import *
from csv import reader
from random import shuffle

def irisLabeler(sampleClass):
    if sampleClass == 'Iris-setosa':
        return 1
    if sampleClass == 'Iris-versicolor':
        return -1
    if sampleClass == 'Iris-virginica':
        return -1

dataset = []
with open('iris.csv', newline='') as irisFile:
    irisReader = reader(irisFile)
    for row in irisReader:
        inputs = list(map(float, row[:4]))
        label = irisLabeler(row[4])
        
        dataset.append(LabeledInputs(label, inputs))

shuffle(dataset)
trainingSet = dataset[:100]
testSet = dataset[100:150]

p = Perceptron(0.5, [0,0,0,0])
print('Training...')
train(p, trainingSet)
print('Done!')

hits = 0
for li in testSet:
    if attemptToFire(p, li.inputs) == li.label: hits += 1

print('Accuracy: ', hits / 50)


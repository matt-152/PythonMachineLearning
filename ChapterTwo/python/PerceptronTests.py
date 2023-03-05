from Perceptron import *
from csv import reader

def test_attemptToFire():
    p = Perceptron(0.5, [0])
    assert attemptToFire(p, [1]) == -1
    p = Perceptron(0.5, [1])
    assert attemptToFire(p, [1]) == 1
    assert attemptToFire(p, [0]) == -1
    assert attemptToFire(p, [0.6]) == 1
    assert attemptToFire(p, [0.4]) == -1
    p = Perceptron(0.7, [1])
    assert attemptToFire(p, [0.6]) == -1

    p = Perceptron(0.5, [1,1])
    assert attemptToFire(p, [1,0]) == 1
    assert attemptToFire(p, [0,1]) == 1
    assert attemptToFire(p, [0.26, 0.25]) == 1
    assert attemptToFire(p, [0.24, 0.25]) == -1

def test_adjustWeightsAdjustsWeights():
    p = Perceptron(0, [0])
    li = LabeledInputs(-1, [0])
    oldWeights = p.weights
    adjustWeights(p, li)
    assert p.weights == oldWeights
    li2 = LabeledInputs(1, [1])
    adjustWeights(p, li2)
    assert p.weights[0] > oldWeights[0]
    oldWeights2 = p.weights
    li3 = LabeledInputs(-1, [1])
    adjustWeights(p, li3)
    assert p.weights[0] < oldWeights2[0]

def test_adjustWeightsConverges():
    p = Perceptron(0.5, [0.1])
    li = LabeledInputs(1, [6])
    oldWeights = p.weights
    adjustWeights(p, li)
    assert p.weights == oldWeights
    li2 = LabeledInputs(1, [5])
    adjustWeights(p, li2)
    assert sum(p.weights) > sum(oldWeights)
    oldWeights2 = p.weights
    li3 = LabeledInputs(-1, [4.9])
    adjustWeights(p,li3)
    assert sum(p.weights) <= sum(oldWeights2)
    assert sum(p.weights) > sum(oldWeights)
    oldWeights3 = p.weights
    li4 = LabeledInputs(1, [4.99])
    adjustWeights(p, li4)
    assert sum(p.weights) >= sum(oldWeights3)
    assert sum(p.weights) <= sum(oldWeights2)

def test_train():
    p = Perceptron(0.5, [0])
    li = LabeledInputs(1, [1])
    train(p, [li])
    assert attemptToFire(p, li.inputs) == li.label
    li2 = LabeledInputs(-1, [0.5])
    train(p, [li, li2])
    assert attemptToFire(p, li.inputs) == li.label
    assert attemptToFire(p, li2.inputs) == li2.label
    li3 = LabeledInputs(1, [0.6])
    train(p, [li, li2, li3])
    assert attemptToFire(p, li.inputs) == li.label
    assert attemptToFire(p, li2.inputs) == li2.label
    assert attemptToFire(p, li3.inputs) == li3.label

    p = Perceptron(1, [0])
    li = LabeledInputs(1, [1])
    train(p, [li])
    assert attemptToFire(p, li.inputs) == li.label
    li2 = LabeledInputs(-1, [0.5])
    train(p, [li, li2])
    assert attemptToFire(p, li.inputs) == li.label
    assert attemptToFire(p, li2.inputs) == li2.label

if __name__ == '__main__':
    print('Running...')
    test_attemptToFire()
    test_adjustWeightsAdjustsWeights()
    test_adjustWeightsConverges()
    test_train()
    print('Tests complete')


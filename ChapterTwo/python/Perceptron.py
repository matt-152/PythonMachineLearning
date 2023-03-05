class Perceptron:
    def __init__(self, threshold, weights):
        self.weights = [-1 * threshold] + weights

class LabeledInputs:
    def __init__(self, label, inputs):
        self.label = label
        self.inputs = inputs

def attemptToFire(perceptron, inputs):
    inputs_ = _addThreshPseudoInput(inputs)
    weightsTimesInputs = map(_multiply, perceptron.weights, inputs_)
    dotProductWI = sum(weightsTimesInputs)

    if dotProductWI > 0: return 1
    else: return -1

def _multiply(x, y): return x * y

def adjustWeights(perceptron, labeledInputs):
    perceptronOut = attemptToFire(perceptron, labeledInputs.inputs)
    iCoeff = labeledInputs.label - perceptronOut
    learningRule = lambda w, i: w + iCoeff * i

    inputs_ = _addThreshPseudoInput(labeledInputs.inputs)

    adjustedWeights = map(learningRule, perceptron.weights, inputs_)
    perceptron.weights = list(adjustedWeights)

    weightsWereAdjusted = iCoeff == 2 or iCoeff == -2
    return weightsWereAdjusted

def _addThreshPseudoInput(inputs):
    yield 1
    for param in inputs:
        yield param

def train(perceptron, labeledDataset, targetAccuracy=1):
    passingResults = 0
    dsIter = _infDatasetIter(labeledDataset)

    while passingResults < len(labeledDataset) * targetAccuracy:
        labeledInputs = next(dsIter)
        returnCode = adjustWeights(perceptron, labeledInputs)
        weightsWereAdjusted = returnCode

        if weightsWereAdjusted: passingResults = 0
        else: passingResults += 1

def _infDatasetIter(dataset):
    for line in dataset:
        yield line
    for line in _infDatasetIter(dataset):
        yield line


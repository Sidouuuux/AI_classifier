from random import *

ITERATIONS = 10
LEARNING_RATE = 1e-9

class Neuron:
    def __init__(self, weights, learningRate):
        self.weights = []
        self.learningRate = learningRate
        for i in range(weights):
            self.weights.append(random()* 0.2 - 0.1)

    def run(self, inputs):
        output = 0
        for i in range(len(self.weights)):
            output += inputs[i] * self.weights[i]
        return output

    def learn(self, inputs, desired):
        output = self.run(inputs)
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i] + self.learningRate \
                         * (desired - output) * inputs[i]
        return abs(desired - output)

    def __repr__(self):
        return str(self.weights)

if __name__ == '__main__':
    dataset = []    
    f = open('toto.txt')
    for line in f.readlines():
        line = line.strip()
        if line:
            dataset.append(list(map(float, line.replace(',', '.').split())))
    f.close()

    n = Neuron(len(dataset[0]) - 1, LEARNING_RATE)
    print(n)

    f = open('error.txt', 'w')
    for iteration in range(ITERATIONS):
        for example in dataset:
            error = n.learn(example[:-1], example[-1])
            f.write('%f\n' % (error * error))
            #print(n)

    f.close()
        
    print(n)

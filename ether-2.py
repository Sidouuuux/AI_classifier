import numpy
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from datetime import *
from math import *

DAYS = [10, 20, 30]

def prepareData(filename):
    f = open(filename)
    f.readline() #skip first line

    prices = []
    for line in f.readlines():
        fields = line.split(',')
        close = float(fields[1])
        prices.append(close)

    data = []
    for i in range(max(DAYS), len(prices)):
        day = []
        for j in DAYS:
            day.append(prices[i - j])
        data.append(day)

    inputs = numpy.array(data)
    desired = numpy.array(prices[max(DAYS):])
    return inputs, desired, prices

if __name__ == '__main__':
    inputs, desired, prices = prepareData('ether.csv')

    #print(inputs)
    #print(desired)

    print(inputs.shape)
    print(desired.shape)

    reg = LinearRegression()
    reg.fit(inputs, desired)

    print("Coefficients de régression : ", reg.coef_)
    print(reg.score(inputs, desired))


    print([prices[-30], prices[-20], prices[-10]])
    test = [prices[-30], prices[-20], prices[-10]]
    prediction = reg.predict([test])
    print(prediction)

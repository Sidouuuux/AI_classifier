import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from datetime import *

if __name__ == '__main__':
    f = open('ether.csv')
    f.readline() #skip first line

    data = []
    for line in f.readlines():
        fields = line.split(',')
        #dt = datetime.fromtimestamp(int(fields[0]) / 1000)
        dt = int(fields[0]) / 1000
        close = float(fields[1])
        data.append([[dt], [close]]) #add timestamp and close into a new data row

    data = np.array(data)
    inputs = data[:,0] #first column
    outputs = data[:,1] #second column

    print(len(inputs), len(outputs))

    
    reg = LinearRegression()
    reg.fit(inputs, outputs)

    print(reg.coef_)

    future = datetime.fromisoformat('2022-05-11T10:00:20').timestamp()
    print(future)

    print(reg.predict([[future]]))
    
    f.close()

    

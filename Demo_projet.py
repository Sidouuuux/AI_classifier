import numpy as np
import matplotlib.pyplot as plt
import skimage.io
import os
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm
import time
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from math import *
from sklearn.model_selection import GridSearchCV
import pickle
import sys

IMAGE_REDUCED_SIZE = 72
FOLDER = './images/'
FILENAME = '500model.dat'
# max 7390
DATA_SIZE = 1000

# For example purpose only
# example = example[:, :, 0] #keep only R component (from RGB)
# showImage(example)


def showImage(image):
    print(image.shape)
    fig = plt.figure()
    plt.imshow(image)
    plt.show()


def getData(_image_folder=FOLDER, _reduce_to=IMAGE_REDUCED_SIZE, _data=DATA_SIZE):
    filenames = []
    classes = []
    data = []
    numpy_file = f'data_{str(_image_folder[2:-1])}_{str(_data)}_{str(_reduce_to)}.npy'
    print(numpy_file)
    for filename in os.listdir(_image_folder):
        if filename.endswith('.jpg'):
            filenames.append(_image_folder + filename)

    if os.path.isfile(numpy_file):
        for i in tqdm(range(_data)):
            filename = filenames[i]
            classes.append(filename.replace(_image_folder, '').split('_')[0])
        data = np.load(numpy_file)
        # print(len(classes))
        # print(set(classes))
        print(len(set(classes)))
        return data, classes

    else:
        for i in tqdm(range(_data)):
            filename = filenames[i]
            classes.append(filename.replace(_image_folder, '').split('_')[0])

            example = skimage.io.imread(filename)
            example = skimage.transform.resize(
                example, (_reduce_to, _reduce_to, 3))
            # showImage(example)
            line = example[:, :, 0].reshape(1, -1)
            data.append(line[0])

    data = np.array(data)
    np.save(numpy_file, data)
    # data = load('data.npy')
    return data, classes


if __name__ == "__main__":
    t0 = time.time()

    dataInputs, classes = getData()
    print(len(dataInputs))
    print(len(classes))
    # print(len(classes))
    # print(len(dataInputs))
    train_inputs, test_inputs, \
        train_desired, test_desired = train_test_split(
            dataInputs, classes, test_size=0.20)
    # print(dataInputs)
    # mlp = MLPClassifier(learning_rate_init=.000001,
    #                     max_iter=3000, hidden_layer_sizes=(15,))  # A vous !
    # --------------------------------------------------------------------------------------
    # mlp.fit(train_inputs, train_desired)  # Apprentissage
    parameter_space = {
        'hidden_layer_sizes': [
            (1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,), (9,), (10,),
            (11,), (12,), (13,), (14,), (15,), (16,), (17,), (18,),
            (19,), (20,), (21,)
        ],
        'max_iter': [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000],
        'learning_rate_init': [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001],
    }
    clf = GridSearchCV(MLPClassifier(), parameter_space, cv=3,
                       scoring='accuracy')
    clf.fit(train_inputs, train_desired)  # Apprentissage
    print("Best parameters set found on development set:")
    print(clf.best_params_)
    train_score = clf.score(train_inputs, train_desired)
    test_score = clf.score(test_inputs, test_desired)
    # Score d'apprentissage de régression
    print(f'Training score : {train_score * 100:.2f}%')
    print(f'Test score     : {test_score * 100:.2f}%')

    with open(FILENAME, 'wb') as file:
        pickle.dump(clf, file)
    t1 = time.time()

    # print(clf.loss_)
    # print(test_inputs[0])
    print(clf.predict([test_inputs[0]]))
    print(clf.predict_proba([test_inputs[0]]))
    print("time : ")
    print(t1-t0)
    print("learning_rate_init=.007, activation=identity, alpha=0.05, hidden_layer_sizes=(15,), learning_rate=constant")
    fig, ax = plt.subplots()
    ax.plot(clf.loss_curve_)
    plt.yscale('log')
    plt.show()

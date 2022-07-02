import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from math import *


def loadData(filename):
    f = open(filename)
    # f.readline()
    return np.loadtxt(f, delimiter=',', skiprows=1)


def showSample(sample):
    fig, ax = plt.subplots()
    ax.matshow(sample.reshape(28, 28))
    plt.show()


def showWeights(mlp):
    fig, axes = plt.subplots(3, 3)

    iLayer = 0
    print(mlp.coefs_[iLayer])
    print(mlp.coefs_[iLayer].T)
    # use global min / max to ensure all weights are shown on the same scale
    vmin, vmax = mlp.coefs_[iLayer].min(), mlp.coefs_[iLayer].max()
    for coef, ax in zip(mlp.coefs_[iLayer].T, axes.ravel()):
        square = int(sqrt(len(coef)))
        ax.matshow(coef.reshape(square, square), cmap=plt.cm.gray, vmin=.5 * vmin,
                   vmax=.5 * vmax)
        ax.set_xticks(())
        ax.set_yticks(())

    plt.show()


if __name__ == "__main__":
    filename = "mnist_784_light.csv"
    data = loadData(filename)

    inputs = data[:, :-1]  # Toutes les lignes, sauf la dernière colonne
    desired = data[:, -1]  # Toutes les lignes, juste la dernière colonne

    # print(inputs.shape)
    # print(desired.shape)

    # Séparation jeu de test / jeu d'apprentissage
    train_inputs, test_inputs, \
        train_desired, test_desired = train_test_split(
            inputs, desired, test_size=0.20)

    mlp = MLPClassifier(learning_rate_init=1e-4)
    mlp.fit(train_inputs, train_desired)  # Apprentissage

    # print(mlp.loss_)  # Erreur globale du réseau en apprentissage

    train_score = mlp.score(train_inputs, train_desired)
    test_score = mlp.score(test_inputs, test_desired)
    # Score d'apprentissage de régression
    print(f'Training score : {train_score * 100:.2f}%')
    print(f'Test score     : {test_score * 100:.2f}%')  # Score de test

    fig, ax = plt.subplots()
    ax.plot(mlp.loss_curve_)
    plt.yscale('log')
    plt.show()

    print(mlp.predict([test_inputs[0]]))
    print(mlp.predict_proba([test_inputs[0]]))
    showSample(test_inputs[0])  # Affichage d'un échantillon

    #print(confusion_matrix(classTrue, classPred))

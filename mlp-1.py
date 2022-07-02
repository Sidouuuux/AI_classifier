import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPClassifier


def loadData(filename):
    f = open(filename)
    f.readline()
    return np.loadtxt(f, delimiter=',')


def showSample(sample):
    fig, ax = plt.subplots()
    ax.matshow(sample.reshape(28, 28))
    plt.show()


if __name__ == "__main__":
    filename = "mnist_784_light.csv"
    data = loadData(filename)
    print(data)

    inputs = data[:, :-1]  # Toutes les lignes, sauf la dernière colonne
    desired = data[:, -1]  # Toutes les lignes, juste la dernière colonne

    # Séparation jeu de test / jeu d'apprentissage
    datasize = len(data)
    splitPoint = int(datasize * 0.25)  # 25% pour l'apprentissage
    train_inputs = inputs[:splitPoint]
    train_desired = desired[:splitPoint]
    test_inputs = inputs[splitPoint:]
    test_desired = desired[splitPoint:]

    # showSample(inputs[15]) #Affichage d'un échantillon

    mlp = MLPClassifier(learning_rate_init=0.0025,
                        hidden_layer_sizes=(7,))
    mlp.fit(train_inputs, train_desired)  # Apprentissage

    print(mlp.loss_)  # Erreur globale du réseau en apprentissage

    train_score = mlp.score(train_inputs, train_desired)
    test_score = mlp.score(test_inputs, test_desired)
    # Score d'apprentissage de classification (en %)
    print(f'Training score : {train_score * 100:.2f}%')
    print(f'Test score     : {test_score * 100:.2f}%')  # Score de test

    fig, ax = plt.subplots()
    ax.plot(mlp.loss_curve_)
    plt.yscale('log')
    plt.show()

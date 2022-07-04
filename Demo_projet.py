import numpy as np
import matplotlib.pyplot as plt
import skimage.io
import os
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from math import *
from sklearn.model_selection import GridSearchCV
import pickle
import sys

IMAGE_REDUCED_SIZE = 28
FOLDER = './images/'
FILENAME = '2model.dat'

# For example purpose only
# example = example[:, :, 0] #keep only R component (from RGB)
# showImage(example)


def showImage(image):
    print(image.shape)
    fig = plt.figure()
    plt.imshow(image)
    plt.show()


if __name__ == "__main__":
    filenames = []
    classes = []
    for filename in os.listdir(FOLDER):
        if filename.endswith('.jpg'):
            filenames.append(FOLDER + filename)
            # classes.append(filename.split('_')[0])
    # print(list(set(classes)))
    # print(new_list = list(set(my_list))

    dataInputs = []
    print(len(filenames))
    for i in range(len(filenames)):

        print(f'{str(i)} : {filename}')

        filename = filenames[i]

        # print(filename.replace(FOLDER, '').split('_')[0])
        classes.append(filename.replace(FOLDER, '').split('_')[0])

        example = skimage.io.imread(filename)
        example = skimage.transform.resize(
            example, (IMAGE_REDUCED_SIZE, IMAGE_REDUCED_SIZE,3))

        # showImage(example)
        print(type(example))
        print(example.shape)
        line = example[:, :, 0].reshape(1, -1)
        dataInputs.append(line[0])
        example = []
        # print(line.shape)
        # print(example[:, :, 0])

    #example = line.reshape(IMAGE_REDUCED_SIZE, IMAGE_REDUCED_SIZE, -1)
    # showImage(example)
    print("LET'S BEGIIIIN")
    dataInputs = np.array(dataInputs)
    # print(len(classes))
    # print(len(dataInputs))
    train_inputs, test_inputs, \
        train_desired, test_desired = train_test_split(
            dataInputs, classes, test_size=0.20)
    # print(dataInputs)
    mlp = MLPClassifier(max_iter=500, learning_rate_init=.004)  # A vous !
    # --------------------------------------------------------------------------------------
    mlp.fit(train_inputs, train_desired)  # Apprentissage
    # parameter_space = {
    #     'hidden_layer_sizes': [
    #         (1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,), (9,), (10,), (11,
    #                                                                       ), (12,), (13,), (14,), (15,), (16,), (17,), (18,), (19,), (20,), (21,)
    #     ],
    #     'activation': ['tanh', 'relu', 'identity', 'logistic'],
    #     # 'solver': ['sgd', 'adam', 'lbfgs', 'sgd'],
    #     'alpha': [0.0001, 0.05],
    #     'learning_rate': ['constant', 'adaptive'],
    # }
    # clf = GridSearchCV(MLPClassifier(), parameter_space, cv=3,
    #                    scoring='accuracy')
    # clf.fit(train_inputs, train_desired)  # Apprentissage
    # print("Best parameters set found on development set:")
    # print(clf.best_params_)
    train_score = mlp.score(train_inputs, train_desired)
    test_score = mlp.score(test_inputs, test_desired)
    # Score d'apprentissage de régression
    print(f'Training score : {train_score * 100:.2f}%')
    print(f'Test score     : {test_score * 100:.2f}%')  # Score de test
    with open(FILENAME, 'wb') as file:
        pickle.dump(mlp, file)
    # print(mlp.loss_)
    # print(test_inputs[0])
    print(mlp.predict([test_inputs[0]]))
    print(mlp.predict_proba([test_inputs[0]]))
    fig, ax = plt.subplots()
    ax.plot(mlp.loss_curve_)
    plt.yscale('log')
    plt.show()

    # print(mlp.predict([test_inputs[0]]))
    # print(mlp.predict_proba([test_inputs[0]]))
    # showSample(test_inputs[0])  # Affichage d'un échantillon

    #print(confusion_matrix(classTrue, classPred))
    # --------------------------------------------------------------------------------------

    # --- DEMO AUTOENCODER ---
    # autoEncoder = MLPRegressor(max_iter=2000, hidden_layer_sizes=(100, ))
    # autoEncoder.fit(dataInputs, dataInputs)

    # fig, ax = plt.subplots()
    # ax.plot(autoEncoder.loss_curve_)
    # plt.yscale('log')
    # plt.show()

    # example = skimage.io.imread('./images/Abyssinian_1.jpg')
    # example = skimage.transform.resize(
    #     example, (IMAGE_REDUCED_SIZE, IMAGE_REDUCED_SIZE))
    # example = example.reshape(1, -1)
    # result = autoEncoder.predict(example)
    # result = result.reshape(IMAGE_REDUCED_SIZE, IMAGE_REDUCED_SIZE, -1)
    # showImage(result)

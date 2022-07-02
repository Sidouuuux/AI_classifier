import numpy as np
import matplotlib.pyplot as plt
import skimage.io
import os
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier

IMAGE_REDUCED_SIZE = 32
FOLDER = './images/'

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
            classes.append(filename.split('_')[0])
    # print(list(set(classes)))
    # print(new_list = list(set(my_list))

    dataInputs = []
    for i in range(10):
        filename = filenames[i]
        example = skimage.io.imread(filename)
        example = skimage.transform.resize(
            example, (IMAGE_REDUCED_SIZE, IMAGE_REDUCED_SIZE))

        # showImage(example)

        line = example[:, :, 0].reshape(1, -1)
        dataInputs.append(line[0])
        # print(line.shape)
        print(example[:, :, 0])
        STOP

    #example = line.reshape(IMAGE_REDUCED_SIZE, IMAGE_REDUCED_SIZE, -1)
    # showImage(example)

    dataInputs = np.array(dataInputs)
    # print(dataInputs)
    mlp = MLPClassifier()  # A vous !
    # --------------------------------------------------------------------------------------

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
    # --------------------------------------------------------------------------------------

    # --- DEMO AUTOENCODER ---
    autoEncoder = MLPRegressor(max_iter=2000, hidden_layer_sizes=(100, ))
    autoEncoder.fit(dataInputs, dataInputs)

    fig, ax = plt.subplots()
    ax.plot(autoEncoder.loss_curve_)
    plt.yscale('log')
    plt.show()

    example = skimage.io.imread('./images/Abyssinian_1.jpg')
    example = skimage.transform.resize(
        example, (IMAGE_REDUCED_SIZE, IMAGE_REDUCED_SIZE))
    example = example.reshape(1, -1)
    result = autoEncoder.predict(example)
    result = result.reshape(IMAGE_REDUCED_SIZE, IMAGE_REDUCED_SIZE, -1)
    showImage(result)

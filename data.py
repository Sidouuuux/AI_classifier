from PIL import Image
import numpy as np
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage.io
import os
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
import tqdm
from pathlib import Path

IMAGE_REDUCED_SIZE = 32
FOLDER = './images/'

filenames = []
classes = []
data = []
if Path('animals.npy').is_file() and Path('labels.npy').is_file():
    animals = np.load("animals.npy")
    labels = np.load("labels.npy")
else:
    i = 0

    for filename in os.listdir(FOLDER):
        if i > 30:
            break
        if filename.endswith('.jpg'):
            filenames.append(FOLDER + filename)
            example = skimage.io.imread(FOLDER + filename)
            example = skimage.transform.resize(
                example, (IMAGE_REDUCED_SIZE, IMAGE_REDUCED_SIZE))
            line = example[:, :, 0].reshape(1, -1)

            # imag = cv2.imread(FOLDER + filename)
            # img_from_ar = Image.fromarray(imag, 'RGB')
            # resized_image = img_from_ar.resize((50, 50))
            data.append(line[0])
            classes.append(filename.split('_')[0])
            print(i)
            i += 1

    animals = np.array(data)
    labels = np.array(classes)

    np.save("animals", animals)
    np.save("labels", labels)

s = np.arange(animals.shape[0])
np.random.shuffle(s)
animals = animals[s]
labels = labels[s]
num_classes = len(np.unique(labels))
data_length = len(animals)
(x_train, x_test) = animals[(int)(0.8*data_length)
 :], animals[:(int)(0.2*data_length)]
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255
train_length = len(x_train)
test_length = len(x_test)
(y_train, y_test) = labels[(int)(0.8*data_length)
 :], labels[:(int)(0.2*data_length)]

mlp = MLPClassifier()  # A vous !
# --------------------------------------------------------------------------------------

train_score = mlp.score(x_train, train_desired)
test_score = mlp.score(test_inputs, test_desired)

# Score d'apprentissage de régression
print(f'Training score : {train_score * 100:.2f}%')
print(f'Test score     : {test_score * 100:.2f}%')

# import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Flatten, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dropout, BatchNormalization
from tensorflow.keras.optimizers import SGD
from keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import itertools


def load_cifar_data():
    """
    Loads the CIFAR-10 dataset using Keras and preprocess for training.
    """
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    return x_train, y_train, x_test, y_test, labels


x_train, y_train, x_test, y_test, labels = load_cifar_data()

print("x_train: ")
print(x_train.shape)

print("y_train: ")
print(y_train.shape)

print("x_test: ")
print(x_test.shape)

print("y_test: ")
print(y_test.shape)

x_train, y_train, x_test, y_test, labels = load_cifar_data()

# Visualize the dataset

M = 8
N = 10
figure, axes = plt.subplots(M, N, figsize=(20, 20))
index = [0] * 10

for i in range(0, 200):
    if (labels[int(y_train[i])] == 'airplane') and (index[0] < M):
        axes[index[0], 0].imshow(x_train[i, 1:])
        axes[0, 0].set_title(labels[int(y_train[i])])
        axes[index[0], 0].axis('off')
        index[0] = index[0] + 1
    if (labels[int(y_train[i])] == 'automobile') and (index[1] < M):
        axes[index[1], 1].imshow(x_train[i, 1:])
        axes[0, 1].set_title(labels[int(y_train[i])])
        axes[index[1], 1].axis('off')
        index[1] = index[1] + 1
    if (labels[int(y_train[i])] == 'bird') and (index[2] < M):
        axes[index[2], 2].imshow(x_train[i, 1:])
        axes[0, 2].set_title(labels[int(y_train[i])])
        axes[index[2], 2].axis('off')
        index[2] = index[2] + 1
    if (labels[int(y_train[i])] == 'cat') and (index[3] < M):
        axes[index[3], 3].imshow(x_train[i, 1:])
        axes[0, 3].set_title(labels[int(y_train[i])])
        axes[index[3], 3].axis('off')
        index[3] = index[3] + 1
    if (labels[int(y_train[i])] == 'deer') and (index[4] < M):
        axes[index[4], 4].imshow(x_train[i, 1:])
        axes[0, 4].set_title(labels[int(y_train[i])])
        axes[index[4], 4].axis('off')
        index[4] = index[4] + 1
    if (labels[int(y_train[i])] == 'dog') and (index[5] < M):
        axes[index[5], 5].imshow(x_train[i, 1:])
        axes[0, 5].set_title(labels[int(y_train[i])])
        axes[index[5], 5].axis('off')
        index[5] = index[5] + 1
    if (labels[int(y_train[i])] == 'frog') and (index[6] < M):
        axes[index[6], 6].imshow(x_train[i, 1:])
        axes[0, 6].set_title(labels[int(y_train[i])])
        axes[index[6], 6].axis('off')
        index[6] = index[6] + 1
    if (labels[int(y_train[i])] == 'horse') and (index[7] < M):
        axes[index[7], 7].imshow(x_train[i, 1:])
        axes[0, 7].set_title(labels[int(y_train[i])])
        axes[index[7], 7].axis('off')
        index[7] = index[7] + 1
    if (labels[int(y_train[i])] == 'ship') and (index[8] < M):
        axes[index[8], 8].imshow(x_train[i, 1:])
        axes[0, 8].set_title(labels[int(y_train[i])])
        axes[index[8], 8].axis('off')
        index[8] = index[8] + 1
    if (labels[int(y_train[i])] == 'truck') and (index[9] < M):
        axes[index[9], 9].imshow(x_train[i, 1:])
        axes[0, 9].set_title(labels[int(y_train[i])])
        axes[index[9], 9].axis('off')
        index[9] = index[9] + 1

x_test = x_test / 255
x_train = x_train / 255

y_test = y_test.flatten(order='C')
y_train = y_train.flatten(order='C')

y_categorical_test = to_categorical(y_test, 10)
y_categorical_train = to_categorical(y_train, 10)


def dense_model(input_shape, num_classes):
    model = keras.Sequential()
    model.add(Flatten())

    model.add(Dense(2048, activation='relu'))

    model.add(Dense(1024, activation='relu'))

    model.add(Dense(512, activation='relu'))

    model.add(Dense(256, activation='relu'))

    model.add(Dropout(0.3))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=SGD(learning_rate=0.01), metrics=['accuracy'])
    return model


def train_model(model, x, y, x_test, y_test, batch_size=128, epochs=10):
    earlyStop = EarlyStopping(monitor='val_loss', patience=2)
    dataGenerator = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
    trainGenerator = dataGenerator.flow(x, y_categorical_train, batch_size)
    steps = x.shape[0] // batch_size
    model.fit(trainGenerator, epochs=10, steps_per_epoch=steps, validation_data=(x_test, y_categorical_test),
              callbacks=[earlyStop], batch_size=batch_size)
    print(model.summary())


model = dense_model(x_train.shape[1:], 10)
train_model(model, x_train, y_train, x_test, y_test)


def cnn_model(input_shape, num_classes):
    model = keras.models.Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", padding='same'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(128, kernel_size=(3, 3), activation="relu"))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Flatten())

    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.3))

    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=SGD(learning_rate=0.01), metrics=['accuracy'])

    return model


# Create and train model
model_cnn = cnn_model(x_train.shape[0:], 10)
train_model(model_cnn, x_train, y_train, x_test, y_test)

# Make some predictions
testImage = 160
plt.imshow(x_test[testImage])
labels[int(np.argmax(model_cnn.predict(x_test[testImage].reshape(1, 32, 32, 3))))]


def plot_confusion_matrix(cm, classes, cmap=plt.cm.binary):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.xticks(np.arange(len(classes)), classes)
    plt.yticks(np.arange(len(classes)), classes)

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'), horizontalalignment="center",
                 color="white" if cm[i, j] > cm.max() / 2 else "black")

    plt.ylabel('True category')
    plt.xlabel('Predicted category')


plot_confusion_matrix(confusion_matrix(y_test, model_cnn.predict(x_test).argmax(axis=1)), list(range(10)))


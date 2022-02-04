import tensorflow.keras.datasets.mnist as mnist
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 255
x_test  = x_test / 255

print("x_train shape: ")
print(x_train.shape)
print("y_train shape: ")
print(y_train.shape)

images = x_train[:10]
labels = y_train[:10]

fig, axes = plt.subplots(1, 10,  figsize=(15,15))
for i in range(10):
    axe = axes[i]
    axe.imshow(images[i], cmap='gray')
    axe.set_title(format(labels[i]))
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy.linalg import eigh

x_train = np.reshape(x_train,(60000,784))

u,s,vh = np.linalg.svd(x_train, full_matrices=False)

X = StandardScaler().fit_transform(x_train)

values, vectors = eigh(np.matmul(X.T, X), eigvals = (782, 783))

T = np.vstack((np.matmul(vectors.T, X.T), y_train)).T

dataFrame = pd.DataFrame(T, columns = ['First principle component', 'Second principle component', 'Marker'])

sns.FacetGrid(dataFrame, hue = 'Marker', height= 5).map(sns.scatterplot, 'First principle component', 'Second principle component')
plt.show()

N = np.matrix(x_train.T) * np.matrix(x_train)
A,B,C = np.linalg.svd(N)
eigenValues = B ** 2 / np.sum(B ** 2)

figure = plt.figure(figsize=(10, 5))
singularValues = np.arange(len(eigenValues)) + 1

singularValuesPlot = [0] * 30
eigenValuesPlot = [0] * 30
for i in range(0, 30):
  singularValuesPlot[i] = singularValues[i + 1]
  eigenValuesPlot[i] = eigenValues[i + 1]

plt.plot(singularValuesPlot, eigenValuesPlot, 'o--', linewidth=2)
plt.xlabel('Principal Component')
plt.ylabel('Eigenvalue')
plt.show()

from sklearn import decomposition

pca = decomposition.PCA()
dataFrameScaled = pd.DataFrame(StandardScaler().fit_transform(X))
dataFramePCA = pd.DataFrame(pca.fit_transform(dataFrameScaled))

cumSum = np.cumsum(pca.explained_variance_ratio_) * 100
numOfMainComponents = np.argmax(cumSum >= 95)

plt.figure(figsize=[10, 5])
plt.ylabel('Cumulative Variance')
plt.xlabel('Principal components')
plt.axvline(x = numOfMainComponents)
plt.plot(cumSum)

print("Number of components capturing 95% of the variance: ")
print(numOfMainComponents)

from sklearn.decomposition import IncrementalPCA


def plot_at_principleComponentVector(numOfMainComponents):
    incrementalPCA = IncrementalPCA(n_components=numOfMainComponents)
    imageTransform = incrementalPCA.inverse_transform(incrementalPCA.fit_transform(x_train))
    imageTransform = np.reshape(imageTransform, (60000, 28, 28))
    images = imageTransform[:5]
    labels = y_train[:5]
    fig, axes = plt.subplots(1, 5, figsize=(15, 15))
    for i in range(5):
        axe = axes[i]
        axe.imshow(images[i], cmap='gray')
        axe.set_title(format(labels[i]))


principleComponentVector = [5, 15, 30, 100]

for i in range(0, 4):
    plot_at_principleComponentVector(principleComponentVector[i])
plt.show()

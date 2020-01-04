import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Encoded.csv')

X = dataset.iloc[:, :].values
y = dataset.iloc[:, :-1].values
X[:,0]*=-1
y[:,0]*=-1

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense


classifier = Sequential()
classifier.add(Dense(units = 129, kernel_initializer = 'uniform', activation = 'relu', input_dim = 129))
classifier.add(Dense(units = 56, kernel_initializer = 'uniform', activation = 'relu', input_dim = 56))
classifier.add(Dense(units = 10, kernel_initializer = 'uniform', activation = 'relu', input_dim = 10))
classifier.add(Dense(units = 50, kernel_initializer = 'uniform', activation = 'relu', input_dim = 50))
classifier.add(Dense(units = 128, kernel_initializer = 'uniform', activation = 'relu', input_dim = 128))


classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier.fit(X, y, batch_size = 10, epochs = 100)

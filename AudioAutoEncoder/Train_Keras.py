import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('end.csv')

X = dataset.iloc[:, :].values
y = dataset.iloc[:, :-1].values

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense


classifier = Sequential()
classifier.add(Dense(units = 18, kernel_initializer = 'uniform', activation = 'relu', input_dim = 18))
classifier.add(Dense(units = 9, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 3, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 9, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 17, kernel_initializer = 'uniform', activation = 'relu'))


classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier.fit(X, y, batch_size = 10, epochs = 100)

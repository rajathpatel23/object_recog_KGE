# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 18:58:38 2018

@author: rpsworker
"""

import pandas as pd
import io

import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn import metrics
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.optimizers import SGD
from sklearn.utils import shuffle
import tensorflow as tf
from keras import backend as K
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn import datasets, neighbors, linear_model


data_location = "C:\\Users\\rpsworker\\Documents\\NLP\\data_project\\"


num_cores = 6
CPU = True
GPU = False
if GPU:
    num_GPU = 1
    num_CPU = 4
if CPU:
    num_CPU = 1
    num_GPU = 0
num_classes = 2
config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,\
        inter_op_parallelism_threads=num_cores, allow_soft_placement=True,\
        device_count = {'CPU' : num_CPU, 'GPU' : num_GPU})
session = tf.Session(config=config)
K.set_session(session)

data = pd.read_pickle(data_location+"train_vec_file.pkl")
data = shuffle(data)

X = data["object_vec"].tolist()
y = data["label_type"].tolist()
X = np.array(X)
y = np.array(y)
model = Sequential()
model.add(Dense(128, activation='relu', input_dim=100))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(12, activation='softmax'))
model.summary()
cv = StratifiedKFold(n_splits= 10, shuffle=True)
output_list = []
for train, test in cv.split(X, y):
    X_train, X_test = X[train], X[test]
    y_train, y_test = y[train], y[test]
    y_train_1 = to_categorical(y_train)
    y_test_1 = to_categorical(y_test)
    model = Sequential()
    model.add(Dense(128, activation='relu', input_dim=100))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(12, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer= 'adam',
                  metrics=['accuracy'])
    model.fit(X_train, y_train_1,
              epochs=200,
              batch_size=500, validation_data=(X_test, y_test_1))
    score = model.evaluate(X_test, y_test_1, batch_size=128)
    print(score[1])
    output_list.append(score[1])
output_list = np.array(output_list)
r = np.mean(output_list)
print(r)
model.save("baseline_model.h5")

loss_list = model.history.history["loss"]
loss_val = model.history.history["val_loss"]
plt.plot(loss_list, label="training loss")
plt.plot(loss_val, label="validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss value")
plt.legend()
plt.show()

train_acc = model.history.history["acc"]
val_acc = model.history.history["val_acc"]
plt.plot(train_acc, label="training accuracy")
plt.plot(val_acc, label="validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy value")
plt.legend()
plt.show()


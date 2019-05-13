# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 10:14:44 2018

@author: rajat.patel
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import pickle
import tensorflow as tf
from sklearn import metrics
from keras.models import Sequential, Model
from keras.layers.core import Activation, Dropout, Flatten, Dense
from keras.layers.merge import Concatenate
from keras.layers import Conv1D, MaxPooling1D,Input
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.optimizers import SGD
from sklearn.utils import shuffle
import tensorflow as tf
from keras import backend as K
import matplotlib.pyplot as plt
from keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

#%%


num_cores = 6
CPU = False
GPU = True
if GPU:
    num_GPU = 1
    num_CPU = 6
if CPU:
    num_CPU = 1
    num_GPU = 0
num_classes = 2
config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,\
        inter_op_parallelism_threads=num_cores, allow_soft_placement=True,\
        device_count = {'CPU' : num_CPU, 'GPU' : num_GPU})
session = tf.Session(config=config)
K.set_session(session)

data_location1 = "C:\\Users\\rpsworker\\Documents\\NLP\\COCO_IMAGE\\COCO_IMAGES\\train2017\\"
file_location = "C:\\Users\\rpsworker\\Documents\\NLP\\data_project\\"


image_file_names = pickle.load(open(file_location+"label_to_imagedict.pickle", "rb"))
image_objects = pickle.load(open(file_location+"label_to_object_final.pickle", "rb"))
#%%
X = []
X_image_file = []
y = []
for label in image_objects.keys():
    for k in range(len(image_objects[label])):
        if len(image_objects[label][k]) > 0:
            file_name= image_file_names[label][k]["file_name"]
            X_image_file.append(file_name)
#%%
data_location = "C:\\Users\\rpsworker\\Documents\\NLP\\data_project\\"
data = pd.read_pickle(data_location+"train_vec_file.pkl")
X = data["object_vec"].tolist()
y = data["label_type"].tolist()
y = np.array(y)
X_image = np.load(data_location + "encoded_image_vec.npz")
X_image_vec = X_image["arr_0"]
X_image_vec = X_image_vec.reshape(X_image_vec.shape[0], X_image_vec.shape[1])
X = np.array(X)
X_input = np.concatenate((X_image_vec, X), axis=1)

Input_data = Input(shape=(4096, 1))
x = Conv1D(32, 4, activation="relu")(Input_data)
x = MaxPooling1D(pool_size=2)(x)
x = Conv1D(16, 4, activation="relu")(x)
x = MaxPooling1D(pool_size=2)(x)
x = Conv1D(8, 4, activation="relu")(x)
x = MaxPooling1D(pool_size=2)(x)
x = Conv1D(4, 4, activation="relu")(x)
x = MaxPooling1D(pool_size=2)(x)
x = Conv1D(2, 4, activation="relu")(x)
x = MaxPooling1D(pool_size=2)(x)
x = Flatten()(x)
em_input = Input(shape = (100, ), name="em_input")
x = Concatenate()([x, em_input])
x = Dense(64, activation="relu")(x)
soft_out = Dense(12, activation="softmax")(x)
model = Model(inputs=[Input_data,em_input ], output=soft_out)
model.summary()

cv = StratifiedKFold(n_splits= 10, shuffle=True)
output_list = []
for train, test in cv.split(X_input, y):
    X_train, X_test = X_input[train], X_input[test]
    y_train, y_test = y[train], y[test]
    y_train_1 = to_categorical(y_train)
    y_test_1 = to_categorical(y_test)
    X_img_train = X_train[:, :4096]
    X_em_train = X_train[:, 4096:]
    X_img_test = X_test[:, :4096]
    X_em_test = X_test[:, 4096:]
    X_img_train = X_img_train.reshape(X_img_train.shape[0], X_img_train.shape[1], 1)
    X_img_test = X_img_test.reshape(X_img_test.shape[0], X_img_test.shape[1], 1)
    model.compile(loss='categorical_crossentropy',
              optimizer= 'adam',
              metrics=['accuracy'])
    model.fit([X_img_train, X_em_train], y_train_1,
              validation_data=([X_img_test, X_em_test], y_test_1),
              epochs=40, batch_size=512)
    score = model.evaluate([X_img_test, X_em_test], y_test_1, batch_size=512)
    print(score[1])
    output_list.append(score[1])
model.save("proposed_model.h5")

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
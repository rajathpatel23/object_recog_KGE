# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 23:55:39 2018

@author: rpsworker
"""

# load the model
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
import pickle

data_location = "C:\\Users\\rpsworker\\Documents\\NLP\\COCO_IMAGE\\COCO_IMAGES\\train2017\\"
file_location = "C:\\Users\\rpsworker\\Documents\\NLP\\data_project\\"


image_file_names = pickle.load(open(file_location+"label_to_imagedict.pickle", "rb"))
image_objects = pickle.load(open(file_location+"label_to_object_final.pickle", "rb"))


X = []
X_image_file = []
y = []
for label in image_objects.keys():
    for k in range(len(image_objects[label])):
        if len(image_objects[label][k]) > 0:
            X.append(image_objects[label][k])
            file_name= image_file_names[label][k]["file_name"]
            X_image_file.append(file_name)
            y.append(label)

encoded_image_vec = []
model = VGG16(weights='imagenet', include_top=True)
model.summary()
for file_name in X_image_file:
    img_path = data_location + file_name
    img = load_img(img_path, target_size=(224, 224))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
#     print(features.shape)
    model_extractfeatures = Model(input=model.input, output=model.get_layer('fc2').output)
    fc2_features = model_extractfeatures.predict(x)
#     print(fc2_features.shape)
    fc2_features = fc2_features.reshape((4096,1))
    encoded_image_vec.append(fc2_features.tolist())

encoded_image_vec = np.array(encoded_image_vec)
print(encoded_image_vec.shape)
np.savez_compressed(file_location + "encoded_image_vec", encoded_image_vec)

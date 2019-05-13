# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 23:15:47 2018

@author: rpsworker
"""

import pandas as pd
import numpy as np
import pickle
from gensim.models import KeyedVectors
from gensim.test.utils import datapath, get_tmpfile
from collections import Counter, defaultdict
from gensim.models import Word2Vec
from gensim.scripts.glove2word2vec import glove2word2vec
from sklearn.preprocessing import LabelEncoder

#%%
data_location = "C:\\Users\\rpsworker\\Documents\\NLP\\data_project\\"
model_location  = "C:\\Users\\rpsworker\\Documents\\GitHub\\KGE-FE-SR_1\\Glove_data"
read_object = pickle.load(open(data_location+"label_to_object_final.pickle", "rb"))

X = []
y = []

for label in read_object:
    for obj in range(len(read_object[label])):
        if len(read_object[label][obj]) > 0:
            X.append(read_object[label][obj])
            y.append(label)
        else:
            print(False)

le = LabelEncoder()
le.fit(y)
print(le.classes_)
y_label = le.transform(y)

#%%
def entity_embedding_mat(model_1, X_vec, X1):
    y1 = []
    for i in range(len(X1)):
        if len(X[i]) > 0:
            vector = np.zeros((100, ))
            for j in range(len(X[i])):
    #            print(X[i])
                a = X[i][j].strip().split(" ")

                temp_vec = np.zeros((100, ))
                for k in a:
                    if k in model_1.wv.vocab.keys():
                        temp_vec+=model.wv[k]
                    else:
                        temp_vec+=np.random.normal(0,1, (100, ))
                vector+=temp_vec
            vector = vector/len(X[i])
            X_vec.append(vector)
    return(X_vec)
#%%
def build_vector(object_vec, label_name, label_type):
    df = pd.DataFrame(columns = ["object_vec","label_name", "label_type"])
    df["object_vec"] = object_vec
    df["label_name"] = label_name
    df["label_type"] = label_type
    return(df)
#%%
glove_file = datapath(model_location+"\\glove.6B.100d.txt")
tmp_file = get_tmpfile("test_word2vec.txt")
glove2word2vec(glove_file, tmp_file)
model = KeyedVectors.load_word2vec_format(tmp_file)
#%%
X_vec_1 = []

#%%
X_vec_1= entity_embedding_mat(model, X_vec_1, X)

#%%
for k in range(len(X_vec_1)):
    if np.any(np.isnan(X_vec_1[k])):
        print(k)
        print(True)
#%%
train_data = build_vector(X_vec_1, y, y_label)
train_data.to_pickle(data_location+"train_vec_file.pkl", protocol=2)
#%%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_vec_1, y_label, test_size=0.2, random_state=43)

#%%
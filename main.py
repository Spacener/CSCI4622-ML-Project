# ------------------------------------------------
# Team LearnCraft
# 4/20/22
#
# main.py
# ------------------------------------------------


from matplotlib.image import imread
import csv
#import learncraft_v1
from generateData import generate_n_images
import tensorflow.keras as tfk
from tensorflow.nn import selu as swish

import numpy as np
import os
import pandas as pd 

from PIL import Image

#import keras_tuner as kt
import keras
import keras.layers
import keras.callbacks
import keras.utils.all_utils as kr_utils
import keras.regularizers
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
# import tensorflow_addons as tfa

from sklearn.model_selection import train_test_split
print("-"*500)
n = 5000

n = 50
if False:
    generate_n_images(n, showImages=False)

X = []
Y = []
ores = []
num_classes = 7
# traverse the range of features
for data_index in range(n):
    #print("[DATA]: Reading datapoint: {}".format(data_index))

    # read the corresponding image feature
    img = imread("data/image_{}.png".format(data_index))
    onehot_list = [0]*7

    # read the corresponding file of location features
    with open('data/data_{}.csv'.format(data_index), newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) ==0:
                continue
            index = row[2]

            onehot_list[int(index)] = 1

    Y.append(onehot_list)
    # write ore data to a list
    X.append(img.tolist())
    #Y.append(onehot_list)


#labels = np.zeros((len(Y),7))
#for i, y in enumerate(Y):
#    labels[i] = kr_utils.to_categorical(y, num_classes=7)
#Y = list(labels)
print('\n\n\n\n\n')
x = X
while type(x) == list:
    print(len(x),end=" ")
    x = x[0]
    
print(type(x))

print(Y)

x = Y
while type(x) == list:
    print(len(x),end=" ")
    x = x[0]

print('\n\n\n\n\n')
train_percent = .7

X_train = X[0:int(len(X)*train_percent)]
Y_train = Y[0:int(len(Y)*train_percent)]

X_test = X[int(len(X)*train_percent):]
Y_test = X[int(len(Y)*train_percent):]
print(len(X_train),len(Y_train))
initial_model = tfk.applications.inception_v3.InceptionV3(input_shape=(576,576,3), include_top=False)
print("after init_model")
base_out = initial_model.output

l1 = tfk.layers.GlobalAveragePooling2D()(base_out)

temp = tfk.layers.Dense(512, activation=swish)(l1) 
temp = tfk.layers.Dropout(0.5)(temp)
temp = tfk.layers.Dense(256, activation=swish)(temp)
temp = tfk.layers.Dropout(0.5)(temp)
temp = tfk.layers.Dense(num_classes, activation='softmax')(temp)
tmodel = tfk.models.Model(inputs=initial_model.input, outputs=temp)
print("after tmodel")
#tmodel.summary()

tmodel.compile(loss=tfk.losses.CategoricalCrossentropy(), # this means that teh network returns the log probabilities and not probas
              optimizer=tfk.optimizers.Adam(learning_rate=5e-5), # The optimizer that smooths the gradient
              metrics=["accuracy"]) # We want to track accuracy

print("after compiling")
checkpoint_callbk = tf.keras.callbacks.ModelCheckpoint(
    "best_tiny_model", # name of file to save the best model to
    monitor="accuracy", # prefix val to specify that we want the model with best macroF1 on the validation data
    verbose=1, # prints out when the model achieve a better epoch
    mode="max", # the monitored metric should be maximized
    save_freq="epoch", # clear
    save_best_only=True, # of course, if not, every time a new best is achieved will be savedf differently
    save_weights_only=True # this means that we don't have to save the architecture, if you change the architecture, you'll loose the old weights
)

reduceLR_callbk = keras.callbacks.ReduceLROnPlateau(monitor='accuracy', factor=0.6, patience=8, verbose=1, mode='max', min_lr=5e-5)

print("after reduceLR_callbk")
Y_train_tensor = tf.convert_to_tensor(Y_train)
print("after Y_train_tensor")
tmodel.fit(X_train, tf.convert_to_tensor(Y_train), callbacks=[reduceLR_callbk, checkpoint_callbk], epochs=10, validation_split=.3, batch_size = 1)

tmodel.load_weights("best_tiny_model")

#y_hat = tmodel.predict(test_generator) # logits of the 53 classes
#y_hat = np.argmax(y_hat, axis=1) # take the classe with the hgiher logit
#test_generator.df.label = y_hat
#test_generator.df.to_csv("submission.csv", index=False) # we don't want to add the column of indices

# print(X[0],Y[0])

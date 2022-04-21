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

n = 5000

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
    onehot_list = np.zeros(7)


    # read the corresponding file of location features
    with open('data/data_{}.csv'.format(data_index), newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) ==0:
                continue
            index = row[2]

            onehot_list[int(index)] = 1


    # write ore data to a list
    X.append([img])
    Y.append(onehot_list)

X = np.asarray(X)
Y = np.asarray(Y)

train_percent = .7

X_train = X[0:int(len(X)*train_percent)]
Y_train = Y[0:int(len(Y)*train_percent)]

X_test = X[int(len(X)*train_percent):]
Y_test = X[int(len(Y)*train_percent):]

initial_model = tfk.applications.inception_v3.InceptionV3(input_shape=(400,400,3), include_top=False)

base_out = initial_model.output

l1 = tfk.layers.GlobalAveragePooling2D()(base_out)

temp = tfk.layers.Dense(512, activation=swish)(l1) 
temp = tfk.layers.Dropout(0.5)(temp)
temp = tfk.layers.Dense(256, activation=swish)(temp)
temp = tfk.layers.Dropout(0.5)(temp)
temp = tfk.layers.Dense(num_classes, activation='softmax')(temp)
tmodel = tfk.models.Model(inputs=initial_model.input, outputs=temp)
    
tmodel.summary()
    
tmodel.compile(loss=tfk.losses.CategoricalCrossentropy(), # this means that teh network returns the log probabilities and not probas
              optimizer=tfk.optimizers.adam_v2.Adam(learning_rate=5e-5), # The optimizer that smooths the gradient
              metrics=["accuracy"]) # We want to track accuracy

checkpoint_callbk = tf.keras.callbacks.ModelCheckpoint(
    "best_tiny_model", # name of file to save the best model to
    monitor="accuracy", # prefix val to specify that we want the model with best macroF1 on the validation data
    verbose=1, # prints out when the model achieve a better epoch
    mode="max", # the monitored metric should be maximized
    save_freq="epoch", # clear
    save_best_only=True, # of course, if not, every time a new best is achieved will be savedf differently
    save_weights_only=True # this means that we don't have to save the architecture, if you change the architecture, you'll loose the old weights
)

reduceLR_callbk = keras.callbacks.ReduceLROnPlateau(monitor='val_macroF1', factor=0.6, patience=8, verbose=1, mode='max', min_lr=5e-5)

tmodel.fit(X_train, Y_train, callbacks=[reduceLR_callbk, checkpoint_callbk], epochs=100, validation_split=.3, batch_size = 1)

tmodel.load_weights("best_tiny_model")

y_hat = tmodel.predict(test_generator) # logits of the 53 classes
y_hat = np.argmax(y_hat, axis=1) # take the classe with the hgiher logit
test_generator.df.label = y_hat
test_generator.df.to_csv("submission.csv", index=False) # we don't want to add the column of indices

#print(X[0],Y[0])
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
n = 2000
import random
random.seed(42)
num_classes = 9

generate_n_images(n, showImages=False,sub_directory="train")
generate_n_images(n*0.5, showImages=False,sub_directory="test")

train_generator = tfk.preprocessing.image_dataset_from_directory("./data/train", image_size=(500,500),validation_split=.3, subset="training",seed=42)
valid_generator = tfk.preprocessing.image_dataset_from_directory("./data/train", image_size=(500,500),validation_split=.3, subset="validation",seed=42)
test_generator = tfk.preprocessing.image_dataset_from_directory("./data/test", image_size=(500,500),seed=42)

def preprocess(generator):
    return generator.map(lambda x, y: ( tfk.applications.inception_v3.preprocess_input(x),y))

train_generator = preprocess(train_generator) 
valid_generator = preprocess(valid_generator)
test_generator  = preprocess(test_generator)
    
initial_model = tfk.applications.inception_v3.InceptionV3(input_shape=(500,500,3), include_top=False)

base_out = initial_model.output

l1 = tfk.layers.GlobalAveragePooling2D(name="end_of_inception")(base_out)

temp = tfk.layers.Dense(512, activation=swish)(l1) 
temp = tfk.layers.Dropout(0.5)(temp)
temp = tfk.layers.Dense(256, activation=swish)(temp)
temp = tfk.layers.Dropout(0.5)(temp)
temp = tfk.layers.Dense(num_classes)(temp)
tmodel = tfk.models.Model(inputs=initial_model.input, outputs=temp)
print("after tmodel")
#tmodel.summary()

tmodel.compile(loss=tfk.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer=tfk.optimizers.SGD(learning_rate = 5e-5), metrics=["accuracy"])

print("after compiling")
checkpoint_callbk = tf.keras.callbacks.ModelCheckpoint("learncraft", monitor="val_accuracy", verbose=1, mode="max", save_freq="epoch", save_best_only=True, save_weights_only=True)

reduceLR_callbk = keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.6, patience=8, verbose=1, mode='max', min_lr=5e-5)

print("after reduceLR_callbk")
tmodel.fit(train_generator, callbacks=[reduceLR_callbk, checkpoint_callbk],validation_data=valid_generator,epochs=25)

#tmodel.load_weights("learncraft")

#y_hat = tmodel.predict(test_generator)
#y_hat = np.argmax(y_hat, axis=1) 

# print(X[0],Y[0])

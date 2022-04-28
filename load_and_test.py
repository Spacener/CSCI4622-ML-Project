# ------------------------------------------------
# Team LearnCraft
# 4/28/22
#
# load_and_test.py
# ------------------------------------------------
'''
a python script that loads the already fitted learncraft code
and calculates and reports the accuracy on a test set

'''

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
n = 1000
import random
random.seed(42)
num_classes = 9
user_input = input("Generate new test set? (y/n)")
while user_input.lower() not in ['y','yes','n','no']:
    print("invalid input")
    user_input = input("Generate new test set? (y/n)")

generate_new_test_set = 'y' in user_input # set
if generate_new_test_set:
    n = int(input("size of test set?"))
    generate_n_images(n, showImages=False,sub_directory="test")


test_generator = tfk.preprocessing.image_dataset_from_directory("./data/test", image_size=(500,500),seed=42,batch_size=1)


test_generator =  test_generator.map(lambda x, y: ( tfk.applications.inception_v3.preprocess_input(x),y))

initial_model = tfk.applications.inception_v3.InceptionV3(input_shape=(500,500,3), include_top=False)

base_out = initial_model.output

l1 = tfk.layers.GlobalAveragePooling2D(name="end_of_inception")(base_out)

temp = tfk.layers.Dense(512, activation=swish)(l1) 
temp = tfk.layers.Dropout(0.5)(temp)
temp = tfk.layers.Dense(256, activation=swish)(temp)
temp = tfk.layers.Dropout(0.5)(temp)
temp = tfk.layers.Dense(num_classes)(temp)
tmodel = tfk.models.Model(inputs=initial_model.input, outputs=temp)

tmodel.load_weights("learncraft")

correct = 0
i = 0
for batch in test_generator.as_numpy_iterator():
    if(i%int((n/100)) == 0):
        progress = int(i/int(n/100))
        print("[{}{}]".format("="*progress," "*(100-progress)),end='\r')
    y_hat = tmodel.predict(batch[0])
    y_hat = np.argmax(y_hat, axis=1) 
    batch_correct = sum([y_pred == y_true for y_pred, y_true in zip(y_hat,batch[1])])
    correct += batch_correct
    i+=1;
    
print("{}\naccuracy on test set:{}".format(" "*102,correct/1000))

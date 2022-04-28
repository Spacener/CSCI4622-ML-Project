import tensorflow.keras as tfk
from tensorflow.nn import selu as swish
import cv2
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
from matplotlib.image import imread

# load the model
initial_model = tfk.applications.inception_v3.InceptionV3(input_shape=(500,500,3), include_top=False)

base_out = initial_model.output

l1 = tfk.layers.GlobalAveragePooling2D(name="end_of_inception")(base_out)

temp = tfk.layers.Dense(512, activation=swish)(l1) 
temp = tfk.layers.Dropout(0.5)(temp)
temp = tfk.layers.Dense(256, activation=swish)(temp)
temp = tfk.layers.Dropout(0.5)(temp)
temp = tfk.layers.Dense(9)(temp)
tmodel = tfk.models.Model(inputs=initial_model.input, outputs=temp)

tmodel.load_weights("learncraft")
# ask user for image input
directoryname = input("input path to image to classify (can be a single image or a path to a directory of images):")
while not os.path.exists(directoryname):
    print("path not found")
    directoryname = input("input path to image to classify:")

labels = os.listdir("data/train/")
resize = tfk.layers.Resizing(500,500)

def classify_image(image,filename):
    images = np.array([image])
    resized = tfk.layers.Resizing(500, 500)(images)
    processed_image = tfk.applications.inception_v3.preprocess_input(resized)
    y = tmodel.predict(processed_image)
    classification = labels[np.argmax(y)]
    print("{} \t\t {}".format(filename,classification))

if(os.path.isfile(directoryname)):
    # given just one file, classify the image
    image = np.array(Image.open(directoryname).convert("RGB"))
    classify_image(image,directoryname)
else:
    # given a directory classify each file in the directory (ignoring subdirectories) 
    for file in os.listdir(directoryname):
        filename = "{}{}{}".format(directoryname,"/" if directoryname[-1]!="/" else "",file)
        if(os.path.isfile(filename)):
            image =np.array(Image.open(filename).convert("RGB"))
            classify_image(image,filename)
            
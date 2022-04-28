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

directoryname = input("input path to image to classify:")
while not os.path.exists(directoryname):
    print("path not found")
    directoryname = input("input path to image to classify:")

image = cv2.imread(directoryname)
if image.shape[2] == 4:
    image = image[:,:,:3]
    print(image.shape)
resized = tfk.layers.Resizing(500, 500)(image)
processed_image = tfk.applications.inception_v3.preprocess_input(resized)
print(np.array([processed_image]).shape)
y = tmodel.predict(np.array([processed_image]))
print(np.argmax(y),y)
classification = os.listdir("data/train/")[np.argmax(y)]
print(os.listdir("data/train/"))
cv2.imshow("Classified as {}".format(classification), imread(directoryname))
cv2.waitKey(0)
cv2.destroyAllWindows()


    
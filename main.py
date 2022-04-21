# ------------------------------------------------
# Team LearnCraft
# 1/20/22
#
# main.py
# ------------------------------------------------


import cv2
import numpy as np
import csv
import learncraft_v1


n = 100
X = []
Y = []
ores = []
# traverse the range of features
for data_index in range(n):
    print("[DATA]: Reading datapoint: {}".format(data_index))

    # read the corresponding image feature
    img = cv2.imread("data/image_{}.png".format(data_index))

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
    X.append(img)
    Y.append(onehot_list)

print(Y)
# initial_model = keras.applictions.inception_v3.InceptionV3(input_shape=X.shape, )
    
print(X[0],Y[0])
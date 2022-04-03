# ------------------------------------------------
# Team LearnCraft
# 1/20/22
#
# main.py
# ------------------------------------------------


import cv2
#import numpy as np
#import csv
n = 100
X = []
Y = []
# traverse the range of features
for data_index in range(n):
    print("[DATA]: Reading datapoint: {}".format(data_index))

    # read the corresponding image feature
    img = cv2.imread("data/image_{}.png".format(data_index))
    ore = []

    # read the corresponding file of location features
    with open('data/data_{}.csv'.format(data_index), newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            ore.append((row[0], row[1]))

    # write ore data to a list
    ores.append(ore)
    X.append(img)
    Y.append(ores)

print(X[0],Y[0])
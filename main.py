# ------------------------------------------------
# Team LearnCraft
# 1/20/22
#
# main.py
# ------------------------------------------------


import cv2
import numpy as np
import csv

ores = [] # initialize an empty list to store the features in

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
    print(ore)

    # display the corresponding image
    cv2.imshow(img, "Sample {}".format(data_index))
    cv2.waitkey(0)
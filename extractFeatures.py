# ------------------------------------------------
# Team LearnCraft
# 1/20/22
#
# extractFeatures.py
# contains functions to extract color-related features
# ------------------------------------------------


import cv2
import numpy as np
from generateData import generate_n_images
import csv
import imutils
import matplotlib.pyplot as plt

# Gather images and ore locations
n = 100
show = False

generate_n_images(n=n, showImages=False, saveImages=True)

ores = []  # initialize an empty list to store the features in

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


    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(gray, 30, 150)
    thresh = cv2.threshold(gray, 160, 220, cv2.THRESH_BINARY_INV)[1]

    # blackThresh = cv2.threshold(gray, 0, 30, cv2.THRESH_BINARY_INV)[1]
    #if show:
    #    cv2.imshow("blackThresh", thresh)

    mask = thresh.copy()
    output = cv2.bitwise_and(img, img, mask=mask)

    # we apply erosions to reduce the size of foreground objects
    mask = thresh.copy()
    mask = cv2.erode(mask, None, iterations=3)
    mask = cv2.bitwise_not(mask)

    inputImg = mask.copy() # threheld and eroded image
    connectivity = 4

    (numLabels, labels, stats, centroids) = cv2.connectedComponentsWithStats(
        thresh, connectivity, cv2.CV_32S)

    # print("n_labels: {}".format(numLabels))
    # print("labels: {}".format(labels))
    # print("centroids: {}".format(centroids))

    for i in range(0, numLabels-1):


        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        if not np.isnan(centroids[i][0]):
            (cX, cY) = (int(centroids[i][0]), int(centroids[i][1]))

            # print("centroids[i]: {}".format(centroids[i]))
            cv2.circle(img, (cX, cY), 20, (0,0,255), 2)
        '''
        if show:
            # cv2.imshow("Eroded Mask", mask)
            cv2.imshow("Thresh", thresh)
            print("[ORE[{}]]: {}".format(i,ores[i]))
            for ore in ores[i]:
                print("ore: {}".format(ore))
                cv2.circle(img, (int(ore[0]),int(ore[1])), 10, (0,255,255), 2)
            cv2.imshow("image", img)

            cv2.waitKey(0)
        '''
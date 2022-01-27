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
n = 1

generate_n_images(n=n, showImages=False, saveImages=True)

for data_index in range(n):
    print("[DATA]: Reading datapoint: {}".format(data_index))
    img = cv2.imread("data/image_{}.png".format(data_index))
    ore = []
    with open('data/data_{}.csv'.format(data_index), newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            ore.append(row)

    # load the input image (whose path was supplied via command line
    # argument) and display the image to our screen
    # cv2.imshow("Image", img)

    blurred = cv2.GaussianBlur(img, (5, 5), 0)

    # convert the image to grayscale
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("Gray", gray)

    # applying edge detection we can find the outlines of objects in
    # images
    edged = cv2.Canny(gray, 30, 150)
    # cv2.imshow("Edged", edged)

    # threshold the image by setting all pixel values less than 225
    # to 255 (white; foreground) and all pixel values >= 225 to 255
    # (black; background), thereby segmenting the image
    thresh = cv2.threshold(gray, 160, 220, cv2.THRESH_BINARY_INV)[1]

    # a typical operation we may want to apply is to take our mask and
    # apply a bitwise AND to our input image, keeping only the masked
    # regions
    mask = thresh.copy()
    output = cv2.bitwise_and(img, img, mask=mask)
    # cv2.imshow("Output", output)

    # we apply erosions to reduce the size of foreground objects
    mask = thresh.copy()
    mask = cv2.erode(mask, None, iterations=6)
    mask = cv2.bitwise_not(mask)

    # similarly, dilations can increase the size of the ground objects
    # mask = thresh.copy()
    # mask = cv2.dilate(mask, None, iterations=5)
    # cv2.imshow("Dilated", mask)

    # find contours (i.e., outlines) of the foreground objects in the
    # thresholded image


    inputImg = mask.copy() # threheld and eroded image
    connectivity = 4

    n = 4
    (numLabels, labels, stats, centroids) = cv2.connectedComponentsWithStats(
        thresh, connectivity, cv2.CV_32S)

    print("n_labels: {}".format(numLabels))
    print("labels: {}".format(labels))
    print("centroids: {}".format(centroids))

    for i in range(0, numLabels):


        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        if not np.isnan(centroids[i][0]):
            (cX, cY) = (int(centroids[i][0]), int(centroids[1][1]))

            print("centroids[i]: {}".format(centroids[i]))
            cv2.circle(img, (cX, cY), 2, (0,0,255), 2)


    # for (i, c) in enumerate(cnts):
    #     ((x, y), _) = cv2.minEnclosingCircle(c)
    #     # cv2.putText(output, "#{}".format(i + 1), (int(x) - 45, int(y) + 20),
    #     #             cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 5)
    #     cv2.drawContours(output, [c], -1, (0, 255, 0), 2)


    # f_image = cv2.imread("coins.jpg")
    # f, axs = plt.subplots(1, 2, figsize=(12, 5))
    # axs[0].imshow(f_image, cmap="gray")
    # axs[1].imshow(img, cmap="gray")
    # axs[1].set_title("Total Money Count = {}".format(count))

    # draw the total number of contours found in purple
    # text = "I found {} objects!".format(len(cnts))
    # cv2.putText(output, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 0, 159), 2)
    # cv2.imshow("Contours", output)

    cv2.imshow("Eroded Mask", mask)
    cv2.imshow("Thresh", thresh)
    cv2.imshow("image", img)

    cv2.waitKey(0)

# ------------------------------------------------
# Team LearnCraft
# 1/20/22
#
# main.py
# ------------------------------------------------


import cv2
import numpy as np
import learncraft_v1

import csv
n = 100
X = []
Y = []
ores = [[],[]]
# traverse the range of features
for data_index in range(n):
    print("[DATA]: Reading datapoint: {}".format(data_index))

    # read the corresponding image feature
    img = cv2.cv2.imread("data/image_{}.png".format(data_index))
    orex = []
    orey = []
    # read the corresponding file of location features
    with open('data/data_{}.csv'.format(data_index), newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if(len(row) == 0): continue
            orex.append(row[0])
            orey.append(row[1])


    # write ore data to a list
    ores[0].append(orex)
    ores[1].append(orey)
    X.append(img)
    Y.append(ores)



X = np.asarray(X)
Y = np.asarray(Y)
Trainingsize = 0.8*n
ValidationSize = 0.2*n

print("giving everything to the robit")
Robit = learncraft_v1.KNNClassifier(k=3).fit(X[:int(round(0.8*n))], Y[:int(round(0.8*n))])
print("the robit got everything")
y_hat = Robit.predict(X[int(round(0.8*n)):])

print(Robit.accuracy(y_hat,Y[int(round(0.8*n))]))

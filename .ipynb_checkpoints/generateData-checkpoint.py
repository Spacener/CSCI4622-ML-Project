# ------------------------------------------------
# Team LearnCraft
# 1/20/22
#
# generateData.py
# A script to generate random minecraft images using
# the textures of the blocks, and save the locations
# of valuable ores in the image
# ------------------------------------------------


import cv2
import numpy as np
import csv
import os
import random
from pathlib import Path

def generate_n_images(n=0, showImages=True, saveImages=True,
                     sub_directory=""):

    random.seed(42)
    directoryname = "data/"+sub_directory+"/"
    if os.path.exists(directoryname):
        for sd in os.listdir(directoryname):
            if os.path.exists(directoryname+sd):
                for f in os.listdir(directoryname+sd):
                    os.remove(directoryname+sd+"/"+f)
    
            
    if n is None:
        n = input("[PROMPT]: How many training images would you like to generate?: ")

    # initialize default paths
    valueablePath = "textures/valuableBlocks/diamond_ore.png"
    path = "textures/commonBlocks/stone.png"

    for i in range(int(n)):

        # randomly select for deepslate or stone
        if np.random.random() >= 0.5: # deepslate coloration!
            
            path = "textures/commonBlocks/deepslate.png"
            bgColor = (70, 70, 70)

            # randomly select a valuable block to include
            rand = np.random.random()
            index = -1
            if rand < 1/9:
                valueablePath = "textures/valuableBlocks/deepslate_diamond_ore.png"
                location,index = "diamond", 0
            elif rand < 2/9:
                valueablePath = "textures/valuableBlocks/deepslate_gold_ore.png"
                location,index = "gold", 1
            elif rand < 3/9:
                valueablePath = "textures/valuableBlocks/deepslate_lapis_ore.png"
                location,index = "lapis", 2
            elif rand < 4/9:
                valueablePath = "textures/valuableBlocks/deepslate_iron_ore.png"
                location,index = "iron", 3
            elif rand < 5/9:
                valueablePath = "textures/valuableBlocks/deepslate_coal_ore.png"
                location,index = "coal", 4
            elif rand < 6/9:
                valueablePath = "textures/valuableBlocks/deepslate_redstone_ore.png"
                location,index = "redstone", 5
            elif rand < 7/9:
                valueablePath = "textures/valuableBlocks/deepslate_emerald_ore.png"
                location,index = "emerald", 6
            elif rand < 8/9:
                valueablePath = "./textures/valuableBlocks/deepslate_copper_ore.png"
                location,index = "copper", 7
            else:
                valueablePath = "textures/commonBlocks/deepslate.png"
                location,index = "no-ores", 8
        else: # Stone coloration!
            path = "textures/commonBlocks/stone.png"
            bgColor = (130, 130, 130)

            # randomly select a valuable block to include
            rand = np.random.random()
            if rand < 1/9:
                valueablePath = "textures/valuableBlocks/diamond_ore.png"
                location,index = "diamond", 0
            elif rand < 2/9:
                valueablePath = "textures/valuableBlocks/gold_ore.png"
                location, index = "gold", 1
            elif rand < 3/9:
                valueablePath = "textures/valuableBlocks/lapis_ore.png"
                location,index = "lapis", 2
            elif rand < 4/9:
                valueablePath = "textures/valuableBlocks/iron_ore.png"
                location,index = "iron", 3
            elif rand < 5/9:
                valueablePath = "textures/valuableBlocks/coal_ore.png"
                location,index = "coal", 4
            elif rand < 6/9:
                valueablePath = "textures/valuableBlocks/redstone_ore.png"
                location,index = "redstone", 5
            elif rand < 7/9:
                valueablePath = "textures/valuableBlocks/emerald_ore.png"
                location,index = "emerald", 6
            elif rand < 8/9:
                valueablePath = "./textures/valuableBlocks/copper_ore.png"
                location,index = "copper", 7
            else:
                valueablePath = "./textures/commonBlocks/stone.png"
                location,index = "no-ores", 8
        print(location,index)
        backgroundBlock = cv2.imread(path)
        valuableBlock = cv2.imread(valueablePath)
        valueLocationsTiles = []
        
        # Each corner is a cell of 4 blocks
        # Four corners make a chunk of 16 square blocks


        # make a valuable corner by random placement!
        rand = np.random.random()
        if rand < 0.3:
            valueCorner = cv2.vconcat([
                cv2.hconcat([backgroundBlock, backgroundBlock]),
                cv2.hconcat([backgroundBlock, valuableBlock])
            ])
            valueLocationsTiles.append((valueCorner.shape[1]*0.75, valueCorner.shape[0]*0.75))
        elif rand < 0.6:
            valueCorner = cv2.vconcat([
                cv2.hconcat([backgroundBlock, valuableBlock]),
                cv2.hconcat([valuableBlock, valuableBlock])
            ])
            valueLocationsTiles.append((valueCorner.shape[1]*0.25, valueCorner.shape[0]*0.75))
            valueLocationsTiles.append((valueCorner.shape[1] * 0.75, valueCorner.shape[0] * 0.25))
            valueLocationsTiles.append((valueCorner.shape[1] * 0.75, valueCorner.shape[0] * 0.75))
        elif rand < 0.9:
            valueCorner = cv2.vconcat([
                cv2.hconcat([valuableBlock, backgroundBlock]),
                cv2.hconcat([valuableBlock, valuableBlock])
            ])
            valueLocationsTiles.append((valueCorner.shape[1] * 0.25, valueCorner.shape[0]*0.25))
            valueLocationsTiles.append((valueCorner.shape[1] * 0.25, valueCorner.shape[0] * 0.75))
            valueLocationsTiles.append((valueCorner.shape[1] * 0.75, valueCorner.shape[0] * 0.75))
        else: # valueless image
             valueCorner = cv2.vconcat([
                cv2.hconcat([backgroundBlock, backgroundBlock]),
                cv2.hconcat([backgroundBlock, backgroundBlock])
            ])

        # assign a dull corner, with no valuable blocks
        dullCorner = cv2.vconcat([
                cv2.hconcat([backgroundBlock, backgroundBlock]),
                cv2.hconcat([backgroundBlock, backgroundBlock])
            ])

        # 16 blocks
        dullChunk = cv2.vconcat([
                cv2.hconcat([dullCorner, dullCorner]),
                cv2.hconcat([dullCorner, dullCorner])
            ])

        # valuable chunk
        chunk = cv2.vconcat([
                cv2.hconcat([dullCorner, dullCorner]),
                cv2.hconcat([dullCorner, valueCorner])
            ])

        # combine 4 chunks
        img = cv2.vconcat([
                cv2.hconcat([dullChunk, dullChunk]),
                cv2.hconcat([dullChunk, chunk])
            ])

        tiled = np.tile(img, (2,2,1)) # tile the image

        # create an empty list to contain locations of valuable blocks
        valueLocation = []
        for value in valueLocationsTiles:
            valueLocation.append((value[0]+valueCorner.shape[1]+chunk.shape[1],
                                  value[1]+valueCorner.shape[1]+chunk.shape[1]))
            valueLocation.append((value[0] + valueCorner.shape[1] + 3*chunk.shape[1],
                                  value[1] + valueCorner.shape[1] + 3*chunk.shape[1]))
            valueLocation.append((value[0] + valueCorner.shape[1] + 3*chunk.shape[1],
                                  value[1] + valueCorner.shape[1] + chunk.shape[1]))
            valueLocation.append((value[0] + valueCorner.shape[1] + chunk.shape[1],
                                  value[1] + valueCorner.shape[1] + 3*chunk.shape[1]))

        #cv2.imshow("chunk", img)

        outputImage = tiled.copy()

        # perspective transform

        previousWidth = tiled.shape[1]
        previousHeight = tiled.shape[0]

        scale_percent = 20
        ogWidth = int(backgroundBlock.shape[1] * scale_percent)
        ogHeight = int(backgroundBlock.shape[0] * scale_percent)

        width = tiled.shape[1]

        # randomly select perspective view or flat
        if np.random.random() >= 0.5:
            leftHeight = tiled.shape[0]
            rightHeight = tiled.shape[0]
        else:
            # randomly select left or right perspective
            if np.random.random() >= 0.5:
                leftHeight = 2*tiled.shape[0]/3
                rightHeight = tiled.shape[0]*1.5
            else:
                rightHeight = 2*tiled.shape[0]/3
                leftHeight = tiled.shape[0]*1.5

            # specify conjugate x,y coordinates (not y,x)
            input = np.float32([[0,0], [previousWidth,0], [previousWidth,previousHeight], [0,previousHeight]])
            output = np.float32([[0,0], [width-1,0], [width-1,rightHeight-1], [0,leftHeight-1]])

            # compute perspective matrix
            matrix = cv2.getPerspectiveTransform(input,output)

            # do perspective transformation setting area outside input to black
            perspective = cv2.warpPerspective(tiled, # named tiled, but it's actually perspective
                                             matrix,
                                             (ogWidth,ogHeight),
                                             cv2.INTER_LINEAR,
                                             borderMode=cv2.BORDER_CONSTANT,
                                             borderValue=bgColor)

            outputImage = perspective.copy()

            # compute value matrix
            for v in range(len(valueLocation)):
                px = (matrix[0][0] * valueLocation[v][0] + matrix[0][1] * valueLocation[v][1] + matrix[0][2]) / ((matrix[2][0] * valueLocation[v][0] + matrix[2][1] * valueLocation[v][1] + matrix[2][2]))
                py = (matrix[1][0] * valueLocation[v][0] + matrix[1][1] * valueLocation[v][1] + matrix[1][2]) / ((matrix[2][0] * valueLocation[v][0] + matrix[2][1] * valueLocation[v][1] + matrix[2][2]))
                valueLocation[v] = (int(px), int(py))  # after transformation

        # crop the image randomly
        newWidth1 = int(0.5*outputImage.shape[1])
        newHeight1 = int(0.5*outputImage.shape[0])
        newWidth2 = int(0.6*outputImage.shape[1])
        newHeight2 = int(0.6*outputImage.shape[0])

        # crop the image
        randHeight = 0
        randWidth = 0
        cropped_image = outputImage[
                        randHeight:newHeight2,
                        randWidth:newWidth2
                        ]

        # scale the image
        ratio = 3
        newWidth = int(cropped_image.shape[1] * ratio)
        newHeight = int(cropped_image.shape[0] * ratio)
        dim = (newWidth, newHeight)
        resized = cv2.resize(cropped_image, dim, interpolation=cv2.INTER_AREA)

        ores = []

        for v in range(len(valueLocation)): # cropping and scaling ore locations
            if valueLocation[v][1] < randHeight or valueLocation[v][1] > newHeight2:
                valueLocation[v] = None
            elif valueLocation[v][0] < randWidth or valueLocation[v][0] > newWidth2:
                valueLocation[v] = None
            else:
                valueLocation[v] = (valueLocation[v][0] - randWidth, valueLocation[v][1] - randHeight)

            if valueLocation[v] is not None:
                valueLocation[v] = (int(valueLocation[v][0] * ratio), int(valueLocation[v][1] * ratio))
                # Want a circle? Here's a circle
                # cv2.circle(resized, (int(valueLocation[v][0]), int(valueLocation[v][1])), 20, (0, 100, 255), 2)
                ores.append((int(valueLocation[v][0]), int(valueLocation[v][1]), index))

        if showImages:
            print("\n[DEBUG]: IMAGE DISPLAYED. PRESS ANY KEY TO ESCAPE")
            for ore in range(len(ores)):
                print("[DATA]: ores[{}] at {}".format(ore, ores[ore]))

            # cv2.imshow("tiled", tiled)
            # cv2.imshow("outputImage", outputImage)
            # cv2.imshow("cropped_image", cropped_image)
            cv2.imshow("resized", resized)

            thresh = cv2.threshold(resized, 160, 220, cv2.THRESH_BINARY_INV)[1]
            thresh = cv2.bitwise_not(thresh)
            cv2.imshow("Window", thresh)

            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # save the images
        if saveImages:
            
            if not os.path.exists(sub_directory):
                if not os.path.exists(sub_directory +"/"+ location+"/"):
                    newpath = "data/"+sub_directory+"/"+ location+"/"
                    Path(newpath).mkdir(parents=True, exist_ok=True)

            if sub_directory[-1]!='/' and len(sub_directory) > 0:
                sub_directory+='/'
            print("data/{}{}/image_{}.png".format(sub_directory,location,i))
            cv2.imwrite("data/{}{}/image_{}.png".format(sub_directory,location,i), resized)
            print("[DATA]: Image saved!")


            with open('data/data_{}.csv'.format(i), mode='w') as ore_data:
                ore_writer = csv.writer(ore_data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

                for ore in ores:
                    ore_writer.writerow(ore)

        if n == 1:
            print()
            return resized, ores




    print("[SUCCESS]: All samples generated!")


# generate_n_images(100,showImages=False,saveImages=True,sub_directory="train")
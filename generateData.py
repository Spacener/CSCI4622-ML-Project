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

n_to_generate = input("[PROMPT]: How many training images would you like to generate?: ")

# initialize default paths
valueablePath = "textures/valuableBlocks/diamond_ore.png"
path = "textures/commonBlocks/stone.png"

for i in range(int(n_to_generate)):

    # randomly select for deepslate or stone
    if np.random.random() >= 0.5: # deepslate coloration!
        path = "textures/commonBlocks/deepslate.png"
        bgColor = (70, 70, 70)

        # randomly select a valuable block to include
        rand = np.random.random()
        if rand < 1/7:
            valueablePath = "textures/valuableBlocks/deepslate_diamond_ore.png"
        elif rand < 2/7:
            valueablePath = "textures/valuableBlocks/deepslate_gold_ore.png"
        elif rand < 3/7:
            valueablePath = "textures/valuableBlocks/deepslate_lapis_ore.png"
        elif rand < 4/7:
            valueablePath = "textures/valuableBlocks/deepslate_iron_ore.png"
        elif rand < 5/7:
            valueablePath = "textures/valuableBlocks/deepslate_coal_ore.png"
        elif rand < 6/7:
            valueablePath = "textures/valuableBlocks/deepslate_redstone_ore.png"
        elif rand < 1:
            valueablePath = "textures/valuableBlocks/deepslate_emerald_ore.png"

    else: # Stone coloration!
        path = "textures/commonBlocks/stone.png"
        bgColor = (130, 130, 130)

        # randomly select a valuable block to include
        rand = np.random.random()
        if rand < 1/7:
            valueablePath = "textures/valuableBlocks/diamond_ore.png"
        elif rand < 2/7:
            valueablePath = "textures/valuableBlocks/gold_ore.png"
        elif rand < 3/7:
            valueablePath = "textures/valuableBlocks/lapis_ore.png"
        elif rand < 4/7:
            valueablePath = "textures/valuableBlocks/iron_ore.png"
        elif rand < 5/7:
            valueablePath = "textures/valuableBlocks/coal_ore.png"
        elif rand < 6/7:
            valueablePath = "textures/valuableBlocks/redstone_ore.png"
        elif rand < 1:
            valueablePath = "textures/valuableBlocks/emerald_ore.png"

    backgroundBlock = cv2.imread(path)
    valuableBlock = cv2.imread(valueablePath)
    valueLocationsTiles = []

    # Each corner is a cell of 4 blocks
    # Four corners make a chunk of 16 square blocks


    # make a valuable corner by random placement!
    rand = np.random.random()
    if rand < 0.25:
        valueCorner = cv2.vconcat([
            cv2.hconcat([backgroundBlock, backgroundBlock]),
            cv2.hconcat([backgroundBlock, valuableBlock])
        ])
        valueLocationsTiles.append((valueCorner.shape[1]*0.75, valueCorner.shape[0]*0.75))
    elif rand < 0.5:
        valueCorner = cv2.vconcat([
            cv2.hconcat([backgroundBlock, valuableBlock]),
            cv2.hconcat([valuableBlock, valuableBlock])
        ])
        valueLocationsTiles.append((valueCorner.shape[1]*0.25, valueCorner.shape[0]*0.75))
        valueLocationsTiles.append((valueCorner.shape[1] * 0.75, valueCorner.shape[0] * 0.25))
        valueLocationsTiles.append((valueCorner.shape[1] * 0.75, valueCorner.shape[0] * 0.75))
    elif rand < 0.75:
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

    # cv2.imshow("chunk", img)

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
    randHeight = np.random.randint(0,newHeight1)
    randWidth = np.random.randint(0,newWidth1)
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
            cv2.circle(resized, (int(valueLocation[v][0]), int(valueLocation[v][1])), 20, (0, 100, 255), 2)
            ores.append((int(valueLocation[v][0] * ratio), int(valueLocation[v][1] * ratio)))

    print("\n[DEBUG]: IMAGE DISPLAYED. PRESS ANY KEY TO ESCAPE")
    for ore in range(len(ores)):
        print("[DATA]: ores[{}] at {}".format(ore, ores[ore]))

    # cv2.imshow("tiled", tiled)
    # cv2.imshow("outputImage", outputImage)
    # cv2.imshow("cropped_image", cropped_image)
    cv2.imshow("resized", resized)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

print("[SUCCESS]: Samples generated!")
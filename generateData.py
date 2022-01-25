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

valuable_block_names = [
    "coal_ore.png",
    "deepslate_coal_ore.png",
    "deepslate_diamond_ore.png",
    "deepslate_emrald_ore.png",
    "deepslate_gold_ore.png",
    "deepslate_iron_ore.png",
    "deepslate_lapis_ore.png",
    "deepslate_redstone_ore.png",
    "diamond_ore.png",
    "emrald_ore.png",
    "iron_ore.png",
    "lapis_ore.png",
    "redstone_ore.png"
]
# n_to_generate = input("How many training images would you like to generate?: ")

# initialize
valueablePath = "textures/valuableBlocks/diamond_ore.png"
path = "textures/commonBlocks/stone.png"

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

# Each corner is a cell of 4 blocks
# Four corners make a chunk of 16 square blocks

valueLocationsTiles = []

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
    valueLocationsTiles.append((valueCorner.shape[1] * 0.75, valueCorner.shape[0] * 0.25))
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

img = cv2.vconcat([
        cv2.hconcat([dullChunk, dullChunk]),
        cv2.hconcat([dullChunk, chunk])
    ])

cv2.imshow("chunk", img)
tiled = np.tile(img, (2,2,1)) # tile the image

outputImage = tiled.copy()

# perspective transform
# specify desired output size

previousWidth = tiled.shape[1]
previousHeight = tiled.shape[0]

scale_percent = 2000
ogWidth = int(backgroundBlock.shape[1] * scale_percent / 100)
ogHeight = int(backgroundBlock.shape[0] * scale_percent / 100)

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

# crop the image randomly
newWidth = outputImage.shape[1]
newHeight = outputImage.shape[0]

cropped_image = outputImage[
                np.random.randint(0,int(newHeight*.5)):int(newHeight*0.6),
                np.random.randint(0,int(newWidth*.5)):int(newWidth*0.6)
                ]

# scale the image
scale_percent = np.random.randint(180, 200)
newWidth = int(newWidth * scale_percent / 100)
newHeight = int(newHeight * scale_percent / 100)
dim = (newWidth, newHeight)
resized = cv2.resize(cropped_image, dim, interpolation=cv2.INTER_AREA)

print("[DEBUG]: IMAGE DISPLAYED. PRESS ANY KEY TO ESCAPE")
for valueLocation in valueLocationsTiles:
    print("[VALUE]: {}".format(valueLocation))
cv2.imshow("tiled", tiled)
# cv2.imshow("cropped_image", cropped_image)
cv2.imshow("resized", resized)

cv2.waitKey(0)
cv2.destroyAllWindows()
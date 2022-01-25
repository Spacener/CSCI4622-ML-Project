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

# randomly select for deepslate or stone
if np.random.random() > 0.5:
    path = "textures/valuableBlocks/deepslate_diamond_ore.png"
    bgColor = (50, 50, 50)
    # randomly select a valuable block to include
    if np.random.random() > 0.2:
else:
    path = "textures/valuableBlocks/diamond_ore.png"
    bgColor = (130, 130, 130)


img = cv2.imread(path)

inputImage = np.tile(img, (10,20,1)) # tile the image

# perspective transform
# specify desired output size

previousWidth = inputImage.shape[1]
previousHeight = inputImage.shape[0]

width = inputImage.shape[1]

perspectiveView = False

scale_percent = 2000
ogWidth = int(img.shape[1] * scale_percent / 100)
ogHeight = int(img.shape[0] * scale_percent / 100)

if np.random.random() >= 0.5: # randomly select perspective view or flat
    leftHeight = inputImage.shape[0]
    rightHeight = inputImage.shape[0]
else: # randomly select left or right perspective
    if np.random.random() >= 0.5:
        leftHeight = 2*inputImage.shape[0]/3
        rightHeight = inputImage.shape[0]*1.5
    else:
        rightHeight = 2*inputImage.shape[0]/3
        leftHeight = inputImage.shape[0]*1.5

    # specify conjugate x,y coordinates (not y,x)
    input = np.float32([[0,0], [previousWidth,0], [previousWidth,previousHeight], [0,previousHeight]])
    output = np.float32([[0,0], [width-1,0], [width-1,rightHeight-1], [0,leftHeight-1]])

    # compute perspective matrix
    matrix = cv2.getPerspectiveTransform(input,output)

    # do perspective transformation setting area outside input to black
    inputImage = cv2.warpPerspective(inputImage,
                                     matrix,
                                     (ogWidth,ogHeight),
                                     cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=bgColor)

# Crop image
# Save the lesser height value
height = rightHeight
if leftHeight < rightHeight: height = leftHeight
if rightHeight < leftHeight: height = rightHeight
else: height = rightHeight

# crop the image randomly
cropped_image = inputImage[np.random.randint(0,int(height-100)):int(height), np.random.randint(0,int(width-100)):int(width)]


# scale the image

scale_percent = np.random.randint(3000, 4000)
newWidth = int(img.shape[1] * scale_percent / 100)
newHeight = int(img.shape[0] * scale_percent / 100)
dim = (newWidth, newHeight)
resized = cv2.resize(cropped_image, dim, interpolation=cv2.INTER_AREA)


print("[DEBUG]: IMAGE DISPLAYED. PRESS ANY KEY TO ESCAPE")
# cv2.imshow("perspective", inputImage)
# cv2.imshow("cropped_image", cropped_image)
cv2.imshow("resized", resized)

cv2.waitKey(0)
cv2.destroyAllWindows()
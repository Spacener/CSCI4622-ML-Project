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

path = "textures/valuableBlocks/diamond_ore.png"
img = cv2.imread(path)

scale_percent = 2000
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

print("[DEBUG]: IMAGE DISPLAYED. PRESS ANY KEY TO ESCAPE")
cv2.imshow("img", resized)

cv2.waitKey(0)
cv2.destroyAllWindows()
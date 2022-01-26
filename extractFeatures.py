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

# Gather images and ore locations
generate_n_images(n=5, showImages=False, saveImages=True)
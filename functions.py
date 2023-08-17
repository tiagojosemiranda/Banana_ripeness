import cv2
import matplotlib.pyplot as plt
import numpy as np

import os

# function to transform image to be used in cnn
# takes an image
# returns the transformed image

def transform(og):
    t = cv2.resize(og, (64, 64))
    t = cv2.cvtColor(t, cv2.COLOR_RGB2HSV) # BGR to RGB 

    hsv_max = [179, 255, 255]

    t = np.divide(t, hsv_max)

    return t

def inverse_transform(img_t):
    hsv_max = [179, 255, 255]
    t = np.multiply(img_t, hsv_max)
    t = t.astype("uint8")
    

    return t


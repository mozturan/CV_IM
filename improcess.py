from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
#reading image
path = "./image processesing/howework3/bloodcells.jpg"
im = np.array(Image.open(path).convert("L"))
def linearTransform(im,a,b): #for low contrast image transform
    Y_linear = (a/255.0)*im+b #changing histogram to between b and (a+b)
    return np.uint8(np.round(Y_linear))
def histogram(image):
    row,col = image.shape
    hist = [0] * 256
    #reading each pixel for histogram
    for i in range(row):
        for j in range(col):
            hist[image[i, j]]+=1
            hist = np.array(hist)
    #returning histogram
    return hist
##### transforming images #####
Y_low = linearTransform(im,a=50,b=50)
Y_dark= linearTransform(im,80,0)
Y_bright = linearTransform(im,125,130)
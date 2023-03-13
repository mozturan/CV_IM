from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
path1 = "./office_2.jpg"
path2 = "./office_6.jpg"
#reads dark and bright versions of a image in 8bit greyscale
imgdark = np.array(Image.open(path1).convert("L"))
imglight = np.array(Image.open(path2).convert("L"))
#thresholding for TH(127)
imD = 255*(imgdark >= 128) #thresholded dark image
imL = 255*(imglight >= 128) #thresholded light image
#thresholding for TH(mean)
mD = np.mean(imgdark) #mean value for dark img
mL = np.mean(imglight) #mean value for light image
imDmean = 255*(imgdark >= mD) #thresholded dark image
imLmean = 255*(imglight >= mL) #thresholded light image
#thresholding for TH(median)
medD = np.median(imgdark) #mean value for dark img
medL = np.median(imglight) #mean value for light image
imDmed = 255*(imgdark >= medD) #thresholded dark image
imLmed = 255*(imglight >= medL) #thresholded light image
#display images
z = np.concatenate((imLmed, imDmed), axis=1)
imgplot = plt.imshow(z, cmap='gray')
# plt.hist(imgdark)
# plt.hist(imglight)
plt.show()

path = "./eiffel2-2.jpg"
im = np.array(Image.open(path).convert("L"))
l=2**8
gama = 0.5
r = im/(l-1)
y = r**gama
y = np.round(y*(l-1)) # 0-255
#display
imgs = np.concatenate((im, y), axis=1)
# imgplot = plt.imshow(y, cmap='gray')
plt.hist(y) #histogram
plt.show()

path = "./ww.jpg"
im = np.array(Image.open(path).convert("L"))
#ww : 78-206
#lineer function
def pixelVal(pix, r1, s1, r2, s2):
    if (0 <= pix and pix <= r1):
        return (s1 / r1)*pix
    elif (r1 < pix and pix <= r2):
        return ((s2 - s1)/(r2 - r1)) * (pix - r1) + s1
    else:
        return ((255 - s2)/(255 - r2)) * (pix - r2) + s2
#params
r1 = 90
s1 = 0
r2 = 180
s2 = 255
pixelVal_vec = np.vectorize(pixelVal)
# Apply contrast stretching.
y_s = pixelVal_vec(im, r1, s1, r2, s2)
#display
y = np.concatenate((im, y_s), axis =1)
zz= plt.imshow(y,cmap='gray')
plt.hist(y_s)
plt.show()



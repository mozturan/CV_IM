from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from improcess import Y_low,Y_dark,Y_bright
pathOriginal = "./image processesing/howework3/bloodcells.jpg"
im = np.array(Image.open(pathOriginal).convert("L"))
def ProbMassFunc(image):
    row,col = image.shape
    hist = [0] * 256
    #reading each pixel for histogram
    for i in range(row):
        for j in range(col):
            hist[image[i, j]]+=1
            # pmf = P(X): probability mass func = histogram(h) / (MxN)
            pmf = np.array(hist)/(row * col)
            hist = np.array(hist)
            #returning pmf and hist
    return pmf, hist

def cumulativeSum(pmf):
    cdf=[]
    cumsum=0
    #computing cumulative sum func
    for i in range(len(pmf)):
        cumsum += pmf[i]
        cdf.append(cumsum)
        #returnin cumulative sum of probability mass func for every pixel value in image
    return np.array(cdf)
def histogramEqualization(imag):
    pmf, hist_X = ProbMassFunc(imag) #gets pmf and hist of input imageX matrix
    cdf = cumulativeSum(pmf) #cumulative dist-func between 0-1

    tf = np.round(255 * cdf) #campute transfer func values between 0-
    255
    row, col = imag.shape
    Y = np.zeros_like(imag) #creating an empty output Y matrix
    # applying transfer func pixel by pixel
    for i in range(row):
        for j in range(col):
            Y[i, j] = tf[imag[i, j]]
            pmf_Y, hist_Y = ProbMassFunc(Y)
            #return transformed image, original and new istogram,
        # and transform function
    return Y , hist_Y
    
Y, hist_Y = histogramEqualization(im)
Y_dark_eq, hist_Y_dark_eq = histogramEqualization(Y_dark)
Y_bright_eq, hist_Y_bright_eq = histogramEqualization(Y_bright)
Y_low_eq, hist_Y_low_eq = histogramEqualization(Y_low)
figure, axis = plt.subplots(nrows=4,ncols=1)
axis[0].plot(hist_Y)
axis[1].plot(hist_Y_dark_eq)
axis[2].plot(hist_Y_bright_eq)
axis[3].plot(hist_Y_low_eq)
plt.show()
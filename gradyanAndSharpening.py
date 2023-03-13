from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
#reading image
path = "./homework6/board.jpg"
im = np.array(Image.open(path).convert("L"))

def findSum(arr):
    #gets sum of inputs
    values = arr.flatten()
    summ=0
    for i in range(len(values)):
        summ +=values[i]
    return summ

def kernels(all=True, a =1):
    gx = np.array([[ 0, 0, 0],
    [-1, 0, 1],
    [ 0, 0, 0]])

    gy = gx.copy()
    gy = gy.T # gy is transpose of gx
    # Laplacian matrix for 90 degree
    l = np.array([[ 0, 1, 0],
    [ 1,-4, 1],
    [ 0, 1, 0]])
    # Sharpeningn matrix for 90 degree
    s = np.array([[ 0,-1*a, 0],
    [-1*a, 1+4*a,-1*a],
    [ 0,-1*a, 0]])
    # Sharpening matrix for 45 degree
    eightdirection = np.array([[-1,-1,-1],

    [-1, 9,-1],
    [-1,-1,-1]])

    if (all):
        return gx,gy,l, s, eightdirection
    return gx, gy
def convo2d(image, filter, threshold = False):
#setting parameters

    image_row, image_col = image.shape
    filter_row, filter_col = filter.shape
    #applies padding(greater than size of original image)
    # and return output image(size of original image)
    padded_image, output= padding(image, filter)
    #applies filter pixelwise
    for row in range(image_row):
        for col in range(image_col):
            #applies filters
            output[row, col]=findSum(filter * padded_image[row:row +

            filter_row, col:col + filter_col])
        #normalization selection
    if threshold:
        #thresholds the values greater than 255 and lower than 0
        output_image = output
        output_image = np.round(output_image)
        mask = output_image > 255
        output_image[mask] = 255
        mask = output_image < 0
        output_image[mask] = 0
    else:
        #normalization: linear transform just in case
        output_image = ((output - output.min()) / (output.max() -
        output.min())) # values between 0-1
        output_image = np.round(output_image*255)
        #return normalized (0-255) image
    return output_image

def padding(image, filter):
    #setting parameters
    image_row, image_col = image.shape
    filter_row, filter_col = filter.shape
    #generating new output image
    output = np.zeros((image_row,image_col))
    #setting pad size
    pad_height = int((filter_row - 1) / 2)
    pad_width = int((filter_col - 1) / 2)
    #applies padding to image

    padded_image = np.zeros((image_row + (2 * pad_height), image_col +
    (2 * pad_width)))
    padded_image[pad_height:padded_image.shape[0] - pad_height,
    pad_width:padded_image.shape[1] - pad_width] = image
    return padded_image, output
    
gx, gy, l, s5, eight= kernels(True,a=5)
outGy = convo2d(image=im,filter=gy)
outGx = convo2d(image=im,filter=gx)
gradient = np.sqrt((outGx**2) + (outGy**2))
outLaplacian =convo2d(im,l, threshold=True)
outSharpenedk5 =convo2d(im,s5, threshold=True)
eightway = convo2d(im, eight, threshold=True)
gx, gy, l, s1 , e= kernels(True)
outSharpenedk1 = convo2d(im,s1, threshold=True)
figure, axis = plt.subplots(nrows=2,ncols=2)
axis[0,0].imshow(im, cmap = "gray")
axis[0,0].set_title('orijinal görüntü')
axis[0,1].imshow(outLaplacian, cmap = "gray")
axis[0,1].set_title('Laplacien')
axis[1,0].imshow(outSharpenedk5, cmap = "gray")
axis[1,0].set_title('K=5 katsayısı ile keskinleştirme')
axis[1,1].imshow(outSharpenedk1, cmap = "gray")
axis[1,1].set_title('K=1 katsayısı ile keskinleştirme')
plt.show()
figure, axis = plt.subplots(nrows=2,ncols=2)
axis[0,0].imshow(im, cmap = "gray")
axis[0,0].set_title('orijinal görüntü')
axis[0,1].imshow(gradient, cmap = "gray")
axis[0,1].set_title('gradient genliği')
axis[1,0].imshow(outGx, cmap = "gray")
axis[1,0].set_title('gx')
axis[1,1].imshow(outGy, cmap = "gray")
axis[1,1].set_title('gy')
plt.show()
figure, axis = plt.subplots(nrows=1,ncols=3)
axis[0].imshow(im, cmap = "gray")
axis[0].set_title('orijinal görüntü')
axis[1].imshow(outSharpenedk1, cmap = "gray")
axis[1].set_title('k=1 katsayısı ile keskinleştirilmiş')
axis[2].imshow(eightway, cmap = "gray")
axis[2].set_title('8yön (45 derece) matrisi ile keskinleştirilmiş')
plt.show()
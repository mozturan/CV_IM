from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
#reading image
path = "./image processesing/homework4/people.gif"
im = np.array(Image.open(path).convert("L"))

def kernel(row, col): # shape as (x,y)
#creating a filter out of ones  
    return np.ones((row,col), np.float32)
def variance(block):
    #  gets avarega of the input block(size of filter)
    avarage = mean(block)
    block = block.flatten()
    variance = 0
#computing variance: (pixel-mean)^2
    for i in range(len(block)):
        variance += (block[i]-avarage)**2
    #returns variance
    return variance / (len(block) - 1)
def mean(block):
    #returning mean value of an input block(size of filter)
    block = block.flatten()
    blockSum = 0
    #sum of all values in input block
    for i in range(len(block)):
        blockSum += block[i]
    #returns mean on input block
    return blockSum / len(block)

def blockProcessing(image, filter):
    #setting parameters
    image_row, image_col = image.shape
    filter_row, filter_col = filter.shape

#applies padding(greater than size of original image)
# and return output image(size of original image)
    padded_image, output= padding(image, filter)
#applies filter pixelwise
    for row in range(image_row):
        for col in range(image_col):
        #this part is switchable( between mean and variance filters)
        #applies mean filter
            output[row, col]=mean(filter * padded_image[row:row +

            filter_row, col:col + filter_col])
            #applies variance filter
            output[row, col]=variance(filter * padded_image[row:row +

            filter_row, col:col + filter_col])
            #normalization: linear transform just in case
    output_image = ((output - output.min()) / (output.max() -
            output.min())) # values between 0-1
    #return normalized (0-255) image
    return np.round(output_image*255)

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
#outputs - filters 3x3, 9x9, 15x15
output3 = blockProcessing(im,kernel(3,3))

output9 = blockProcessing(im,kernel(9,9))
output15 = blockProcessing(im,kernel(15,15))
#thresholding the outputs of variance filter
im3th = 255*(output3 >= np.mean(output3.flatten())) #thresholded
im9th = 255*(output9 >= np.mean(output9.flatten())) #thresholded
im15th = 255*(output15 >= np.mean(output15.flatten())) #thresholded
#plotting the outputs
figure, axis = plt.subplots(nrows=2,ncols=2)
axis[0, 0].imshow(im, cmap = "gray")
axis[0, 0].set_title('original image')
axis[0, 1].imshow(output3, cmap = "gray")
axis[0, 1].set_title('3x3 Filter')
axis[1, 0].imshow(output9, cmap = "gray")
axis[1, 0].set_title('9x9 Filter')
axis[1, 1].imshow(output15, cmap = "gray")
axis[1, 1].set_title('15x15 Filter')
plt.show()
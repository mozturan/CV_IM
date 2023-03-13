import numpy as np #for matrix etc..
import matplotlib.pyplot as plt #to display images and histograms
import cv2 #to read image
#harris corner main func
def harrisCorner(inputArr):
inputt = inputArr
corners = []
#bluring for noise reduction
# inputArr = gaussianFilter(inputArr,3)
#get derivatives using sobel kernels
sx,sy = gradients(inputArr)
B1 = np.power(sy,2)
A1 = np.power(sx,2)
C1 = np.multiply(sx,sy)
#gaussian filter kernel = 3*2*sigma+1 = 7==> 7x7
A = gaussianFilter(A1, sigma=1)
B = gaussianFilter(B1, sigma=1)
C = gaussianFilter(C1, sigma=1)
k= 0.05
corner_det = np.multiply(A,B)-np.power(C,2)
corner_trc = A+B
#cornerness matrix
cornerness = corner_det-k*np.power(corner_trc,2)
c_max = cornerness.max()
c_thresh = 0.001*c_max
#corners for blocklike
II_cornerness = cornerness>c_thresh
#threshold and get corner coordinates with cornerness value
for row in range(cornerness.shape[0]):
for col in range(cornerness.shape[1]):
if cornerness[row,col]>c_thresh:
c = cornerness[row,col]
corners.append([c, row, col,0])

cornerList = suppressCorners(corners,5)

# to display corners on image
im_corner = np.zeros((inputArr.shape[0], inputArr.shape[1],3),
dtype='uint8')
I_cornerness = np.zeros((inputArr.shape[0], inputArr.shape[1]),
dtype='uint8')
for k in cornerList:
I_cornerness[k[1], k[2]] = 1
mask = I_cornerness == 1
im_corner[:,:,0]=inputt
im_corner[:,:,1]=inputt
im_corner[:,:,2]=inputt
im_corner[mask]=[255,0,0]
fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8,
8),

sharex=True, sharey=True)

ax1.imshow(im_corner, cmap=plt.cm.gray)
ax1.axis('off')
ax1.set_title('Image with corner coordinates', fontsize=20)
ax2.imshow(I_cornerness, cmap=plt.cm.gray)
ax2.axis('off')
ax2.set_title('corners after suppression', fontsize=20)
ax3.imshow(II_cornerness, cmap=plt.cm.gray)
ax3.axis('off')
ax3.set_title('corners before supression', fontsize=20)
fig.tight_layout()
plt.show()
#return [cornerness value, row, col, 0]
return cornerList
#non-maximum suppression for harris corner
def suppressCorners(cornerList, windowsize = 7):
#sorting listt
cornerList.sort(reverse=True)
#marking unwanted neighbors based on w Size
for i in cornerList:
index = []
if i[3] != 1:

for j in cornerList:
if j[3] != 1:
dist = np.sqrt((j[1] - i[1])**2 + (j[2] - i[2])**2)
if (dist <= windowsize and dist > 0):
j[3] = 1

#filtering out neghbors
final = filter(lambda x: x[3] == 0, cornerList)
return list(final)
#gaussian filtering
def gaussianFilter(im, sigma = 1):
halfwid = 3*sigma
num = 2*halfwid+1
#since sigma is 1 default, w=7
xx = np.linspace(-halfwid,halfwid,num)
yy = np.linspace(-halfwid,halfwid,num)
xx ,yy =np.meshgrid(xx,yy)
e = -1/(np.pi * (sigma**4))
d = 1-( ((xx**2) + (yy**2)) / (2*(sigma**2)))
a = 1/(2*np.pi*(sigma**2))
exp= np.e**(- ((xx ** 2) + (yy ** 2))/(2*(sigma**2)))
g= a*exp
l= e*d*exp
# laplace = convo(im, l)
gauss = convo(im, g)
return gauss
#compute gradients sx,sy
def gradients(im):
##Sobel operator kernels.
kernel_x = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
#getting Ix and Iy by convo of kernels
sx = convo(im,kernel_x)
sy = convo(im,kernel_y)
return sx,sy
#convolution func
def convo(image, filter):

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
output[row, col]=np.sum(filter * padded_image[row:row +

filter_row, col:col + filter_col])
return output
#padding for convolution
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
#computes gradient magnitude
def gradient_magnitude(sx, sy):
#computes gradient magnitude
horizontal_gradient_square = np.power(sx, 2)
vertical_gradient_square = np.power(sy, 2)
sum_squares = horizontal_gradient_square + vertical_gradient_square
gradMagnitude = np.sqrt(sum_squares)
return gradMagnitude

#computes gradient direction
def gradient_direction(sx, sy):
#to avoid undefined values we add a small value to sx
grad_direction = np.arctan(sy/(sx+0.00000001))
grad_direction = np.rad2deg(grad_direction)
grad_direction = grad_direction%180
return grad_direction
#returns gradM and gradD at once
def getGradients(img):
sx,sy = gradients(img)
gradientMagnitude = gradient_magnitude(sx,sy)
gradientDirection = gradient_direction(sx,sy)
return gradientMagnitude,gradientDirection
#compute HoG of a cell (8x8)
def cellHOG(cell, bins):
#cell = np array, bins =angles
#creating a hog box 1x8
hogList = np.zeros(shape=(bins.size))
#getting gradients of cell
cellMagnitude, cellDirection = getGradients(cell)
#adding gradient magnitudes with interpolation
for row in range(cell.shape[0]):
for col in range(cell.shape[1]):
currentMagnitude = cellMagnitude[row,col]
currentDirection = cellDirection[row,col]
if currentDirection > 157.5:
xx = np.abs(currentDirection-157.5)
yy = np.abs(currentDirection-180)
hogList[7] = (currentDirection*yy)/(xx+yy)
hogList[0] = (currentDirection*yy)/(xx+yy)
else:
for k in range(len(bins)):
if bins[k] == currentDirection:
hogList[k] += currentMagnitude
elif currentDirection>bins[k] and

currentDirection<bins[k+1]:

a = np.abs(currentDirection-bins[k])
b = np.abs(currentDirection-bins[k+1])
hogList[k] = (currentDirection*b)/(a+b)
hogList[k+1] = (currentDirection*a)/ (a+b)

hogsum = np.sum(hogList)
hogList = hogList / hogsum
#plotting the hog
# plt.bar(x=np.arange(8), height=hogList, align="center", width=0.8)
# plt.show()
return hogList
#gets HoG feature vector of a corner in a block
def HoGVector(image, corner):
#gets hog vektor of a corner in corner list
x= corner[1]
y= corner[2]
#if corner is too close to image edges ignore that
if (x-16) < 0 or (x+16) > image.shape[0]:
return None
if (y-16) < 0 or (y+16) > image.shape[1]:
return None
xx= [0,8,16,24,32]
yy= [0,8,16,24,32]
#32x32 shape block around corner coordinates
block = image[x-16:x+16,y-16:y+16]
#angles - directions
hist_bins = np.array([0,22.5,45,67.5,90,112.5,135,157.5])
#splitting block into cells
cells = []
for i in range(len(xx)-1):
for j in range(len(yy)-1):
cell = block[xx[i]:xx[i+1], yy[j]:yy[j+1]]
cells.append(cell)
HoG4Block = []
HoG4Block =np.array(HoG4Block)
#gets hogs for every cell
for cell in cells:
hogOfCell = cellHOG(cell, hist_bins)
HoG4Block = np.concatenate((HoG4Block, hogOfCell))
return HoG4Block
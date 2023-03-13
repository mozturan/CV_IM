from PIL import Image #to read image
import numpy as np #to floor the values
from matplotlib import pyplot as plt #to show image
pathOfboys = './DIP/boys8bit.jpg' #8bit greyscale image
im = np.array(Image.open(pathOfboys)) #reads image
img = []
max8bit = 256 #Vmax for 8 bit (2^k)
min8bit = 0 # Vmin for 8 bit
for i in range(8):
    k=i+1
    y= np.floor(((im-min8bit)/(max8bit-min8bit))*(2**k)) #X -> Y (0-2^k)
    q= (max8bit-min8bit)/(2**k) #quantization range
    x = q*(y+0.5) #Y -> X (0-255)
    img.append(x)
a= np.concatenate((img[0:2]), axis=1) # 1 and 2 bit
b= np.concatenate((img[2:4]), axis=1) # 3 and 4 bit
c= np.concatenate((img[4:6]), axis=1) # 5 and 6 bit
d= np.concatenate((img[6:]), axis=1) # 7 and 8 bit
img = np.concatenate((d,c,b,a), axis=0)
plt.imshow((img), cmap='gray') #display new images
plt.show() #show images

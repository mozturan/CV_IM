from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
#IMAGE READING
path = "./image processesing/homework5/2.gif"
im =np.array(Image.open(path).convert('L'))
row,col =im.shape #512x512
def getNoisyImages(image, k, std):
    noisedImages = {} #creating a dictionary for noised images
    #adding noises to image
    for i in range(k):
    #generating gaussian noise
        noise = np.random.normal(0,std, np.shape(im))
        #gets Noisy Image values between (-std,255+std) dtype:float
        noisedImages[i] = image+noise
        #Avoiding MIN-MAX Normalisation,TH Normalisation of Noisy Image
        noisedImages[i] = np.round(noisedImages[i])
        mask = noisedImages[i] > 255
        noisedImages[i][mask] = 255
        mask = noisedImages[i] < 0
        noisedImages[i][mask] = 0
        #returns Noisy Images -integer values between 0-255
    return noisedImages
def average(image,k,std):
    row,col =image.shape #512x512
    #generate noised images
    images = getNoisyImages(image=image,k=k,std=std)
    #creating an outout image
    output = np.zeros((row,col), dtype=float)
    for i in range(len(images)):
        output +=images[i]
        output = output / len(images)
        output_image = ((output - output.min()) / (output.max() -
        output.min())) # values between 0-1
    #return values 0-255
    return np.round(output_image*255)

#PLOTTING TO DISPLAY

var = 30**2
std = np.sqrt(var)
out100 = np.uint8(average(im,100,std))
out50 = np.uint8(average(im,50,std))
out10 = np.uint8(average(im,10,std))
out5 = np.uint8(average(im,5,std))
NoisyImage = (getNoisyImages(im,1,std))

figure, axis = plt.subplots(nrows=1,ncols=2)
axis[0].imshow(im, cmap = "gray")
axis[0].set_title("orijinal görüntü")
axis[1].imshow(NoisyImage[0], cmap = "gray")
axis[1].set_title(f"Gürültü (std ={std}) eklenmiş görüntü")
plt.show()
figure, axis = plt.subplots(nrows=1,ncols=2)
axis[0].imshow(out5, cmap = "gray")
axis[0].set_title(f'std = {std} gürültülü ve k = 5 ile\n ortalamasıalınmış görüntü')
axis[1].imshow(out10, cmap = "gray")
axis[1].set_title(f'std = {std} ve gürültülü k = 10 ile\n ortalamasıalınmış görüntü')
plt.show()
figure, axis = plt.subplots(nrows=1,ncols=2)
axis[0].imshow(out50, cmap = "gray")
axis[0].set_title(f'std = {std} gürültülü ve k = 50 ile\n ortalamasıalınmış görüntü')
axis[1].imshow(out100, cmap = "gray")
axis[1].set_title(f'std = {std} gürültülü ve k = 100 ile\n ortalamasıalınmış görüntü')
plt.show()
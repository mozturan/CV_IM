from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
image = np.array(Image.open( "./homework12/10.gif").convert("L"))
im_fft = np.fft.fft2(image)
im_fft_shift = np.fft.fftshift(im_fft)
im_fft_magnitude = 20*np.log(abs(im_fft_shift))

def noise_add(image):
    row,col = image.shape
    noisy = image.copy()
    for x in range(row):
        noisy[x] = image[x] + (30*np.cos(2*np.pi*(20/row)*(x)))
    return noisy
def linear_notch_filter(image):
    row, col = image.shape
    filter = np.ones((row,col),np.uint8)
    filter[:,col//2] = 0
    filter[row//2 -10:row//2 +10, :] = 1
    return filter
def notch_reject(image):
    c1 = 276,256
    c2 = 236,256
    rows, cols = image.shape
    filter = np.ones((rows, cols),np.uint8)
    r =20
    x, y = np.ogrid[:rows, :cols]
    d1 = (x - c1[0]) ** 2 + (y - c1[1]) ** 2 <= r
    filter[d1] = 0
    x, y = np.ogrid[:rows, :cols]
    d2 = (x - c2[0]) ** 2 + (y - c2[1]) ** 2 <= r
    filter[d2] = 0
    return filter

def notch_filter(image):
    rows,cols = image.shape
    c_row, c_col = rows // 2, cols // 2
    filter = np.ones((rows, cols),np.uint8)
    r_1 = 20
    r_2 = 15
    center = [c_row, c_col]
    x, y = np.ogrid[:rows, :cols]
    filter_func = np.logical_and(((x - center[0]) ** 2 + (y - center[1])
    ** 2 >= r_2**2),

    ((x - center[0]) ** 2 + (y - center[1])

    ** 2 <= r_1**2))
    filter[filter_func] = 0
    return filter
def threshold(imm):
    imm= np.round(abs(imm))
    mask = imm >255
    imm[mask] =255
    mask = imm < 0
    imm[mask] = 0
    return imm
noisy = noise_add(image)
noisy_fft = np.fft.fft2(noisy)
noisy_fft_shift = np.fft.fftshift(noisy_fft)
noisy_magnitude =20*np.log(abs(noisy_fft_shift))
filter = notch_filter(image)
fshift = noisy_fft_shift * filter
f_ifft = np.fft.ifftshift(fshift)
imm= np.fft.ifft2(f_ifft)
imm = threshold(imm)
fig, ax = plt.subplots(nrows=1,ncols=2)
ax[0].imshow(image, cmap = "gray")
ax[0].set_title("orijinal görüntü")
ax[1].imshow(im_fft_magnitude,cmap ="gray")
ax[1].set_title("fft'si")
plt.show()
fig, ax = plt.subplots(nrows=1,ncols=2)
ax[0].imshow(noisy, cmap = "gray")

ax[0].set_title("periyodik gürültü eklenmiş görüntü")
ax[1].imshow(noisy_magnitude,cmap ="gray")
ax[1].set_title("fft'si")
plt.show()
fig, ax = plt.subplots(nrows=1,ncols=2)
ax[0].imshow(filter, cmap = "gray")
ax[0].set_title("notch filtresi")
ax[1].imshow(imm,cmap ="gray")
ax[1].set_title("filtre sonucu")
plt.show()
import numpy as np
from numpy.fft import fft2, ifft2
from PIL import Image
import matplotlib.pyplot as plt
image = np.array(Image.open("./homework8/10.gif").convert("L"))
image = np.double(image)
im = image.copy()
def normalization(matrix, log = True, MinMax=True):
    out = matrix.copy()
    out = abs(out)
    if log:
        out = 20*np.log(out)
    if MinMax:
        out = (out - out.min()) / (out.max() - out.min())
        out = np.round(out*255)
    return out

def shift(matrix):
    rows, cols = matrix.shape
    #create shift matrix
    shiftMatrix = np.zeros((rows,cols))
    for row in range(rows):
        for col in range(cols):
            shiftMatrix[row,col] = (-1)**(row+col)
    sMatrix = matrix*shiftMatrix
    return sMatrix

def fastFT(matrix):
    out = fft2(matrix)
    phase = np.angle(out)
    return out, phase
# im = shift(im)
imageFFT, phase = fastFT(im)
imageFFT = normalization(imageFFT, log=True, MinMax=True)
fig, axs = plt.subplots(nrows=1, ncols=2)
axs[0].imshow(im, cmap="gray")
axs[0].set_title("orijinal görüntü")
axs[1].imshow(imageFFT, cmap="gray")

axs[1].set_title("görüntünün fft'si")
plt.show()
plt.imshow(phase, cmap="gray")
plt.title("-1^(x+y) ile çarpılmış görüntünün faz spektrumu")
plt.show()
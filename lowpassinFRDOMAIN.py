import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def ideal_lowpass_filter(shape, ratio):
    "generating ideal low pass filter"
    H = np.zeros((shape))
    center = np.array(H.shape)/2.0
    d_zero= (min(shape[0], shape[1])/2)*ratio
    for row in range(shape[0]):
        for col in range(shape[1]):
            d = np.sqrt((center[0]-row)**2 + (center[1]-col)**2)
            if d > d_zero:
                H[row,col] = 0
            else:
                H[row,col]= 1

    return H

def shift(matrix):
    rows, cols = matrix.shape
    #create shift matrix
    shiftMatrix = np.zeros((rows,cols))
    for row in range(rows):
        for col in range(cols):
            shiftMatrix[row,col] = (-1)**(row+col)
    sMatrix = matrix*shiftMatrix
    return sMatrix

def normalization(matrix, Threshold=True, MinMax=True):
    out = matrix.copy()
    out = abs(out)
    if Threshold:
        output_image = np.round(out)
        mask = output_image > 255
        output_image[mask] = 255
        mask = output_image < 0
        output_image[mask] = 0
        out = output_image

    if MinMax:
        out = (out - out.min()) / (out.max() - out.min())
        out = np.round(out*255)
    return output_image

def fft(image, ratio):
    shape = image.shape
    H = ideal_lowpass_filter(shape, ratio)
    image_shift = shift(image)
    image_fft_shift = np.fft.fft2(image_shift)
    #filtering frequency domain
    filtered_image = image_fft_shift * H
    #gets ifft and normalize with th and minmax
    filtered_image = np.fft.ifft2(filtered_image)
    filtered_image = normalization(filtered_image)
    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].imshow((filtered_image), cmap="gray")
    ax[0].set_title(f"Frekans domainde \nD_0 = {ratio} * M/2 değeri ile filtrelenmiş görüntü")
    ax[1].imshow(H, cmap="gray")
    ax[1].set_title(f"D_0 = {ratio} * M/2")
    plt.show()
def main():
    image = np.array(Image.open("./homework9/49.gif").convert("L"))
    ratio_values = [0.05, 0.10, 0.20, 0.50, 0.90]
    for value in ratio_values:
        fft(image, value)
if __name__ == '__main__':
    main()
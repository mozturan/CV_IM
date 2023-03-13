import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
def butterwoth_highpass_filter(shape, ratio, n, c1, c2):
    H = np.zeros((shape))
    center = np.array(H.shape)/2.0
    d_zero= min(shape[0], shape[1])*ratio/2
    for row in range(shape[0]):
        for col in range(shape[1]):
            d = np.sqrt((center[0]-row)**2 + (center[1]-col)**2)
            H[row,col] = 1/(1 + (d/d_zero)**(2*n))
            H = 1 - H
            H = c1 + (c2*H)
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
def fft(image, ratio, n, c1, c2):
    shape = image.shape
    H = butterwoth_highpass_filter(shape, ratio, n, c1, c2)
    image_shift = shift(image)
    image_fft_shift = np.fft.fft2(image_shift)
    filtered_image = image_fft_shift * H
    filtered_image = np.fft.ifft2(filtered_image)
    filtered_image = normalization(filtered_image)
    plt.imshow(filtered_image, cmap="gray")
    plt.title("Frekans domainde Butterworth HPF ile filtrelenmiş (keskinleştirilmiş) görüntü\n"
    + f"D0 = {ratio} * M/2 ve n = {n}\n" + f"c1 = {c1} ve c2 = {c2}")
    # fig, ax = plt.subplots(nrows=1, ncols=1)
    # ax[0].imshow((filtered_image), cmap="gray")
    # ax[0].set_title(f"Frekans domainde \nD_0 = {ratio} * M/2 değerinde\n"
    # + " Butterworth HPF ile filtrelenmiş (keskinleştirilmiş) görüntü")
    # ax[1].imshow(H, cmap="gray")
    # ax[1].set_title(f"D_0 = {ratio} * M/2 vs n = {n}\nc1 = {c1}, c2 =    {c2}")
    plt.show()
def main():
    image = np.array(Image.open("./homework10/49.gif").convert("L"))
    c1 = [0.5,1,1.2]
    c2 = [0, 0.5, 1, 1.5]
    for x in c1:
        for y in c2:
            fft(image, 0.2, 2, x, y)

if __name__ == '__main__':
    main()
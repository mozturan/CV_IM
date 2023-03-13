import numpy as np
from matplotlib import pyplot as plt
from math import log10, sqrt
from noiseandhealing import im,out10,out50,out5,out100,std
def findSum(arr):
#gets sum of inputs
    values = arr.flatten()
    summ=0
    for i in range(len(values)):
        summ +=values[i]
    return summ
def MAEmse(i0,iy):
    #calculating MAE and MSE
    err=iy-i0 #error
    sq = err**2 #error^2
    ab = np.absolute(err) #|error|
    mseSum = findSum(sq)
    maeSum = findSum(ab)
    MSE = mseSum / (err.shape[0] * err.shape[1])
    MAE = maeSum / (err.shape[0] * err.shape[1])
    return MAE,MSE
def PSNR(i0, iy):
    #get mse
    mae, mse = MAEmse(i0,iy)
    #P-signal
    max_pixel = 255.0
    #find signal to noise ratio (SNR)
    SNR = (max_pixel**2) / mse
    psnr = 10*log10(SNR)
    return psnr
mae5,x=MAEmse(im,out5)
mae10,y=MAEmse(im,out10)
mae50,z =MAEmse(im,out50)
mae100,r =MAEmse(im,out100)


print("\n")
print(f"std={std} gürültülü ve k=5 değerlinde ortalama alma için MAE :" + str(mae5))
print(f"std={std} gürültülü ve k=10 değerlinde ortalama alma için MAE : " + str(mae10))
print(f"std={std} gürültülü ve k=50 değerlinde ortalama alma için MAE : " + str(mae50))
print(f"std={std} gürültülü ve k=100 değerlinde ortalama alma için MAE :" + str(mae100))
print("\n")
print(f"std={std} gürültülü ve k=5 değerlinde ortalama alma için PSNR :" + str(PSNR(im,out5)))
print(f"std={std} gürültülü ve k=10 değerlinde ortalama alma için PSNR :" + str(PSNR(im,out10)))
print(f"std={std} gürültülü ve k=50 değerlinde ortalama alma için PSNR :" + str(PSNR(im,out50)))
print(f"std={std} gürültülü ve k=100 değerlinde ortalama alma içinPSNR : " + str(PSNR(im,out100)))
print("\n")
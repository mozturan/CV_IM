from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft
def generateSinWave(frequency, numberOfPeriods, samplesPerPeriod,
theta):
    #calculate the time
    duration = numberOfPeriods/frequency
    samples = numberOfPeriods*samplesPerPeriod
    #creating samples with given values
    x = np.linspace(0, duration, samples, endpoint=False)
    y = np.sin((2 * np.pi) * frequency * x + theta)
    return x, y
def generateSquareWave(frequency, numberOfPeriods, samplesPerPeriod,
theta):
    #calculate the time
    duration = numberOfPeriods/frequency
    samples = numberOfPeriods*samplesPerPeriod
    #creating samples with given values
    x = np.linspace(0, duration, samples, endpoint=False)
    y= signal.square(2 * np.pi * frequency* x + theta)
    return x, y
x, y = generateSinWave(frequency=4, numberOfPeriods=4,
samplesPerPeriod=90, theta=0)
x, y = generateSquareWave(frequency=4, numberOfPeriods=4,
samplesPerPeriod=90, theta=0)
#find fft and normalize
yfft = fft(y)
#find magnitude and normalize
yfftMagnitude = abs(yfft)/len(x)
xfft = np.arange(len(x))
#plotting
plt.plot(x, y,"g")
plt.title("4Hz ve 4 salınımlı her periyotta 90 örnekli kare dalga")
plt.ylabel('Genlik')
plt.xlabel('zaman - T (sn)')
plt.show()
plt.plot(xfft, yfftMagnitude, "r")
plt.title("Frekans domainde genlik")
plt.xlabel('x(n) - fourier katsayıları')
plt.ylabel('FFT Genliği |X(freq)|')
plt.show()
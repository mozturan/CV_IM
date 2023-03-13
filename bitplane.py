from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
# path = "./computer vision/images/png/coins.png"
path = "./image processesing/howework3/money.jpg"
im = np.array(Image.open(path).convert("L"))
bit_planes ={}#creating bit planes dictionary
#getting bit planes
print(type(bit_planes))
for i in range(1,9):
    bit_planes[i] = np.mod(np.floor(im/(2**(i-1))), 2)
#plotting to display
figure, axis = plt.subplots(nrows=2,ncols=4)

for k in range(4):
    axis[0, k].imshow(bit_planes[k+1], cmap = "gray")
    axis[0, k].set_title(f'bitplane {k+1}')
for t in range(4):
    axis[1, t].imshow(bit_planes[t+5], cmap = "gray")
    axis[1, t].set_title(f'bit plane {t+5}')
plt.show()
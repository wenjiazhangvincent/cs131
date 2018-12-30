import matplotlib.pyplot as plt
import math

import os
import numpy as np
from PIL import Image
from skimage import color, io

image_path = '%s/image1.jpg' %(os.path.dirname(__file__))

mat = np.array([[1,0,0],[0,1,0],[0,0,1]], dtype=np.uint8)    
image1 = io.imread(image_path)

print (image1.shape)
plt.subplot(2, 2, 1)
plt.imshow(image1)

plt.subplot(2, 2, 2)
image2 = np.array(image1)
image2 *= mat[0]
plt.imshow(image2)

plt.subplot(2, 2, 3)
image3 = np.array(image1)
image3 *= mat[1]
plt.imshow(image3)

plt.subplot(2, 2, 4)
image4 = np.array(image1)
image4 *= mat[2]
plt.imshow(image4)

plt.show()
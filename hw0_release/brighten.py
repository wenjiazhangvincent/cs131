import matplotlib.pyplot as plt
import math

import os
import numpy as np
from PIL import Image
from skimage import color, io

image_path = '%s/image1.jpg' %(os.path.dirname(__file__))

    
image1 = io.imread(image_path)
plt.figure(1)
plt.axis('off')
plt.subplot(2,2,1, title='original')
plt.imshow(image1)
 
#RGB
image2 = np.array(image1)
image2 = image2 // 2
plt.subplot(2,2,2, title='RGB')
plt.imshow(image2)
 
#LAB
image3 = np.array(image1)
tmp = color.rgb2hsv(image3)
tmp[:, :, 0] *= 0.5
image3 = color.hsv2rgb(tmp)
plt.subplot(2,2,3, title='LAB')
plt.imshow(image3)
 
#HSV
image4 = np.array(image1)
tmp = color.rgb2hsv(image4)
tmp[:, :, 2] *= 0.5
image4 = color.hsv2rgb(tmp)
plt.subplot(2,2,4, title='HSV')
plt.imshow(image4)


plt.show()
import math

import numpy as np
from PIL import Image
from skimage import color, io
import linalg

def load(image_path):
    """Loads an image from a file path.

    HINT: Look up `skimage.io.imread()` function.

    Args:
        image_path: file path to the image.

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """
    out = None

    ### YOUR CODE HERE
    # Use skimage io.imread
    out = io.imread(image_path)
    ### END YOUR CODE

    # Let's convert the image to be between the correct range.
    out = out.astype(np.float64) / 255
    return out


def dim_image(image):
    """Change the value of every pixel by following

                        x_n = 0.5*x_p^2

    where x_n is the new value and x_p is the original value.

    Args:
        image: numpy array of shape(image_height, image_width, 3).

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """

    out = None

    ### YOUR CODE HERE
#     out = []
#     for row in image:
#         out.append([lambda x:int(0.5*x**0.5) for x in row])
    out = 0.5 * np.square(image)
    ### END YOUR CODE

    return out


def convert_to_grey_scale(image):
    """Change image to gray scale.

    HINT: Look at `skimage.color` library to see if there is a function
    there you can use.

    Args:
        image: numpy array of shape(image_height, image_width, 3).

    Returns:
        out: numpy array of shape(image_height, image_width).
    """
    out = None

    ### YOUR CODE HERE
    out = color.rgb2gray(image)
#     out = np.sum(image, axis=2)/3
    ### END YOUR CODE

    return out


def rgb_exclusion(image, channel):
    """Return image **excluding** the rgb channel specified

    Args:
        image: numpy array of shape(image_height, image_width, 3).
        channel: str specifying the channel. Can be either "R", "G" or "B".

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """

    out = None

    ### YOUR CODE HERE
    out = np.array(image, dtype=np.float32)
    dic = {'r':0, 'g':1, 'b':2}
    mat = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=np.float32)
    out = out * mat[dic[channel.lower()]]
    ### END YOUR CODE

    return out


def lab_decomposition(image, channel):
    """Decomposes the image into LAB and only returns the channel specified.

    Args:
        image: numpy array of shape(image_height, image_width, 3).
        channel: str specifying the channel. Can be either "L", "A" or "B".

    Returns:
        out: numpy array of shape(image_height, image_width).
    """

    lab = color.rgb2lab(image)
    out = None

    ### YOUR CODE HERE
#     tmp = {'l':[1,2], 'a':[0,2], 'b':[0,1]}
#     out = np.array(lab)
#     for i in range(2):
#         out[:, :, tmp[channel.lower()][i]] = 0
        
    dic = {'l':0, 'a':1, 'b':2}
    mat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    out = lab * mat[dic[channel.lower()]]
    
#     color_dict = {'l': 0, 'a': 1, 'b': 2}
#     lab = (lab + np.abs(np.min(lab)))
#     lab = lab / np.max(lab)
#     lab[:,:,color_dict[channel.lower()]] =  0
#     out = lab
    ### END YOUR CODE

    return out


def hsv_decomposition(image, channel='H'):
    """Decomposes the image into HSV and only returns the channel specified.

    Args:
        image: numpy array of shape(image_height, image_width, 3).
        channel: str specifying the channel. Can be either "H", "S" or "V".

    Returns:
        out: numpy array of shape(image_height, image_width).
    """

    hsv = color.rgb2hsv(image)
    out = None

    ### YOUR CODE HERE
    dic = {'h':0, 's':1, 'v':2}
    mat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    out = hsv * mat[dic[channel.lower()]]
    ### END YOUR CODE

    return out


def mix_images(image1, image2, channel1, channel2):
    """Combines image1 and image2 by taking the left half of image1
    and the right half of image2. The final combination also excludes
    channel1 from image1 and channel2 from image2 for each image.

    HINTS: Use `rgb_exclusion()` you implemented earlier as a helper
    function. Also look up `np.concatenate()` to help you combine images.

    Args:
        image1: numpy array of shape(image_height, image_width, 3).
        image2: numpy array of shape(image_height, image_width, 3).
        channel1: str specifying channel used for image1.
        channel2: str specifying channel used for image2.

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """

    out = None
    ### YOUR CODE HERE
    half = len(image1[0]) // 2
    tmp1 = rgb_exclusion(np.array(image1, dtype=np.float32), channel1)
    tmp2 = rgb_exclusion(np.array(image2, dtype=np.float32), channel2)
    tmp1[:, half:, :] = tmp2[:, half:, :]
    out = tmp1
    ### END YOUR CODE

    return out


def mix_quadrants(image):
    """THIS IS AN EXTRA CREDIT FUNCTION.

    This function takes an image, and performs a different operation
    to each of the 4 quadrants of the image. Then it combines the 4
    quadrants back together.

    Here are the 4 operations you should perform on the 4 quadrants:
        Top left quadrant: Remove the 'R' channel using `rgb_exclusion()`.
        Top right quadrant: Dim the quadrant using `dim_image()`.
        Bottom left quadrant: Brighthen the quadrant using the function:
            x_n = x_p^0.5
        Bottom right quadrant: Remove the 'R' channel using `rgb_exclusion()`.

    Args:
        image1: numpy array of shape(image_height, image_width, 3).

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """
    out = None

    ### YOUR CODE HERE
    half_height = len(image) // 2
    half_width = len(image[0]) // 2
    out = np.array(image)
    #Top left
    out[:half_height, :half_width, :] = rgb_exclusion(out[:half_height, :half_width, :], 'R')
    #Top right
    out[:half_height, half_width:, :] = dim_image(out[:half_height, half_width:, :])
    #Bottom left
    out[half_height:, :half_width, :] *= 0.5
    #Bottom right
    out[half_height:, half_width:, :] = rgb_exclusion(out[half_height:, half_width:, :], 'R')
    ### END YOUR CODE

    return out

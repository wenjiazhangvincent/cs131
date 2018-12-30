"""
CS131 - Computer Vision: Foundations and Applications
Assignment 1
Author: Donsuk Lee (donlee90@stanford.edu)
Date created: 07/2017
Last modified: 10/16/2017
Python Version: 3.5+
"""

import numpy as np


def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE=
    half_Hk, half_Wk = Hk // 2, Wk // 2
    for hi in range(Hi):
        for wi in range(Wi):
            tmp = 0
            for hk in range(Hk):
                if hi-(hk-half_Hk) < 0 or hi-(hk-half_Hk) > Hi-1:
                    continue
                for wk in range(Wk):
                    if  wi-(wk-half_Wk) < 0 or wi-(wk-half_Wk) > Wi-1:
                        continue
                    tmp += kernel[hk][wk] * image[hi-(hk-half_Hk)][wi-(wk-half_Wk)]
            out[hi][wi] = tmp
    ### END YOUR CODE

    return out

def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W).
        pad_width: width of the zero padding (left and right padding).
        pad_height: height of the zero padding (bottom and top padding).

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width).
    """

    H, W = image.shape
    out = None

    ### YOUR CODE HERE
    out = np.zeros((H + 2 * pad_height, W + 2 * pad_width))
    out[pad_height:H+pad_height, pad_width:W+pad_width] = image
    ### END YOUR CODE
    return out


def conv_fast(image, kernel):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    half_Hk, half_Wk = Hk // 2, Wk // 2
    pad_image = zero_pad(image, half_Hk, half_Wk)
    kernel = np.flip(kernel)
    for i in range(half_Hk, Hi+half_Hk):
        for j in range(half_Wk, Wi+half_Wk):
            tmp = np.sum(pad_image[i-half_Hk:i+half_Hk, j-half_Wk:j+half_Wk+1]
                         * kernel)
            out[i-half_Hk][j-half_Wk] = tmp
    ### END YOUR CODE

    return out

def conv_faster(image, kernel):
    """
    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    half_Hk, half_Wk = Hk // 2, Wk // 2
    pad_image = zero_pad(image, half_Hk, half_Wk)
    kernel = np.flip(kernel)
    # The trick is to lay out all the (Hk, Wk) patches and organize them into a (Hi*Wi, Hk*Wk) matrix.
    # Also consider the kernel as (Hk*Wk, 1) vector. Then the convolution naturally reduces to a matrix multiplication.
    mat = np.zeros((Hi*Wi, Hk*Wk))
    for i in range(Hi*Wi):
        row = i // Wi
        col = i % Wi
        mat[i, :] = pad_image[row:row+Hk, col:col+Wk].reshape(1, Hk*Wk)
    out = mat.dot(kernel.reshape(Hk*Wk, 1)).reshape(Hi, Wi)
    ### END YOUR CODE

    return out

def cross_correlation(f, g):
    """ Cross-correlation of f and g.

    Hint: use the conv_fast function defined above.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    ### YOUR CODE HERE
    g = np.flip(g)
    out = conv_fast(f, g)
#     out = conv_faster(f, g)
    ### END YOUR CODE

    return out

def zero_mean_cross_correlation(f, g):
    """ Zero-mean cross-correlation of f and g.

    Subtract the mean of g from g so that its mean becomes zero.

    Hint: you should look up useful numpy functions online for calculating the mean.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    ### YOUR CODE HERE
    g -= np.mean(g)
    out = cross_correlation(f, g)
    ### END YOUR CODE

    return out

def normalized_cross_correlation(f, g):
    """ Normalized cross-correlation of f and g.

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    Hint: you should look up useful numpy functions online for calculating 
          the mean and standard deviation.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    ### YOUR CODE HERE
    f = (f - np.mean(f)) / np.var(f)
    g = (g - np.mean(g)) / np.var(g)
    out = cross_correlation(f, g)
    ### END YOUR CODE

    return out

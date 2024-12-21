import numpy as np
from mahotas import sobel


def EDF(image):
    """Calculate Extended Depth of Field for the given input image Z-stack.
    Based on implementation here: https://mahotas.readthedocs.io/en/latest/edf.html#id3

    Args:
        image (np.array): Input image array of shape (Z, X, Y)

    Returns:
        np.array: EDF selected image
    """

    # get image stack sizes
    stack, h, w = image.shape

    # determine in focusness for each pixel
    focus = np.array([sobel(z, just_filter=True) for z in image])

    # select best focal plane for each pixel
    best = np.argmax(focus, 0)

    image = image.reshape((stack, -1))  # image is now (stack, nr_pixels)
    image = image.transpose()  # image is now (nr_pixels, stack)
    r = image[np.arange(len(image)), best.ravel()]  # Select the right pixel at each location
    r = r.reshape((h, w))  # reshape to get final result

    return r


def maximum_intensity_projection(image):
    """Calculate Extended Depth of Field for the given input image Zstack.

    Args:
        image (np.array): Input image array of shape (Z, X, Y)

    Returns:
        np.array: Maximum Intensity Projected Image.
    """
    return np.max(image, axis=0)

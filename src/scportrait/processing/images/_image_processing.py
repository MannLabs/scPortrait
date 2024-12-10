import numpy as np
import xarray as xr
from numba import jit
from skimage.exposure import rescale_intensity


def rescale_image(image, rescale_range, outrange=(0, 1), dtype="uint16", cutoff_threshold=None, return_float=False):
    # convert to float for better percentile calculation
    img = image.astype("float")

    # define factor to rescale image to uints
    if dtype == "uint16":
        factor = 65535
    elif dtype == "uint8":
        factor = 255

    # get cutoff values
    cutoff1, cutoff2 = rescale_range

    # if cutoff_threshold is given, set all values above the threshold to 1 but do not consider them for percentile calculation
    if cutoff_threshold is not None:
        if img.max() > cutoff_threshold:
            _img = img.copy()
            values = _img[_img < (cutoff_threshold / factor)]
    else:
        values = img

    # calculate percentiles
    p1 = np.percentile(values, cutoff1)
    p99 = np.percentile(values, cutoff2)
    del values

    if return_float:
        return rescale_intensity(img, (p1, p99), outrange)
    else:
        return (rescale_intensity(img, (p1, p99), outrange) * factor).astype(dtype)


def _normalize_image(im, lower_value, upper_value):
    """
    Normalize an input 2D image (np.array) based on input values.

    Args:
        im (np.array): Numpy array of shape (height, width).
        lower_value (int): Lower image value used for normalization, all lower values will be clipped to 0.
        upper_value (int): Upper image value used for normalization, all higher values will be clipped to 1.

    Returns:
        out_im (np.array): Normalized Numpy array.

    Example:
    >>> img = np.random.rand(4, 4)
    >>> norm_img = _normalize_image(img, 200, 15000)
    """

    # Compute inter-percentile range (IPR)
    IPR = upper_value - lower_value

    # Normalize the image
    out_im = im - lower_value

    if IPR != 0:  # add check to make sure IPR is not 0
        out_im = out_im / IPR

    # Clip values outside [0, 1]
    out_im = np.clip(out_im, 0, 1)

    return out_im


def _percentile_norm(im, lower_percentile, upper_percentile):
    """
    Normalize an input 2D image (np.array) based on the defined percentiles.

    This is an helper function used in the percentile_normalization function.

    Args:
        im (np.array): Numpy array of shape (height, width).
        lower_percentile (float): Lower percentile used for normalization, all lower values will be clipped to 0.
        upper_percentile (float): Upper percentile used for normalization, all higher values will be clipped to 1.

    Returns:
        out_im (np.array): Normalized Numpy array.

    Example:
    >>> img = np.random.rand(4, 4)
    >>> norm_img = _percentile_norm(img, 0.001, 0.999)
    """

    # Calculate the quantiles
    lower_value = np.quantile(np.ravel(im), lower_percentile)
    upper_value = np.quantile(np.ravel(im), upper_percentile)

    out_im = _normalize_image(im, lower_value, upper_value)

    return out_im


def percentile_normalization(im, lower_percentile=0.001, upper_percentile=0.999, return_copy=True):
    """
    Normalize an input image channel-wise based on defined percentiles.

    The percentiles will be calculated, and the image will be normalized to [0, 1]
    based on the lower and upper percentile.

    Args:
        im (np.array): Numpy array of shape (height, width) or (channels, height, width).
        lower_percentile (float, optional): Lower percentile used for normalization, all lower values will be clipped to 0. Defaults to 0.001.
        upper_percentile (float, optional): Upper percentile used for normalization, all higher values will be clipped to 1. Defaults to 0.999.

    Returns:
        im (np.array): Normalized Numpy array with dtype == float

    Example:
    >>> img = np.random.rand(3, 4, 4)  # (channels, height, width)
    >>> norm_img = percentile_normalization(img, 0.001, 0.999)
    """
    # if false the input image is directly updated
    if return_copy:
        im = im.copy()  # ensure we are working on a copy not the original data

    # ensure that the dtype is converted to a float before running this function
    if not isinstance(im.dtype, float):
        im = im.astype(np.float32)

    # chek if data is passed as (height, width) or (channels, height, width)
    if len(im.shape) == 2:
        im = _percentile_norm(im, lower_percentile, upper_percentile)

    elif len(im.shape) == 3:
        for i, _channel in enumerate(im):
            im[i] = _percentile_norm(im[i], lower_percentile, upper_percentile)

    else:
        raise ValueError("Input dimensions should be (height, width) or (channels, height, width).")

    return im


def downsample_img(img, N=2, return_dtype=np.uint16):
    """
    Function to downsample an image in shape CXY equivalent to NxN binning using the mean between pixels.
    Takes a numpy array image as input and returns a numpy array as uint16.

    Parameters
    ----------
    img : array
        image to downsample
    N : int, default = 2
        number of pixels that should be binned together using mean between pixels
    """
    downsampled = xr.DataArray(img, dims=["c", "x", "y"]).coarsen(c=1, x=N, y=N, boundary="exact").mean()
    downsampled = (downsampled / downsampled.max() * np.iinfo(return_dtype).max).astype(return_dtype)
    return np.array(downsampled)


def downsample_img_pxs(img, N=2):
    """
    Function to downsample an image in shape CXY equivalent to taking every N'th pixel from each dimension.
    Channels are preserved as is. This does not use any interpolation and is therefore faster than the mean
    method but is less precise.

    Parameters
    ----------
    img : array
        image to downsample
    N : int, default = 2
        the nth pixel to take from each dimension
    """

    downsampled = img[:, 0:-1:N, 0:-1:N]
    return downsampled


def downsample_img_padding(img, N=2):
    """
    Downsample image by a factor of N by taking every Nth pixel.
    Before downsampling this function will pad the image to ensure its compatible with the selected kernel size.

    Parameters
    ----------
    img
        image to be downsampled

    Returns
    -------
    downsampled image

    """

    # check if N fits perfectly into image shape if not calculate how much we need to pad
    if len(img.shape) == 3:
        _, x, y = img.shape
    elif len(img.shape) == 2:
        x, y = img.shape

    if x % N == 0:
        pad_x = (0, 0)
    else:
        pad_x = (0, N - x % N)

    if y % N == 0:
        pad_y = (0, 0)
    else:
        pad_y = (0, N - y % N)

    print(f"Performing image padding to ensure that image is compatible with selected downsample kernel size of {N}.")

    # perform image padding to ensure that image is compatible with downsample kernel size
    img = np.pad(img, ((0, 0), pad_x, pad_y))

    print(f"Downsampling image by a factor of {N}x{N}")

    # actually perform downsampling
    img = downsample_img(img, N=N)

    return img


@jit(nopython=True, parallel=True)  # Set "nopython" mode for best performance, equivalent to @njit
def rolling_window_mean(array, size, scaling=False):
    """
    Compute rolling window mean and normalize the input 2D array.

    The function takes an input 2D array and a window size. It calculates the mean
    within a rolling window of provided size and removes the mean from each element
    in the window. The modified array is returned. If scaling is set to True,
    the chunk is normalized by dividing by its standard deviation.
    Function is numba optimized.

    Args:
        array (np.array): Input 2D numpy array.
        size (int): Size of the rolling window.
        scaling (bool, optional): If True, normalizes the chunk by dividing it by standard deviation, by default False.

    Returns:
        array (np.array): Processed 2D numpy array.

    Example:
    >>> array = np.random.rand(10, 10)
    >>> rolling_array = rolling_window_mean(array, size=5, scaling=False)
    """

    overlap = 0
    lengthy, lengthx = array.shape
    delta = size - overlap

    # calculate the number of steps given the window size
    ysteps = lengthy // delta
    xsteps = lengthx // delta

    x = 0
    y = 0

    # Iterate through the array with the given window size
    for i in range(ysteps):
        for j in range(xsteps):
            y = i * delta
            x = j * delta

            yd = min(y + size, lengthy)
            xd = min(x + size, lengthx)

            # Extract the current window (chunk) and calculate statistics
            chunk = array[y:yd, x:xd]
            std = np.std(chunk.flatten())
            max = np.max(chunk.flatten())

            # Scale the chunk, if scaling is True
            if scaling:
                chunk = chunk / std

            # Compute the mean and remove it from the chunk
            mean = np.median(chunk.flatten())

            if max > 0:
                chunk = chunk - mean
                chunk = np.where(chunk < 0, 0, chunk)

            array[y:yd, x:xd] = chunk

    if scaling:
        array = array / np.max(array.flatten())
    return array


def MinMax(inarr):
    """
    Perform Min-Max normalization on the input array.
    The function scales the input array values to the range [0, 1] using Min-Max
    normalization. If the range of the input array (i.e., max - min) is zero,
    the original array is returned unchanged.

    Args:
        inarr (np.array): Input 2D numpy array.

    Returns:
        np.array: Min-Max normalized numpy array.

    Example:
    >>> array = np.random.rand(10, 10)
    >>> normalized_array = MinMax(array)
    """

    if np.max(inarr) - np.min(inarr) > 0:
        return (inarr - np.min(inarr)) / (np.max(inarr) - np.min(inarr))
    else:
        return inarr

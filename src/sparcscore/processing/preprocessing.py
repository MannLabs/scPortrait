from numba import jit
import numpy as np

from skimage.filters import gaussian
from skimage.feature import peak_local_max
from scipy.ndimage import distance_transform_edt
from mahotas import sobel

def _percentile_norm(im, lower_percentile, upper_percentile):
    """
    Normalize an input 2D image (np.array) based on the defined percentiles.
    
    This is an helper function used in the percentile_normalization function.
    
    Parameters
    ----------
    im : np.array
        Numpy array of shape (height, width).
    lower_percentile : float
        Lower percentile used for normalization, all lower values will be clipped to 0.
    upper_percentile : float
        Upper percentile used for normalization, all higher values will be clipped to 1.

    Returns
    -------
    out_im : np.array
        Normalized Numpy array.
    
    Example
    -------
    >>> img = np.random.rand(4, 4)
    >>> norm_img = _percentile_norm(img, 0.001, 0.999)
    """
    
    # Calculate the quantiles
    lower_value = np.quantile(np.ravel(im),lower_percentile)
    upper_value = np.quantile(np.ravel(im),upper_percentile)

    # Compute inter-percentile range (IPR)                                     
    IPR = upper_value - lower_value

    # Normalize the image
    out_im = im - lower_value 
    
    if IPR != 0: #add check to make sure IPR is not 0
        out_im = out_im / IPR
    
    # Clip values outside [0, 1]
    out_im = np.clip(out_im, 0, 1)
            
    return out_im

def percentile_normalization(im, lower_percentile = 0.001, upper_percentile = 0.999):
    """
    Normalize an input image channel-wise based on defined percentiles.

    The percentiles will be calculated, and the image will be normalized to [0, 1]
    based on the lower and upper percentile.

    Parameters
    ----------
    im : np.array
        Numpy array of shape (height, width) or (channels, height, width).
    lower_percentile : float, between [0, 1]
        Lower percentile used for normalization, all lower values will be clipped to 0.
    upper_percentile : float, between [0, 1]
        Upper percentile used for normalization, all higher values will be clipped to 1.

    Returns
    -------
    im : np.array
        Normalized Numpy array.

    Example
    -------
    >>> img = np.random.rand(3, 4, 4) # (channels, height, width)
    >>> norm_img = percentile_normalization(img, 0.001, 0.999)
    """
    
    # chek if data is passed as (height, width) or (channels, height, width)
    
    if len(im.shape) == 2:
        im = _percentile_norm(im, lower_percentile, upper_percentile)
        
    elif len(im.shape) == 3:
        for i, channel in enumerate(im):
            im[i] = _percentile_norm(im[i], lower_percentile, upper_percentile)
            
    else:
        raise ValueError("Input dimensions should be (height, width) or (channels, height, width).")

    return im
    

@jit(nopython=True, parallel = True) # Set "nopython" mode for best performance, equivalent to @njit
def rolling_window_mean(array, size, scaling = False):
    """
    Compute rolling window mean and normalize the input 2D array.

    The function takes an input 2D array and a window size. It calculates the mean
    within a rolling window of provided size and removes the mean from each element
    in the window. The modified array is returned. If scaling is set to True,
    the chunk is normalized by dividing by its standard deviation.
    Function is numba optimized.
    
    Parameters
    ----------
    array : np.array
        Input 2D numpy array.
    size : int
        Size of the rolling window.
    scaling : bool, optional
        If True, normalizes the chunk by dividing it by standard deviation, by default False.

    Returns
    -------
    array : np.array
        Processed 2D numpy array.

    Example
    -------
    >>> array = np.random.rand(10, 10)
    >>> rolling_array = rolling_window_mean(array, size=5, scaling=False)
    """

    overlap=0
    lengthy, lengthx = array.shape
    delta = size-overlap
    
    #calculate the number of steps given the window size
    ysteps = lengthy // delta
    xsteps = lengthx // delta
    
    x = 0
    y = 0
    
    # Iterate through the array with the given window size
    for i in range(ysteps):
        for j in range(xsteps):
            y = i*delta
            x = j*delta
            
            yd = min(y+size,lengthy)
            xd = min(x+size,lengthx)
            
            # Extract the current window (chunk) and calculate statistics
            chunk = array[y:yd,x:xd]
            std = np.std(chunk.flatten())
            max = np.max(chunk.flatten())
            
             # Scale the chunk, if scaling is True
            if scaling:
                    chunk = chunk / std
                    
            # Compute the mean and remove it from the chunk
            mean = np.median(chunk.flatten())
            
            if max > 0:
                chunk = (chunk - mean)
                chunk = np.where(chunk < 0,0,chunk)
       
            array[y:yd,x:xd] = chunk

    if scaling:
        array = array/np.max(array.flatten())
    return array

def origins_from_distance(array):
    """
    Compute a list of local peaks (origins) in the input array.
    The function applies a Gaussian filter to the input array, creates a binary
    mask using a threshold based on the standard deviation of the filtered data, and
    performs a distance transform on the mask. It finds the local peaks of the
    distance-transformed image with a minimum distance between the peaks,
    and returns the coordinates and a map of the peaks.

    Parameters
    ----------
    array : np.array
        Input 2D numpy array to be processed.
    
    Returns
    -------
    peak_list : np.array
        List of peak coordinates, shape (num_peaks, 2).
    peak_map : np.array
        Binary map of peaks, same shape as input array.

    Example
    -------
    >>> array = np.random.rand(10, 10)
    >>> peak_list, peak_map = origins_from_distance(array)
    """

    # Apply Gaussian filter to the input array
    array = gaussian(array, sigma=1)
    std = np.std(array.flatten())
    
    # Threshold the filtered array and perform distance transform
    thr_mask = np.where(array > 3 * std, 1,0)
    distance = distance_transform_edt(thr_mask)
    
    # Find local peaks in the distance-transformed array
    peak_list  = peak_local_max(distance, min_distance=5, threshold_abs=std,footprint=np.ones((3, 3)))

    # Create a map with True elements at peak locations
    peak_map = np.zeros_like(array, dtype=bool) ### WHAT IS BASE AND WHAT DOES IT DO??
    peak_map[tuple(peak_list.T)] = True
    
    return peak_list, peak_map

def MinMax(inarr):
    """
    Perform Min-Max normalization on the input array.
    The function scales the input array values to the range [0, 1] using Min-Max
    normalization. If the range of the input array (i.e., max - min) is zero,
    the original array is returned unchanged.

    Parameters
    ----------
    inarr : np.array
        Input numpy array to be normalized.

    Returns
    -------
    np.array
        Min-Max normalized numpy array.

    Example
    -------
    >>> array = np.random.rand(10, 10)
    >>> normalized_array = MinMax(array)
    """

    if np.max(inarr) - np.min(inarr) > 0:
        return (inarr - np.min(inarr)) / (np.max(inarr) - np.min(inarr))
    else:
        return inarr
    
def EDF(image):
    """ Calculate Extended Depth of Field for the given input image Zstack.

    Parameters
    ----------
    image : np.array
        Input image array of shape (Z, X, Y)
    
    Returns
    -------
    np.array
        EDF selected image
    """

    #get image stack sizes
    stack, h, w = image.shape

    #determine in focusness for each pixel
    focus = np.array([sobel(z, just_filter=True) for z in image])
    
    #select best focal plane for each pixel
    best = np.argmax(focus, 0)

    image = image.reshape((stack,-1)) # image is now (stack, nr_pixels)
    image = image.transpose() # image is now (nr_pixels, stack)
    r = image[np.arange(len(image)), best.ravel()] # Select the right pixel at each location
    r = r.reshape((h,w)) # reshape to get final result

    return (r)

def maximum_intensity_projection(image):
    """ Calculate Extended Depth of Field for the given input image Zstack.

    Parameters
    ----------
    image : np.array
        Input image array of shape (Z, X, Y)
    
    Returns
    -------
    np.array
        Maximum Intensity Projected Image.
    """
    return(np.max(image, axis=0))
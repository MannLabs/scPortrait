import numpy as np
from skimage.filters import gaussian
from skimage.feature import peak_local_max
from scipy.ndimage import distance_transform_edt

def origins_from_distance(array):
    """
    Compute a list of local peaks (origins) in the input array.
    The function applies a Gaussian filter to the input array, creates a binary
    mask using a threshold based on the standard deviation of the filtered data, and
    performs a distance transform on the mask. It finds the local peaks of the
    distance-transformed image with a minimum distance between the peaks,
    and returns the coordinates and a map of the peaks.

    Args:
        array (np.array): Input 2D numpy array.
    
    Returns:
        peak_list (np.array): List of peak coordinates, shape (num_peaks, 2).
        peak_map (np.array): Binary map of peaks, same shape as input array.

    Example:
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
import matplotlib.pyplot as plt
import numba as nb
import numpy as np
import skfmm
from numba import njit, prange
from scipy import ndimage
from skimage import filters
from skimage.color import label2rgb
from skimage.feature import peak_local_max
from skimage.morphology import binary_erosion, disk
from skimage.morphology import dilation as sk_dilation
from skimage.segmentation import watershed
from skimage.transform import resize
from sparcscore.processing.utils import plot_image


#### Thresholding Functions to binarize input images
def global_otsu(image):
    """Calculate the optimal global threshold for an input grayscale image using Otsu's method.

    Otsu's method maximizes the between-class variance and minimizes the within-class variance.

    Parameters
    ----------
    image : np.array
        Input grayscale image.

    Returns:
    -------
    threshold : float
        Optimal threshold value calculated using Otsu's method.

    Examples:
    --------
    >>> import numpy as np
    >>> from skimage import data
    >>> image = data.coins()
    >>> threshold = global_otsu(image)
    >>> print(threshold)

    """
    # Calculate histogram of input image and bin centers
    counts, bin_edges = np.histogram(np.ravel(image), bins=512)
    bin_centers = bin_edges[:-1] + np.diff(bin_edges) / 2

    # Calculate cumulative sum of counts for class 1 and class 2
    weight1 = np.cumsum(counts)
    weight2 = np.cumsum(counts[::-1])[::-1]

    # class means for all possible thresholds
    mean1 = np.cumsum(counts * bin_centers) / weight1
    mean2 = (np.cumsum((counts * bin_centers)[::-1]) / weight2[::-1])[::-1]

    # Calculate between-class variance for all possible thresholds
    # Clip ends to align class 1 and class 2 variables:
    # The last value of ``weight1``/``mean1`` should pair with zero values in
    # ``weight2``/``mean2``, which do not exist.
    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

    # Find the threshold value that maximizes the between-class variance
    idx = np.argmax(variance12)
    threshold = bin_centers[idx]

    return threshold


def _segment_threshold(
    image,
    threshold,
    dilation=4,
    min_distance=10,
    peak_footprint=7,
    speckle_kernel=4,
    debug=False,
):
    """Perform image segmentation using an input threshold and additional user-defined parameters.

    This is a private helper function for segmenting images using a given threshold. It combines several
    operations like binary erosion and dilation, distance transforms, and watershed segmentation to obtain
    unique labels for distinct regions in the input image.

    Parameters
    ----------
    image : np.array
        Input grayscale image.
    threshold : float
        Threshold value for creating image_mask.
    dilation : int, optional
        Size of the structuring element for the dilation operation (default is 4).
    min_distance : int, optional
        Minimum number of pixels separating peaks (default is 10).
    peak_footprint : int, optional
        Size of the structuring element used for finding local maxima (default is 7).
    speckle_kernel : int, optional
        Size of the structuring element used for speckle removal (default is 4).
    debug : bool, optional
        If True, intermediate results are plotted (default is False).

    Returns:
    -------
    labels : np.array
        Labeled array, where each unique feature in the input image has a unique label.

    Example:
    -------
    This function is meant to be used internally as a helper function for other segmentation methods.
    It should not be used directly.

    """
    image_mask = image > threshold

    # If debug is set to True, plot the binary mask
    if debug:
        plot_image(image_mask, cmap="Greys_r")

    # removing speckles by binary erosion and dilation
    image_mask_clean = binary_erosion(image_mask, footprint=disk(speckle_kernel))
    image_mask_clean = sk_dilation(image_mask_clean, footprint=disk(speckle_kernel - 1))

    # Find peaks in the image mask using a distance transform
    distance = ndimage.distance_transform_edt(image_mask_clean)

    peak_idx = peak_local_max(distance, min_distance=min_distance, footprint=disk(peak_footprint))
    local_maxi = np.zeros_like(
        image_mask_clean, dtype=bool
    )  # Initialize an array of zeros with the same shape as image_mask_clean
    local_maxi[tuple(peak_idx.T)] = True

    # If debug is set to True, plot the local maxima on the binary mask
    if debug:
        fig = plt.figure(frameon=False)
        fig.set_size_inches(10, 10)
        ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(image_mask_clean, cmap="Greys_r")
        plt.scatter(peak_idx[:, 1], peak_idx[:, 0], color="red")
        plt.show()

    # segmentation by fast marching and watershed
    dilated_mask = sk_dilation(image_mask_clean, footprint=disk(dilation))

    fmm_marker = np.ones_like(dilated_mask)
    for center in peak_idx:
        fmm_marker[center[0], center[1]] = 0

    m = np.ma.masked_array(fmm_marker, np.logical_not(dilated_mask))
    distance_2 = skfmm.distance(m)

    # If debug is True, plot the calculated distance_2
    if debug:
        fig = plt.figure(frameon=False)
        fig.set_size_inches(10, 10)
        ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(distance_2, cmap="viridis")
        plt.scatter(peak_idx[:, 1], peak_idx[:, 0], color="red")

    # Assign unique labels to each segmented region
    marker = np.zeros_like(image_mask_clean).astype(int)
    for i, center in enumerate(peak_idx):
        marker[center[0], center[1]] = i + 1

    labels = watershed(distance_2, marker, mask=dilated_mask)

    # If debug is True, visualize the final labels on the image
    if debug:
        image = label2rgb(labels, image / np.max(image), alpha=0.2, bg_label=0)
        plot_image(image)

    return labels


def segment_global_threshold(image, dilation=4, min_distance=10, peak_footprint=7, speckle_kernel=4, debug=False):
    """Segments an image based on a global threshold determined using Otsu's method, followed by peak detection using
    distance transforms and watershed segmentation.

    Parameters
    ----------
    image : np.array
        Input grayscale image.
    dilation : int, optional
        Size of the structuring element for the dilation operation (default is 4).
    min_distance : int, optional
        Minimum number of pixels separating peaks (default is 10).
    peak_footprint : int, optional
        Size of the structuring element used for finding local maxima (default is 7).
    speckle_kernel : int, optional
        Size of the structuring element used for speckle removal (default is 4).
    debug : bool, optional
        If True, intermediate results are plotted (default is False).

    Returns:
    -------
    labels : np.array
        Labeled array, where each unique feature in the input image has a unique label.

    Example:
    -------
    >>> import matplotlib.pyplot as plt
    >>> from skimage import data
    >>> from segment import segment_global_threshold
    >>> coins = data.coins()
    >>> labels = segment_global_threshold(
    ...     coins, dilation=5, min_distance=20, peak_footprint=7, speckle_kernel=3, debug=True
    ... )
    >>> plt.imshow(labels, cmap="jet")
    >>> plt.colorbar()
    >>> plt.show()
    """
    # calculate the global threshold
    threshold = global_otsu(image)

    labels = _segment_threshold(
        image,
        threshold,
        dilation=dilation,
        min_distance=min_distance,
        peak_footprint=peak_footprint,
        speckle_kernel=speckle_kernel,
        debug=debug,
    )

    # returnt the segmented image
    return labels


def segment_local_threshold(
    image,
    dilation=4,
    thr=0.01,
    median_block=51,
    min_distance=10,
    peak_footprint=7,
    speckle_kernel=4,
    median_step=1,
    debug=False,
):
    """This function takes a unprocessed image with low background noise and extracts and segments approximately round foreground objects
    based on intensity. The image is segmented using the local (adaptive) threshold method. It first applies a local threshold based on
    the median value of a block of pixels, followed by peak detection using distance transforms and watershed segmentation.

    Parameters
    ----------
    image : np.array
        Input grayscale image.
    dilation : int, optional
        Size of the structuring element for the dilation operation (default is 4).
    thr : float, optional
        The value added to the median of the block of pixels when calculating the local threshold (default is 0.01).
    median_block : int, optional
        Size of the block of pixels used to compute the local threshold (default is 51).
    min_distance : int, optional
        Minimum number of pixels separating peaks (default is 10).
    peak_footprint : int, optional
        Size of the structuring element used for finding local maxima (default is 7).
    speckle_kernel : int, optional
        Size of the structuring element used for speckle removal (default is 4).
    median_step : int, optional
        The step size for downsampling the image before applying the local threshold (default is 1).
    debug : bool, optional
        If True, intermediate results are plotted (default is False).

    Returns:
    -------
    labels : np.array
        Labeled array, where each unique feature in the input image has a unique label.

    Example:
    -------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from skimage import data
    >>> image = data.coins()
    >>> labels = segment_local_threshold(
    ...     image,
    ...     dilation=4,
    ...     thr=0.01,
    ...     median_block=51,
    ...     min_distance=10,
    ...     peak_footprint=7,
    ...     speckle_kernel=4,
    ...     debug=False,
    ... )
    >>> plt.imshow(labels, cmap="jet")
    >>> plt.colorbar()
    >>> plt.show()
    """
    if speckle_kernel < 1:
        raise ValueError("speckle_kernel needs to be at least 1")

    # Downsample the input image
    downsampled_image = image[::median_step, ::median_step]

    # Calculate the local threshold using median filtering
    local_threshold = filters.threshold_local(downsampled_image, block_size=median_block, method="median", offset=-thr)

    # Resize the local threshold image to match the original input image
    local_threshold = resize(local_threshold, image.shape)

    # Segment the input image using the computed local threshold
    labels = _segment_threshold(
        image,
        local_threshold,
        dilation=dilation,
        min_distance=min_distance,
        peak_footprint=peak_footprint,
        speckle_kernel=speckle_kernel,
        debug=debug,
    )

    return labels


#### Collection of functions to modify, filter or remove labels from segmentation masks


# helper function to allow numba optimization for subtraction
@njit
def _numba_subtract(array1, number):
    """Subtract a minimum number from all non-zero elements of the input array.

    Parameters
    ----------
    array1 : np.ndarray
        Input array.
    number : int
        The number to be subtracted from all non-zero elements.

    Returns:
    -------
    array1 : np.ndarray
        The resulting array after subtracting the number from non-zero elements.

    Example:
    -------
    >>> import numpy as np
    >>> array1 = np.array([[0, 2, 3], [0, 5, 6], [0, 0, 7]])
    >>> _numba_subtract(array1, 1)
    array([[0, 1, 2],
           [0, 4, 5],
           [0, 0, 6]])

    """
    for i in range(array1.shape[0]):  # parallel --> for i in nb.prange(c.shape[0]):
        for j in range(array1.shape[1]):
            if array1[i, j] != 0:
                array1[i, j] = array1[i, j] - number

    return array1


@njit
def _return_edge_labels(input_map):
    """Return the unique labels in contact with the edges of the input_map.

    Parameters
    ----------
    input_map : np.ndarray
        Input segmentation as a 2D numpy array of integers.

    Returns:
    -------
    edge_labels : list
        List of unique labels in contact with the edges of the input_map.

    """
    top_row = input_map[0]
    bottom_row = input_map[-1]
    first_column = input_map[:, 0]
    last_column = input_map[:, -1]

    full_union = (
        set(top_row.flatten())
        .union(set(bottom_row.flatten()))
        .union(set(first_column.flatten()))
        .union(set(last_column.flatten()))
    )
    full_union = {np.int64(i) for i in full_union}
    full_union.discard(0)

    return list(full_union)


def shift_labels(input_map, shift, return_shifted_labels=False, remove_edge_labels=True):
    """Shift the labels of a given labeled map (2D or 3D numpy array) by a specific value.
    Return the shifted map and the labels that are in contact with the edges of the canvas.
    All labels but the background are incremented and all classes in contact with the edges of the
    canvas are returned.

    Parameters
    ----------
    input_map : np.ndarray
        Input segmentation as a 2D or 3D numpy array of integers.
    shift : int
        Value to increment the labels by.
    return_shifted_labels : bool, optional
        If True, return the edge labels after shifting (default is False).
        If False will return the edge labels before shifting.

    Returns:
    -------
    shifted_map : np.ndarray
        The labeled map with incremented labels.
    edge_labels : list
        List of unique labels in contact with the edges of the canvas.

    Example:
    -------
    >>> import numpy as np
    >>> input_map = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
    >>> shift = 10
    >>> shifted_map, edge_labels = shift_labels(input_map, shift)
    >>> print(shifted_map)
    array([[11,  0,  0],
        [ 0, 12,  0],
        [ 0,  0, 13]])
    >>> print(edge_labels)
    [11, 13]

    """
    imap = input_map[:].copy()
    shifted_map = np.where(imap == 0, 0, imap + shift)

    edge_label = []

    if len(imap.shape) == 2:
        edge_label += _return_edge_labels(imap)
    else:
        for dimension in imap:
            edge_label += _return_edge_labels(dimension)

    if return_shifted_labels:
        edge_label = [label + shift for label in edge_label]

    if remove_edge_labels:
        if return_shifted_labels:
            shifted_map = np.where(np.isin(shifted_map, edge_label), 0, shifted_map)
        else:
            _edge_label = [label + shift for label in edge_label]
            shifted_map = np.where(np.isin(shifted_map, _edge_label), 0, shifted_map)

    return shifted_map, list(set(edge_label))


@njit(parallel=True)
def _remove_classes(label_in, to_remove, background=0, reindex=False):
    """Remove specified classes from a labeled input array.

    This function takes a labeled array and removes the specified classes. It
    assigns the background value to these classes and, if reindex=True, reindexes
    the remaining classes.

    Parameters
    ----------
    label_in : np.ndarray
        Input labeled array.
    to_remove : list
        List of label classes to remove.
    background : int, optional
        Value used to represent the background class (default is 0).
    reindex : bool, optional
        If True, reindex the remaining classes after removal (default is False).

    Returns:
    -------
    label : np.ndarray
        Labeled array with specified classes removed or reindexed.

    Example:
    -------
    >>> import numpy as np
    >>> label_in = np.array([[1, 2, 1], [1, 0, 2], [0, 2, 3]])
    >>> to_remove = [1, 3]
    >>> _remove_classes(label_in, to_remove)
    array([[0, 2, 0],
           [0, 0, 2],
           [0, 2, 0]])

    """
    label = label_in.copy()

    # generate library which contains the new class for label x at library[x]
    remove_set = set(to_remove)
    library = np.zeros(np.max(label) + 1, dtype="int")

    # Assign new labels for each class based on whether it is to be removed, and whether reindexing is enabled
    carry = 0
    for current_class in range(len(library)):
        if current_class in remove_set:
            library[current_class] = background
            if reindex:
                carry -= 1
        else:
            library[current_class] = current_class + carry

    # Update the label array based on the library
    for y in prange(len(label)):
        for x in prange(len(label[0])):
            current_label = label[y, x]
            if current_label != background:
                label[y, x] = library[current_label]

    return label


def remove_classes(label_in, to_remove, background=0, reindex=False):
    """Wrapper function for the numba optimized _remove_classes function.

    Parameters
    ----------
    label_in : np.ndarray
        Input labeled array.
    to_remove : list
        List of label classes to remove.
    background : int, optional
        Value used to represent the background class (default is 0).
    reindex : bool, optional
        If True, reindex the remaining classes after removal (default is False).

    Returns:
    -------
    label : np.ndarray
        Labeled array with specified classes removed or reindexed.

    Example:
    -------
    >>> import numpy as np
    >>> label_in = np.array([[1, 2, 1], [1, 0, 2], [0, 2, 3]])
    >>> to_remove = [1, 3]
    >>> remove_classes(label_in, to_remove)
    array([[0, 2, 0],
           [0, 0, 2],
           [0, 2, 0]])

    """
    return _remove_classes(label_in, to_remove, background=background, reindex=reindex)


@njit(parallel=True)
def contact_filter_lambda(label, background=0):
    """Calculate the contact proportion of classes in the labeled array.

    This function calculates the surrounding background and non-background elements
    for each class in the label array and returns the proportion of background elements
    to total surrounding elements for each class.

    Parameters
    ----------
    label : np.ndarray
        Input labeled array.
    background : int, optional
        Value used to represent the background class (default is 0).

    Returns:
    -------
    pop : np.ndarray
        Array containing the contact proportion for each class.

    Example:
    -------
    >>> import numpy as np
    >>> label = np.array([[0, 1, 1], [0, 2, 1], [0, 0, 2]])
    >>> contact_filter_lambda(label)
    array([1.        , 1.,  0.6 ])

    """
    num_labels = np.max(label)

    # Initialize a background_matrix
    background_matrix = np.ones((num_labels + 1, 2), dtype="int")

    for y in range(1, len(label) - 1):
        for x in range(1, len(label[0]) - 1):
            current_label = label[y, x]

            # Count the background (0) and non-background contact pixels
            background_matrix[current_label, 0] += int(label[y - 1, x] == background)
            background_matrix[current_label, 0] += int(label[y, x - 1] == background)
            background_matrix[current_label, 0] += int(label[y + 1, x] == background)
            background_matrix[current_label, 0] += int(label[y, x + 1] == background)

            # Count the non-background contact pixels
            background_matrix[current_label, 1] += int(label[y - 1, x] != current_label)
            background_matrix[current_label, 1] += int(label[y, x - 1] != current_label)
            background_matrix[current_label, 1] += int(label[y + 1, x] != current_label)
            background_matrix[current_label, 1] += int(label[y, x + 1] != current_label)

    # Compute the proportion of background contact pixels
    proportion = background_matrix[:, 0] / background_matrix[:, 1]

    return proportion


def contact_filter(inarr, threshold=1, reindex=False, background=0):
    """Filter the input labeled array based on its contact with background pixels.

    This function filters an input labeled array by removing classes with a background
    contact proportion less than the given threshold.

    Parameters
    ----------
    inarr : np.ndarray
        Input labeled array.
    threshold : int, optional
        Specifies the minimum background contact proportion for class retention (default is 1).
    reindex : bool, optional
        If True, reindexes the remaining classes after removal (default is False).
    background : int, optional
        Value used to represent the background class (default is 0).

    Returns:
    -------
    label : np.ndarray
        Filtered labeled array.

    Example:
    -------
    >>> import numpy as np
    >>> inarr = np.array([[0, 1, 1], [0, 2, 1], [0, 0, 2]])
    >>> contact_filter(inarr, threshold=0.5)
    array([[0, 1, 1],
           [0, 0, 1],
           [0, 0, 0]])

    """
    label = inarr.copy()

    # laulate contact matrix for labels
    background_contact = contact_filter_lambda(label, background=0)

    # extract all classes with less background contact than the threshold, but not the background  class
    to_remove = np.argwhere(background_contact < threshold).flatten()

    to_remove = np.delete(to_remove, np.where(to_remove == background))

    # numba typed list fails to determine type for empty list
    # return without removing classes if no classes should be removed
    if len(to_remove) > 0:
        # remove these classes
        label = remove_classes(label, nb.typed.List(to_remove), reindex=reindex)
    else:
        pass

    return label


@njit
def _class_size(mask, debug=False, background=0):
    """Compute the size (number of pixels) of each class in the given mask.

    This function calculates the size (number of pixels) for each class present
    in the input mask. It returns two arrays - an array containing the center of each class, and an array containing the
    number of pixels in each class. Ignores background as specified in background.

    Parameters
    ----------
    mask : np.ndarray
        Input mask containing classes.
    debug : bool, optional
        Debug flag (default is False).
    background : int, optional
        Value used to represent the background class (default is 0).

    Returns:
    -------
    mean_arr : np.ndarray
        Array containing the sum of the coordinates of each pixel for each class.
    length : np.ndarray
        Array containing the number of pixels in each class.

    Example:
    -------
    >>> import numpy as np
    >>> mask = np.array([[0, 1, 1], [0, 2, 1], [0, 0, 2]])
    >>> _class_size(mask)
    (array([[       nan,        nan],
            [0.33333333, 1.66666667],
            [1.5       , 1.5       ]]),
    array([nan,  3.,  2.]))

    """
    # Get the unique cell_ids and remove the background(0)
    cell_ids = list(np.unique(mask).flatten())
    if 0 in cell_ids:
        cell_ids.remove(background)
    cell_ids = np.array(cell_ids)

    min_cell_id = np.min(
        cell_ids
    )  # need to convert to array since numba min functions requires array as input not list
    # -1 important since otherwise the cell with the lowest id becomes 0 and is ignored (since 0 = background)

    # Adjust mask by subtracting the min_cell_id - 1 from non-zero elements
    if min_cell_id != 1:
        mask = _numba_subtract(mask, min_cell_id - 1)

    # Calculate the total number of classes
    num_classes = np.max(mask) + 1

    # Initialize an array to store the sum of the coordinates for each class
    mean_sum = np.zeros((num_classes, 2))
    #
    #  Initialize an array to store the number of pixels for each class
    length = np.zeros((num_classes, 1))

    # get dimensions of input mask
    rows, cols = mask.shape

    # Iterate through the rows and columns of the mask
    for row in range(rows):
        for col in range(cols):
            # get the class id at the current position
            return_id = mask[row, col]

            # Check if the current class ID is not equal to the background class
            if return_id != background:
                mean_sum[return_id] += np.array(
                    [row, col], dtype="uint32"
                )  # Add the coordinates to the corresponding class ID in mean_sum
                length[return_id][0] += 1  # Increment the number of pixels for the corresponding class ID in length

    # Divide the mean_sum array by the length array to get the mean_arr
    mean_arr = np.divide(mean_sum, length)

    # set background index to np.NaN
    length[background][0] = np.nan
    mean_arr[background] = np.nan

    return mean_arr, length.flatten()


def size_filter(label, limits=None, background=0, reindex=False):
    """Filter classes in a labeled array based on their size (number of pixels).

    This function filters classes in the input labeled array by removing classes
    that have a size (number of pixels) outside the provided limits. Optionally,
    it can reindex the remaining classes.

    Parameters
    ----------
    label : np.ndarray
        Input labeled array.
    limits : list, optional
        List containing the minimum and maximum allowed class size (number of pixels)
        (default is [0, 100000]).
    background : int, optional
        Value used to represent the background class (default is 0).
    reindex : bool, optional
        If True, reindexes the remaining classes after removal (default is False).

    Returns:
    -------
    label : np.ndarray
        Filtered labeled array.

    Example:
    -------
    >>> import numpy as np
    >>> label = np.array([[0, 1, 1], [0, 2, 1], [0, 0, 2]])
    >>> size_filter(label, limits=[1, 2])
    array([[0, 0, 0],
           [0, 2, 0],
           [0, 0, 2]])

    """
    # Calculate the number of pixels for each class in the labeled array
    if limits is None:
        limits = [0, 100000]
    _, points_class = _class_size(label)

    # Find the classes with size below the lower limit and above the upper limit
    below = np.argwhere(points_class < limits[0]).flatten()
    above = np.argwhere(points_class > limits[1]).flatten()

    # Combine the classes below and above the limits to create the list of classes to remove
    to_remove = list(below) + list(above)

    # Remove the specified classes, and optionally reindex the remaining classes
    if len(to_remove) > 0:
        label = remove_classes(label, nb.typed.List(to_remove), reindex=reindex)

    return label


#### Function to get centers of cells
@njit
def numba_mask_centroid(mask, debug=False, skip_background=True):
    """Calculate the centroids of classes in a given mask.

    This function calculates the centroids of each class in the mask and returns an array
    with the (y, x) coordinates of the centroids, the number of pixels associated with
    each class, and the id number of each class.

    Parameters
    ----------
    mask : numpy.ndarray
        Input mask containing classes.
    debug : bool, optional
        Debug flag (default is False).
    skip_background : bool, optional
        If True, skip background class (default is True).

    Returns:
    -------
    center : numpy.ndarray
        Array containing the (y, x) coordinates of the centroids of each class.
    points_class : numpy.ndarray
        Array containing the number of pixels associated with each class.
    ids : numpy.ndarray
        Array containing the id number of each class.

    Example:
    -------
    >>> import numpy as np
    >>> mask = np.array([[0, 1, 1], [0, 2, 1], [0, 0, 2]])
    >>> numba_mask_centroid(mask)
    (array([[0.33333333, 1.66666667],
            [1.5       , 1.5       ]]),
    array([3, 2], dtype=uint32),
    array([1, 2], dtype=int32))
    """
    # need to perform this adjustment here so that we can also work with segmentations that do not start with a seg index of 1!
    # this is relevant when working with segmentations that have been reindexed over different tiles

    # Get the unique cell_ids and remove the background (0)
    cell_ids = list(np.unique(mask).flatten())
    if 0 in cell_ids:
        cell_ids.remove(0)
    cell_ids = np.array(cell_ids)

    min_cell_id = np.min(
        cell_ids
    )  # need to convert to array since numba min functions requires array as input not list
    # -1 important since otherwise the cell with the lowest id becomes 0 and is ignored (since 0 = background)

    # Adjust mask by subtracting the min_cell_id - 1 from non-zero elements
    if min_cell_id != 1:
        mask = _numba_subtract(mask, min_cell_id - 1)

    if skip_background:
        num_classes = np.max(mask)
    else:
        num_classes = np.max(mask) + 1
    class_range = [0, num_classes]

    # Check if there's only background
    if class_range[1] == 0:
        print("no cells in image. Only contains background.")
        # return empty arrays
        return None, None, None

    num_classes = int(
        num_classes
    )  # add explicit conversion to int to ensure that numba can type the function correctly
    points_class = np.zeros((num_classes,), dtype=nb.uint32)
    center = np.zeros(
        (
            num_classes,
            2,
        )
    )
    ids = np.zeros((num_classes,))

    if skip_background:
        for y in range(len(mask)):
            for x in range(len(mask[0])):
                class_id = mask[y, x]
                if class_id != 0:
                    class_id -= 1
                    points_class[class_id] += 1
                    center[class_id] += np.array([x, y])
                    ids[class_id] = class_id + 1

    else:
        for y in range(len(mask)):
            for x in range(len(mask[0])):
                class_id = mask[y, x]
                points_class[class_id] += 1
                center[class_id] += np.array([x, y])
                ids[class_id] = class_id

    x = center[:, 0] / points_class
    y = center[:, 1] / points_class

    center = np.stack((y, x)).T

    if skip_background:
        if min_cell_id != 1:
            ids += min_cell_id - 1
    else:
        if min_cell_id != 1:
            ids[1:] += min_cell_id - 1  # leave the background at 0

    return center, points_class, ids.astype("int32")


#### Helper Numba functions to increase speed of numpy operations


# short-circuiting replacement for np.any()
@nb.jit(nopython=True)
def sc_any(array):
    """Short-circuiting replacement for np.any().

    Parameters
    ----------
    array : np.ndarray
        Input array to check if any values are True

    Returns:
    -------
    bool
        Boolean value indicating if expression evaluated to True or False
    """
    for x in array.flat:
        if x:
            return True
    return False


# short-circuiting replacement for np.all()
@nb.jit(nopython=True)
def sc_all(array):
    """Short-circuiting replacement for np.all().

    Parameters
    ----------
    array : np.ndarray
        Input array to check if all values are True

    Returns:
    -------
    bool
        Boolean value indicating if expression evaluated to True or False
    """
    for x in array.flat:
        if not x:
            return False
    return True

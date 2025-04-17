"""Functions for thresholding and segmenting images."""

import matplotlib.pyplot as plt
import numba as nb
import numpy as np
import skfmm
from numba import njit, prange
from numpy.typing import NDArray
from scipy import ndimage
from skimage import filters
from skimage.feature import peak_local_max
from skimage.morphology import binary_erosion, disk
from skimage.morphology import dilation as sk_dilation
from skimage.segmentation import watershed
from skimage.transform import resize

from scportrait.pipeline._utils.constants import DEFAULT_SEGMENTATION_DTYPE
from scportrait.plotting.vis import plot_image


def global_otsu(image: NDArray) -> float:
    """Calculate optimal global threshold using Otsu's method.

    Args:
        image: Input grayscale image

    Returns:
        Optimal threshold value calculated using Otsu's method

    Examples:
        >>> from skimage import data
        >>> image = data.coins()
        >>> threshold = global_otsu(image)
        >>> print(threshold)
    """
    counts, bin_edges = np.histogram(np.ravel(image), bins=512)
    bin_centers = bin_edges[:-1] + np.diff(bin_edges) / 2

    weight1 = np.cumsum(counts)
    weight2 = np.cumsum(counts[::-1])[::-1]

    mean1 = np.cumsum(counts * bin_centers) / weight1
    mean2 = (np.cumsum((counts * bin_centers)[::-1]) / weight2[::-1])[::-1]

    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

    idx = np.argmax(variance12)
    threshold = bin_centers[idx]

    return threshold


def _segment_threshold(
    image: NDArray,
    threshold: float,
    speckle_kernel: int = 4,
    debug: bool = False,
) -> NDArray:
    """Segment image using threshold and additional parameters.

    Args:
        image: Input grayscale image
        threshold: Threshold value for creating image_mask
        speckle_kernel: Size of structuring element for speckle removal
        debug: Enable debug visualization

    Returns:
        Binary mask after cleaning
    """
    image_mask = image > threshold

    if debug:
        plot_image(image_mask, cmap="Greys_r")

    image_mask_clean = binary_erosion(image_mask, footprint=disk(speckle_kernel))
    image_mask_clean = sk_dilation(image_mask_clean, footprint=disk(speckle_kernel - 1))

    return image_mask_clean


def _generate_labels_from_mask(
    image_mask: NDArray, dilation: int = 4, min_distance: int = 10, peak_footprint: int = 7, debug: bool = False
) -> NDArray:
    """Generate labels from binary mask using distance transform and watershed.

    Args:
        image_mask: Binary mask to generate labels from
        dilation: Size of dilation structuring element
        min_distance: Minimum distance between peaks
        peak_footprint: Size of peak finding structuring element
        debug: Enable debug visualization

    Returns:
        Label array where each segment has unique label
    """
    distance = ndimage.distance_transform_edt(image_mask)
    peak_idx = peak_local_max(distance, min_distance=min_distance, footprint=disk(peak_footprint))
    local_maxi = np.zeros_like(image_mask, dtype=bool)
    local_maxi[tuple(peak_idx.T)] = True

    if debug:
        plt.figure(frameon=False, figsize=(10, 10))
        plt.imshow(image_mask, cmap="Greys_r")
        plt.title("Generated Mask")
        plt.axis("off")
        plt.scatter(peak_idx[:, 1], peak_idx[:, 0], s=2, color="red")
        plt.show()

    dilated_mask = sk_dilation(image_mask, footprint=disk(dilation))

    fmm_marker = np.ones_like(dilated_mask)
    for center in peak_idx:
        fmm_marker[center[0], center[1]] = 0

    m = np.ma.masked_array(fmm_marker, np.logical_not(dilated_mask))
    distance_2 = skfmm.distance(m)

    if debug:
        plt.figure(frameon=False, figsize=(10, 10))
        plt.title("Distance Transform")
        plt.imshow(distance_2, cmap="viridis")
        plt.scatter(peak_idx[:, 1], peak_idx[:, 0], s=2, color="red")
        plt.axis("off")
        plt.show()

    marker = np.zeros_like(image_mask).astype(DEFAULT_SEGMENTATION_DTYPE)
    for i, center in enumerate(peak_idx):
        marker[center[0], center[1]] = i + 1

    labels = watershed(distance_2, marker, mask=dilated_mask)
    return labels.astype(DEFAULT_SEGMENTATION_DTYPE)


def segment_global_threshold(
    image: NDArray,
    dilation: int = 4,
    min_distance: int = 10,
    peak_footprint: int = 7,
    speckle_kernel: int = 4,
    debug: bool = False,
) -> NDArray:
    """Segment image using Otsu's method and watershed.

    Args:
        image: Input grayscale image
        dilation: Size of dilation structuring element
        min_distance: Minimum distance between peaks
        peak_footprint: Size of peak finding structuring element
        speckle_kernel: Size of speckle removal structuring element
        debug: Enable debug visualization

    Returns:
        Label array where each segment has unique label

    Example:
        >>> from skimage import data
        >>> coins = data.coins()
        >>> labels = segment_global_threshold(coins, dilation=5, min_distance=20)
    """
    threshold = global_otsu(image)

    mask = _segment_threshold(
        image,
        threshold,
        speckle_kernel=speckle_kernel,
        debug=debug,
    )

    labels = _generate_labels_from_mask(
        mask, dilation=dilation, min_distance=min_distance, peak_footprint=peak_footprint, debug=debug
    )

    return labels


def segment_local_threshold(
    image: NDArray,
    dilation: int = 4,
    thr: float = 0.01,
    median_block: int = 51,
    min_distance: int = 10,
    peak_footprint: int = 7,
    speckle_kernel: int = 4,
    median_step: int = 1,
    debug: bool = False,
) -> NDArray:
    """Segment image using local threshold method.

    Args:
        image: Input grayscale image
        dilation: Size of dilation structuring element
        thr: Value added to block median for threshold
        median_block: Size of local threshold block
        min_distance: Minimum distance between peaks
        peak_footprint: Size of peak finding structuring element
        speckle_kernel: Size of speckle removal structuring element
        median_step: Step size for downsampling
        debug: Enable debug visualization

    Returns:
        Label array where each segment has unique label

    Raises:
        ValueError: If speckle_kernel < 1
    """
    if speckle_kernel < 1:
        raise ValueError("speckle_kernel needs to be at least 1")

    # Downsample the input image
    downsampled_image = image[::median_step, ::median_step]

    # Calculate local threshold
    local_threshold = filters.threshold_local(downsampled_image, block_size=median_block, method="median", offset=-thr)

    # Resize the local threshold image
    local_threshold = resize(local_threshold, image.shape)

    # Segment using local threshold
    mask = _segment_threshold(
        image,
        local_threshold,
        speckle_kernel=speckle_kernel,
        debug=debug,
    )

    labels = _generate_labels_from_mask(
        mask, dilation=dilation, min_distance=min_distance, peak_footprint=peak_footprint, debug=debug
    )

    return labels


@njit
def _numba_subtract(array1: NDArray, number: int) -> NDArray:
    """Subtract a number from all non-zero elements of array.

    Args:
        array1: Input array
        number: Number to subtract from non-zero elements

    Returns:
        Array with number subtracted from non-zero elements

    Example:
        >>> array1 = np.array([[0, 2, 3], [0, 5, 6], [0, 0, 7]])
        >>> _numba_subtract(array1, 1)
        array([[0, 1, 2],
               [0, 4, 5],
               [0, 0, 6]])
    """
    for i in range(array1.shape[0]):
        for j in range(array1.shape[1]):
            if array1[i, j] != 0:
                array1[i, j] = array1[i, j] - number
    return array1


@njit
def _return_edge_labels_2d(input_map: NDArray) -> list[int]:
    """Get labels touching edges in 2D array."""
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

    full_union = set([np.uint64(i) for i in full_union])  # noqa: C403
    full_union.discard(0)

    return list(full_union)


def _return_edge_labels(input_map: NDArray) -> list[int]:
    """Get labels touching edges in 2D/3D array.

    Args:
        input_map: Input segmentation array (2D or 3D)

    Returns:
        List of unique labels touching edges
    """
    if len(input_map.shape) == 3:
        full_union: list[int] = []
        for i in range(input_map.shape[0]):
            _ids = _return_edge_labels_2d(input_map[i])
            full_union.extend(_ids)
    else:
        full_union = _return_edge_labels_2d(input_map)

    return list(full_union)


def remove_edge_labels(input_map: NDArray) -> NDArray:
    """Remove labels touching edges of array.

    Args:
        input_map: Input segmentation array (2D or 3D)

    Returns:
        Array with edge-touching labels removed
    """
    edge_labels = _return_edge_labels(input_map)
    cleaned_map = np.where(np.isin(input_map, edge_labels), 0, input_map)
    return cleaned_map


def shift_labels(
    input_map: NDArray,
    shift: int,
    return_shifted_labels: bool = False,
    remove_edge_labels: bool = True,
    dtype=np.uint64,
) -> tuple[NDArray, list[int]]:
    """Shift label values by specified amount.

    Args:
        input_map: Input segmentation array
        shift: Value to increment labels by
        return_shifted_labels: Return edge labels after shifting
        remove_edge_labels: Remove edge-touching labels

    Returns:
        Tuple containing:
            - Array with shifted labels
            - List of edge-touching labels

    Example:
        >>> input_map = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
        >>> shifted_map, edge_labels = shift_labels(input_map, 10)
        >>> print(shifted_map)
        array([[11,  0,  0],
            [ 0, 12,  0],
            [ 0,  0, 13]])
        >>> print(edge_labels)
        [11, 13]
    """
    imap = input_map.copy()
    shifted_map = np.where(imap == 0, 0, imap + shift)

    edge_label: list[int] = []

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

    return shifted_map.astype(dtype), list(set(edge_label))


@njit(parallel=True)
def _remove_classes(label_in: NDArray, to_remove: list[int], background: int = 0, reindex: bool = False) -> NDArray:
    """Remove specified classes from labeled array.

    Args:
        label_in: Input labeled array
        to_remove: List of labels to remove
        background: Background label value
        reindex: Reindex remaining labels consecutively

    Returns:
        Array with specified classes removed/reindexed

    Example:
        >>> label_in = np.array([[1, 2, 1], [1, 0, 2], [0, 2, 3]])
        >>> _remove_classes(label_in, [1, 3])
        array([[0, 2, 0],
               [0, 0, 2],
               [0, 2, 0]])
    """
    label = label_in.copy()
    remove_set = set(to_remove)
    library = np.zeros(np.max(label) + 1, dtype="int")

    carry = 0
    for current_class in range(len(library)):
        if current_class in remove_set:
            library[current_class] = background
            if reindex:
                carry -= 1
        else:
            library[current_class] = current_class + carry

    for y in prange(len(label)):
        for x in prange(len(label[0])):
            current_label = label[y, x]
            if current_label != background:
                label[y, x] = library[current_label]

    return label


def remove_classes(label_in: NDArray, to_remove: list[int], background: int = 0, reindex: bool = False) -> NDArray:
    """Remove specified classes from labeled array.

    Args:
        label_in: Input labeled array
        to_remove: List of labels to remove
        background: Background label value
        reindex: Reindex remaining labels consecutively

    Returns:
        Array with specified classes removed/reindexed
    """
    return _remove_classes(label_in, to_remove, background=background, reindex=reindex)


@njit(parallel=True)
def contact_filter_lambda(label: NDArray, background: int = 0) -> NDArray:
    """Calculate contact proportion of classes with background.

    Args:
        label: Input labeled array
        background: Background label value

    Returns:
        Array of background contact proportions for each class

    Example:
        >>> label = np.array([[0, 1, 1], [0, 2, 1], [0, 0, 2]])
        >>> contact_filter_lambda(label)
        array([1.0, 1.0, 0.6])
    """
    num_labels = np.max(label)
    background_matrix = np.ones((num_labels + 1, 2), dtype="int")

    for y in range(1, len(label) - 1):
        for x in range(1, len(label[0]) - 1):
            current_label = label[y, x]

            background_matrix[current_label, 0] += int(label[y - 1, x] == background)
            background_matrix[current_label, 0] += int(label[y, x - 1] == background)
            background_matrix[current_label, 0] += int(label[y + 1, x] == background)
            background_matrix[current_label, 0] += int(label[y, x + 1] == background)

            background_matrix[current_label, 1] += int(label[y - 1, x] != current_label)
            background_matrix[current_label, 1] += int(label[y, x - 1] != current_label)
            background_matrix[current_label, 1] += int(label[y + 1, x] != current_label)
            background_matrix[current_label, 1] += int(label[y, x + 1] != current_label)

    proportion = background_matrix[:, 0] / background_matrix[:, 1]
    return proportion


def contact_filter(inarr: NDArray, threshold: float = 1.0, reindex: bool = False, background: int = 0) -> NDArray:
    """Filter array based on label contact with background.

    Args:
        inarr: Input labeled array
        threshold: Minimum background contact proportion for retention
        reindex: Reindex remaining labels consecutively
        background: Background label value

    Returns:
        Filtered labeled array

    Example:
        >>> inarr = np.array([[0, 1, 1], [0, 2, 1], [0, 0, 2]])
        >>> contact_filter(inarr, threshold=0.5)
        array([[0, 1, 1],
               [0, 0, 1],
               [0, 0, 0]])
    """
    label = inarr.copy()
    background_contact = contact_filter_lambda(label, background=background)
    to_remove = np.argwhere(background_contact < threshold).flatten()
    to_remove = np.delete(to_remove, np.where(to_remove == background))

    if len(to_remove) > 0:
        label = remove_classes(label, nb.typed.List(to_remove), reindex=reindex)

    return label


@njit
def _class_size(mask: NDArray, debug: bool = False, background: int = 0) -> tuple[NDArray, NDArray]:
    """Compute size of each class in mask.

    Args:
        mask: Input mask containing classes
        debug: Debug flag
        background: Background label value

    Returns:
        Tuple containing:
            - Array of class centroids
            - Array of class sizes

    Example:
        >>> mask = np.array([[0, 1, 1], [0, 2, 1], [0, 0, 2]])
        >>> _class_size(mask)
        (array([[       nan,        nan],
                [0.33333333, 1.66666667],
                [1.5       , 1.5       ]]),
        array([nan,  3.,  2.]))
    """
    cell_ids = list(np.unique(mask).flatten())
    if 0 in cell_ids:
        cell_ids.remove(background)
    cell_ids = np.array(cell_ids)

    min_cell_id = np.min(cell_ids)

    if min_cell_id != 1:
        mask = _numba_subtract(mask, min_cell_id - 1)

    num_classes = np.max(mask) + 1
    mean_sum = np.zeros((num_classes, 2))
    length = np.zeros((num_classes, 1))
    rows, cols = mask.shape

    for row in range(rows):
        for col in range(cols):
            return_id = mask[row, col]
            if return_id != background:
                mean_sum[return_id] += np.array([row, col], dtype=np.uint32)
                length[return_id][0] += 1

    mean_arr = np.divide(mean_sum, length)
    length[background][0] = np.nan
    mean_arr[background] = np.nan

    return mean_arr, length.flatten()


def size_filter(label: NDArray, limits: list[int] | None = None, reindex: bool = False) -> NDArray:
    """Filter classes based on size.

    Args:
        label: Input labeled array
        limits: [min_size, max_size] in pixels
        reindex: Reindex remaining labels consecutively

    Returns:
        Filtered labeled array

    Example:
        >>> label = np.array([[0, 1, 1], [0, 2, 1], [0, 0, 2]])
        >>> size_filter(label, limits=[1, 2])
        array([[0, 0, 0],
               [0, 2, 0],
               [0, 0, 2]])
    """
    if limits is None:
        limits = [0, 100000]

    _, points_class = _class_size(label)

    below = np.argwhere(points_class < limits[0]).flatten()
    above = np.argwhere(points_class > limits[1]).flatten()
    to_remove = list(below) + list(above)

    if len(to_remove) > 0:
        label = remove_classes(label, nb.typed.List(to_remove), reindex=reindex)

    return label


@njit
def numba_mask_centroid(
    mask: NDArray, skip_background: bool = True
) -> tuple[NDArray | None, NDArray | None, NDArray | None]:
    """Calculate centroids of labeled regions.

    Args:
        mask: Input mask containing classes
        skip_background: Skip background class calculation

    Returns:
        Tuple containing:
            - Array of (y,x) centroid coordinates
            - Array of region sizes
            - Array of region IDs
        Returns (None, None, None) if no cells found

    Example:
        >>> mask = np.array([[0, 1, 1], [0, 2, 1], [0, 0, 2]])
        >>> centers, sizes, ids = numba_mask_centroid(mask)
    """
    cell_ids = np.unique(mask).flatten()
    if skip_background:
        if 0 in cell_ids:
            cell_ids = cell_ids[1:]

    min_cell_id = np.min(cell_ids)

    if min_cell_id == 0:
        print("no cells in image. Only contains background with value 0.")
        return None, None, None

    num_classes = int(len(cell_ids) if skip_background else len(cell_ids) + 1)

    points_class = np.zeros((num_classes,), dtype=nb.uint64)
    center = np.zeros((num_classes, 2))
    ids = np.full((num_classes,), np.nan)

    if skip_background:
        for y in range(len(mask)):
            for x in range(len(mask[0])):
                class_id = mask[y, x]
                if class_id != 0:
                    idx = np.where(cell_ids == class_id)[0][0]
                    points_class[idx] += 1
                    center[idx] += np.array([x, y])
                    ids[idx] = class_id
    else:
        for y in range(len(mask)):
            for x in range(len(mask[0])):
                class_id = mask[y, x]
                idx = np.where(cell_ids == class_id)[0][0]
                points_class[idx] += 1
                center[idx] += np.array([x, y])
                ids[idx] = class_id

    x = center[:, 0] / points_class
    y = center[:, 1] / points_class
    center = np.stack((y, x)).T

    # Remove background and invalid IDs
    valid_mask = ~np.isnan(ids)
    center = center[valid_mask]
    points_class = points_class[valid_mask]
    ids = ids[valid_mask]

    bg_mask = ids != 0
    center = center[bg_mask]
    points_class = points_class[bg_mask]
    ids = ids[bg_mask]

    return center, points_class, ids.astype(DEFAULT_SEGMENTATION_DTYPE)


@nb.jit(nopython=True)
def sc_any(array: NDArray) -> bool:
    """Short-circuiting replacement for np.any().

    Args:
        array: Input array to check

    Returns:
        True if any element is True, False otherwise
    """
    for x in array.flat:
        if x:
            return True
    return False


@nb.jit(nopython=True)
def sc_all(array: NDArray) -> bool:
    """Short-circuiting replacement for np.all().

    Args:
        array: Input array to check

    Returns:
        True if all elements are True, False otherwise
    """
    for x in array.flat:
        if not x:
            return False
    return True


def remap_mask(input_mask: np.ndarray) -> np.ndarray:
    """Remap mask to have consecutive labels starting from 1.

    Args:
        input_mask: Input mask with non-consecutive labels

    Returns:
        Remapped mask with consecutive labels

    Example:
        >>> input_mask = np.array([[5, 6, 6], [5, 3, 6], [5, 5, 3]])
        >>> remap_mask(input_mask)
        array([[2, 3, 3],
               [2, 1, 3],
               [2, 2, 1]])
    """
    # Create lookup table as an array
    max_label = np.max(input_mask)
    lookup_array = np.zeros(max_label + 1, dtype=np.int32)

    cell_ids = np.unique(input_mask)[1:]
    lookup_table = dict(zip(cell_ids, range(1, len(cell_ids) + 1), strict=True))

    # Populate lookup array based on the dictionary
    for old_id, new_id in lookup_table.items():
        lookup_array[old_id] = new_id

    # Apply the mapping using NumPy indexing
    remapped_mask = lookup_array[input_mask]

    return remapped_mask

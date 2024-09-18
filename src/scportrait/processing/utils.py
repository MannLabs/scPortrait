import matplotlib.pyplot as plt
import numpy as np
from skimage.color import label2rgb
import os


def plot_image(
    array, size=(10, 10), save_name="", cmap="magma", return_fig=False, **kwargs
):
    """
    Visualize and optionally save an input array as an image.

    This function displays a 2D array using matplotlib and can save
    the resulting image as a PNG file.

    Args:
        array (np.array): Input 2D numpy array to be plotted.
        size (tuple of int, optional): Figure size in inches, by default (10, 10).
        save_name (str, optional): Name of the output file, without extension. If not provided, image will not be saved, by default "".
        cmap (str, optional): Color map used to display the array, by default "magma".
        **kwargs: Additional keyword arguments to be passed to `ax.imshow`.

    Returns:
        None: The function will display the image but does not return any values.

    Example:
    >>> array = np.random.rand(10, 10)
    >>> plot_image(array, size=(5, 5))
    """

    fig = plt.figure(frameon=False)
    fig.set_size_inches(size)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(array, cmap=cmap, **kwargs)

    if return_fig:
        return fig

    if save_name != "":
        plt.savefig(save_name + ".png")
        plt.show()
        plt.close()
    else:
        plt.show()
        plt.close()


def visualize_class(
    class_ids, seg_map, image, all_ids=None, return_fig=False, *args, **kwargs
):
    """
    Visualize specific classes in a segmentation map by highlighting them on top of a background image.

    This function takes in class IDs and a segmentation map, and creates an output visualization
    where the specified classes are highlighted on top of the provided background image.

    Args:
        class_ids (array-like): A list or array of integers representing the class IDs to be highlighted.
        seg_map (2D array-like): A 2D array representing the segmentation map, where each value corresponds to a class ID.
        image (2D/3D array-like): Background image (2D or 3D) on which the classes will be highlighted. Its size should match that of `seg_map`.
        *args: Any additional positional arguments that are passed to the underlying plotting functions.
        **kwargs: Any additional keyword arguments that are passed underlying plotting functions.

    Returns:
        None: The function will display the highlighted image but does not return any values.

    Example:
    >>> class_ids = [1, 2]
    >>> seg_map = np.array([[0, 1, 0], [1, 2, 1], [2, 0, 1]])
    >>> background = np.random.random((3, 3)) * 255
    >>> visualize_class(class_ids, seg_map, background)
    """

    # ensure the class ids are a list
    if not isinstance(class_ids, list):
        class_ids = list(class_ids)

    if all_ids is None:
        all_ids = set(np.unique(seg_map)) - set([0])

    # get the ids to keep
    keep_ids = list(all_ids - set(class_ids))

    mask_discard = np.isin(seg_map, keep_ids)
    mask_keep = np.isin(seg_map, class_ids)

    # Set the values in the output map to 2 for the specified class IDs
    outmap = np.where(mask_discard, 2, seg_map)

    # Set the values in the output map to 1 for all class IDs other than the specified classes
    outmap = np.where(mask_keep, 1, outmap)

    vis_map = label2rgb(
        outmap, image=image, colors=["red", "blue"], alpha=0.4, bg_label=0
    )

    fig = plot_image(vis_map, return_fig=True, *args, **kwargs)

    if return_fig:
        return fig


def download_testimage(folder):
    """
    Download a set of test images to a provided folder path.

    This function downloads a set of test images from Zenodo and saves them to a provided folder path.

    Args:
        folder (str): The path of the folder where the test images will be saved.

    Returns:
        returns (list): A list containing the local file paths of the downloaded images.

    Example:
    >>> folder = "test_images"
    >>> downloaded_images = download_testimage(folder)
    Successfully downloaded testimage_dapi.tiff from https://zenodo.org/record/5701474/files/testimage_dapi.tiff?download=1
    Successfully downloaded testimage_wga.tiff from https://zenodo.org/record/5701474/files/testimage_wga.tiff?download=1
    >>> print(downloaded_images)
    ['test_images/testimage_dapi.tiff', 'test_images/testimage_wga.tiff']
    """

    # Define the test images' names and URLs
    images = [
        (
            "testimage_dapi.tiff",
            "https://zenodo.org/record/5701474/files/testimage_dapi.tiff?download=1",
        ),
        (
            "testimage_wga.tiff",
            "https://zenodo.org/record/5701474/files/testimage_wga.tiff?download=1",
        ),
    ]

    import urllib.request

    returns = []
    for name, url in images:
        # Construct the local file path for the current test image
        path = os.path.join(folder, name)

        # Open the local file and write the contents of the test image URL
        f = open(path, "wb")
        f.write(urllib.request.urlopen(url).read())
        f.close()

        # Print a message confirming the download and add the local file path to the output list
        print(f"Successfully downloaded {name} from {url}")
        returns.append(path)
    return returns


def flatten(list):
    """
    Flatten a list of lists into a single list.

    This function takes in a list of lists (nested lists) and returns a single list
    containing all the elements from the input lists.

    Args:
        list (list of lists): A list containing one or more lists as its elements.

    Returns:
        flattened_list (list): A single list containing all elements from the input lists.

    Example:
    >>> nested_list = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
    >>> flatten(nested_list)
    [1, 2, 3, 4, 5, 6, 7, 8, 9]
    """
    # Flatten the input list using list comprehension
    return [item for sublist in list for item in sublist]

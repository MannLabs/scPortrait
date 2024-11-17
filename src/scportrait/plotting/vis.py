import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import label2rgb


def plot_image(array, size=(10, 10), save_name="", cmap="magma", return_fig=False, **kwargs):
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


def visualize_class(class_ids, seg_map, image, all_ids=None, return_fig=False, **kwargs):
    """
    Visualize specific classes in a segmentation map by highlighting them on top of a background image.

    This function takes in class IDs and a segmentation map, and creates an output visualization
    where the specified classes are highlighted on top of the provided background image.

    Args:
        class_ids (array-like): A list or array of integers representing the class IDs to be highlighted.
        seg_map (2D array-like): A 2D array representing the segmentation map, where each value corresponds to a class ID.
        background (2D/3D array-like): Background image (2D or 3D) on which the classes will be highlighted. Its size should match that of `seg_map`.
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
        all_ids = set(np.unique(seg_map)) - {0}

    # get the ids to keep
    keep_ids = list(all_ids - set(class_ids))

    mask_discard = np.isin(seg_map, keep_ids)
    mask_keep = np.isin(seg_map, class_ids)

    # Set the values in the output map to 2 for the specified class IDs
    outmap = np.where(mask_discard, 2, seg_map)

    # Set the values in the output map to 1 for all class IDs other than the specified classes
    outmap = np.where(mask_keep, 1, outmap)

    vis_map = label2rgb(outmap, image=image, colors=["red", "blue"], alpha=0.4, bg_label=0)

    fig = plot_image(vis_map, return_fig=True, **kwargs)

    if return_fig:
        return fig


def plot_segmentation_mask(
    project,
    mask_channel=0,
    image_channel=0,
    selection=None,
    cmap_image="Greys_r",
    cmap_masks="prism",
    alpha=0.5,
    figsize=(10, 10),
):
    """
    Visualize the segmentation mask overlayed with a channel of the input image.

    Parameters
    ----------
    project : scPortrait.pipeline.project.Project
        instance of a scPortrait project.
    mask_channel : int, optional
        The index of the channel to use for the segmentation mask (default: 0).
    image_channel : int, optional
        The index of the channel to use for the image (default: 0).
    selection : tuple(slice, slice), optional
        The selection coordinates for a specific region of interest (default: None).
    cmap_image : str, optional
        The colormap to use for the input image (default: "Greys_r").
    cmap_masks : str, optional
        The colormap to use for the segmentation mask (default: "prism").
    alpha : float, optional
        The transparency level of the segmentation mask (default: 0.5).

    Returns
    -------
    fig : object
        The generated figure object.
    """
    # integer value indicating background (default value)
    background = 0

    segmentation_file = os.path.join(project.seg_directory, project.segmentation_f.DEFAULT_SEGMENTATION_FILE)

    with h5py.File(segmentation_file, "r") as hf:
        segmentation = hf.get(project.segmentation_f.DEFAULT_MASK_NAME)
        channels = hf.get(project.segmentation_f.DEFAULT_CHANNELS_NAME)

        if selection is None:
            segmentation = segmentation[mask_channel, :, :]
            image = channels[image_channel, :, :]
        else:
            segmentation = segmentation[mask_channel, selection[0], selection[1]]
            image = channels[image_channel, selection[0], selection[1]]

    # set background to np.nan so that its not visualized
    segmentation = np.where(segmentation == background, np.nan, segmentation)

    fig = plt.figure(figsize=figsize)
    plt.imshow(image, cmap=cmap_image)
    plt.imshow(segmentation, alpha=alpha, cmap=cmap_masks)
    plt.axis("off")
    return fig

def colorize(im: np.ndarray,
             color: tuple[int, ...] = (1, 0, 0),
             clip_percentile: float =0.0):
    """
    Helper function to create an RGB image from a single-channel image using a
    specific color.

    Args:
        im (np.ndarray): A single-channel image.
        color (tuple[int, ...], optional): The color to use for the image. Defaults to (1, 0, 0).
        clip_percentile (float, optional): The percentile to clip the image at. Defaults to 0.0.

    Returns:
        np.ndarray: The colorized image.
    """
    # Check that we do just have a 2D image
    if im.ndim > 2 and im.shape[2] != 1:
        raise ValueError("This function expects a single-channel image!")

    # Rescale the image according to how we want to display it
    im_scaled = im.astype(np.float32) - np.percentile(im, clip_percentile)
    im_scaled = im_scaled / np.percentile(im_scaled, 100 - clip_percentile)
    im_scaled = np.clip(im_scaled, 0, 1)

    # Need to make sure we have a channels dimension for the multiplication to work
    im_scaled = np.atleast_3d(im_scaled)

    # Reshape the color (here, we assume channels last)
    color = np.asarray(color).reshape((1, 1, -1))
    return im_scaled * color

def generate_composite(images: np.ndarray,
                       colors: list[tuple[int, ...]] = None,
                       plot: bool =False):
    """Create a composite image from a multi-channel image for visualization.

    Args:
        images (np.ndarray): A multi-channel image to be combined.
        colors (list[tuple[int, ...]], optional): A list of colors to use for each channel. Defaults to None.
        plot (bool, optional): Whether to plot the composite image. Defaults to False.

    Returns:
        np.ndarray: The composite image.
    """
    if colors is None:
        colors = [(0, 0, 1), (0, 1, 0), (1, 0, 0), (1, 0, 1)]
    colorized = []
    for image, color in zip(images, colors, strict=False):
        image = colorize(image, color, 0.0)
        colorized.append(image)

    if plot:
        for i in colorized:
            plt.figure()
            plt.imshow(i)

    image = colorized[0]
    for i in range(len(colorized) - 1):
        image += colorized[i + 1]

    return np.clip(image, 0, 1)

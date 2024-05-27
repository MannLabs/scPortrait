import h5py
import matplotlib.pyplot as plt
import os


def plot_segmentation_mask(
    project,
    mask_channel=0,
    image_channel=0,
    selection=None,
    cmap_image="Greys_r",
    cmap_masks="jet",
    alpha=0.5,
    figsize = (10, 10)
):
    """
    Visualize the segmentation mask overlayed with a channel of the input image.

    Parameters
    ----------
    project : sparcspy.pipeline.project.Project
        instance of a sparcspy project.
    mask_channel : int, optional
        The index of the channel to use for the segmentation mask (default: 0).
    image_channel : int, optional
        The index of the channel to use for the image (default: 0).
    selection : tuple(slice, slice), optional
        The selection coordinates for a specific region of interest (default: None).
    cmap_image : str, optional
        The colormap to use for the input image (default: "Greys_r").
    cmap_masks : str, optional
        The colormap to use for the segmentation mask (default: "jet").
    alpha : float, optional
        The transparency level of the segmentation mask (default: 0.5).

    Returns
    -------
    fig : object
        The generated figure object.
    """
    segmentation_file = os.path.join(
        project.seg_directory, project.segmentation_f.DEFAULT_OUTPUT_FILE
    )

    with h5py.File(segmentation_file, "r") as hf:
        segmentation = hf.get(project.segmentation_f.DEFAULT_MASK_NAME)
        channels = hf.get(project.segmentation_f.DEFAULT_CHANNELS_NAME)

        if selection is None:
            segmentation = segmentation[mask_channel, :, :]
            image = channels[image_channel, :, :]
        else:
            segmentation = segmentation[mask_channel, selection[0], selection[1]]
            image = channels[image_channel, selection[0], selection[1]]

    fig = plt.figure(figsize=figsize)
    plt.imshow(image, cmap=cmap_image)
    plt.imshow(segmentation, alpha=alpha, cmap=cmap_masks)
    plt.axis("off")
    return fig

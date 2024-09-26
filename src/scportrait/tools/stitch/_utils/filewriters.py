import os
import shutil
from typing import List, Tuple

import numpy as np
import zarr
from ome_zarr.io import parse_url
from ome_zarr.scale import Scaler
from ome_zarr.writer import write_image
from spatialdata import SpatialData
from spatialdata.models import Image2DModel
from spatialdata.transformations.transformations import Identity
from tifffile import imsave
from yattag import Doc, indent


def write_tif(image_path: str, image: np.array, dtype="uint16"):
    """_summary_

    Parameters
    ----------
    image_path : str
        absolute file path where the image should be written
    image : np.array
        image that should be saved to disk
    dtype : str, optional
        datatype to save the image as, by default "uint16"
    """
    # save using tifffile library to ensure compatibility with very large tif files
    imsave(image_path, image.astype(dtype))


def write_ome_zarr(
    filepath: str,
    image: np.array,
    channels: List[str],
    slidename: str,
    channel_colors: List[str] = None,
    downscaling_size: int = 4,
    n_downscaling_layers: int = 4,
    chunk_size: Tuple[int, int, int] = (1, 1024, 1024),
    overwrite: bool = False,
):
    """write out an image as an OME-Zarr file compatible with napari

    Parameters
    ----------
    filepath : str
        absolute path to the output file where the image should be written
    image : np.array
        image to write out
    channels : [str]
        list of channel names as strings, order needs to be the same as in the passed numpy image
    slidename : str
        string indicating the name of the slide
    channel_colors : [str], optional
        if desired a list of hex colors can be passed which will be used to set each channel color in napar
        the order needs to be as specified in `channels`
        default value is None, then a set of default colors will be used
    downscaling_size : int, optional
        _description_, by default 4
    n_downscaling_layers : int, optional
        _description_, by default 4
    chunk_size : int, int, int, optional
        _description_, by default (1, 1024, 1024)
    overwrite : bool, optional
        boolean value indicating if the outfile should be overwritten if it already exists, by default False

    """
    # delete file if it already exists
    if os.path.isdir(filepath):
        if overwrite:
            shutil.rmtree(filepath)
            print(f"Outfile {filepath} already existed and was deleted.")
        else:
            raise ValueError(f"Outfile {filepath} already exists. Set overwrite to True to delete it.")

    # setup channel colors
    if channel_colors is None:
        # hard coded channel colors for napari that we fall back on if none are specified
        channel_colors = [
            "#e60049",
            "#0bb4ff",
            "#50e991",
            "#e6d800",
            "#9b19f5",
            "#ffa300",
            "#dc0ab4",
            "#b3d4ff",
            "#00bfa0",
        ]

        # ensure that list is as long as number of channels (we just cycle through the colors)
        while len(channels) > len(channel_colors):
            channel_colors = channel_colors + channel_colors
    else:
        if len(channel_colors) != len(channels):
            raise ValueError("Number of channel colors does not match number of channels")

    # setup zarr group to initialize saving process
    loc = parse_url(filepath, mode="w").store
    group = zarr.group(store=loc)
    axes = "cyx"

    group.attrs["omero"] = {
        "name": slidename + ".ome.zarr",
        "channels": [
            {"label": channel, "color": channel_colors[i], "active": True} for i, channel in enumerate(channels)
        ],
    }

    scaler = Scaler(
        copy_metadata=False,
        downscale=downscaling_size,
        in_place=False,
        labeled=False,
        max_layer=n_downscaling_layers,
        method="nearest",
    )  # increase downscale so that large slides can also be opened in napari
    write_image(image, group=group, axes=axes, storage_options={"chunks": chunk_size}, scaler=scaler)


def write_xml(image_paths: List[str], channels: List[str], slidename: str, outdir: str = None):
    """Helper function to generate an XML for import of stitched .tifs into BIAS.

    Parameters
    ----------
    image_path : [str]
        List of absolute paths to the tif files that should be included in the XML.
    channels : [str]
        list of the channel names as they should be labelled in the XML (importat: keep the same order as in the TIF list)
    slidename : str
        string indicating the base name of the stitched images. The XML will be saved as slidename.XML
    outpath : str
        optional string indicating the path to save the XML file. If None, the XML will be saved in the same directory as the images.

    Returns
    -------
    None
        a file is written out.
    """

    # perform some basic sanity checking before proceeding

    # 1. check that the number of channels matches the number of images
    if len(image_paths) != len(channels):
        raise ValueError("Number of channels does not match the number of images")

    # 2. ensure that each channel name is included in each image name
    # this should also ensure that the tif oder and the channel order match
    for i, image_path in enumerate(image_paths):
        if channels[i] not in image_path:
            raise ValueError(f"Channel {channels[i]} is not included in image {image_path}")

    # generate the XML document for writing out
    doc, tag, text = Doc().tagtext()

    xml_header = '<?xml version="1.0" encoding="UTF-8"?>'
    doc.asis(xml_header)
    with tag("BIAS", version="1.0"):
        with tag("channels"):
            for i, channel in enumerate(channels):
                with tag("channel", id=str(i + 1)):
                    with tag("name"):
                        text(channel)
        with tag("images"):
            for i, image_path in enumerate(image_paths):
                with tag("image", url=str(image_path)):
                    with tag("channel"):
                        text(str(i + 1))

    result = indent(doc.getvalue(), indentation=" " * 4, newline="\r\n")

    # write to file
    if outdir is None:
        outdir = os.path.commonpath(image_paths)
    write_path = os.path.join(outdir, slidename + ".XML")

    with open(write_path, mode="w") as f:
        f.write(result)


def write_spatialdata(
    image_path: str,
    image: np.array,
    channel_names: List[str] = None,
    scale_factors: List[int] = None,
    overwrite: bool = False,
):
    # check if the file exists and delete if overwrite is set to True
    if scale_factors is None:
        scale_factors = [2, 4, 8]
    if os.path.exists(image_path):
        if overwrite:
            shutil.rmtree(image_path)
        else:
            raise ValueError(f"File {image_path} already exists. Set overwrite to True to delete it.")

    # check channel names
    if channel_names is None:
        channel_names = [f"channel_{i}" for i in range(image.shape[0])]
    else:
        if len(channel_names) != image.shape[0]:
            raise ValueError("Number of channel names does not match number of channels in image")

    # setup transforms
    transform_original = Identity()

    # convert image to an Image2DModel
    image = Image2DModel.parse(
        image,
        dims=["c", "y", "x"],
        c_coords=channel_names,
        scale_factors=scale_factors,
        transformations={"global": transform_original},
        rgb=False,
    )

    sdata = SpatialData(images={"input_image": image})  # memory bottleneck?
    sdata.write(image_path)

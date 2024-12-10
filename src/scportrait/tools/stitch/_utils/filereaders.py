import os

import numpy as np
import skimage.exposure
from ashlar import filepattern
from ashlar.filepattern import FilePatternReader
from ashlar.reg import BioformatsMetadata, BioformatsReader
from skimage.filters import gaussian
from skimage.util import invert

from scportrait.processing.images._image_processing import rescale_image


class FilePatternReaderRescale(FilePatternReader):
    """Class for reading images based on a file pattern. If desired the images can be rescaled to a certain range while reading."""

    def __init__(
        self,
        path,
        pattern,
        overlap,
        pixel_size=1,
        do_rescale=False,
        WGAchannel=None,
        no_rescale_channel="Alexa488",
        rescale_range=(1, 99),
    ):
        try:
            super().__init__(path, pattern, overlap, pixel_size=pixel_size)
        except (OSError, FileNotFoundError):
            print(
                f"Error: Could not read images with the given pattern {pattern}. Please check the path {path} and pattern."
            )
            found_files = os.listdir(path)
            print(
                f"At the provided location the the files follow the naming convention:{found_files[0:max(5, len(found_files))]} "
            )

        self.do_rescale = do_rescale
        self.WGAchannel = WGAchannel
        self.no_rescale_channel = no_rescale_channel
        self.rescale_range = rescale_range

    @staticmethod
    def rescale(img, rescale_range=(1, 99), cutoff_threshold=None):
        return rescale_image(img, rescale_range, cutoff_threshold=cutoff_threshold)

    @staticmethod
    # placeholer method kept for compatibility with old code
    # should be reimplemented in the future to allow for more flexible illumination correction
    def correct_illumination(img, sigma=30, double_correct=False, rescale_range=(1, 99), cutoff_threshold=None):
        img = rescale_image(img, rescale_range, cutoff_threshold=cutoff_threshold, return_float=True)

        # calculate correction mask
        correction = gaussian(img, sigma)
        correction = invert(correction)
        correction = skimage.exposure.rescale_intensity(correction, out_range=(0, 1))

        correction_lows = np.where(img > 0.5, 0, img) * correction
        img_corrected = skimage.exposure.rescale_intensity(img + correction_lows, out_range=(0, 1))

        if double_correct:
            correction_mask_highs = invert(correction)
            correction_mask_highs_02 = skimage.exposure.rescale_intensity(
                np.where(img_corrected < 0.5, 0, img_corrected) * correction_mask_highs
            )
            img_corrected_double = skimage.exposure.rescale_intensity(img_corrected - 0.25 * correction_mask_highs_02)

            return (img_corrected_double * 65535).astype("uint16")
        else:
            return (img_corrected * 65535).astype("uint16")

    def read(self, series, c):
        img = super().read(series, c)

        if not self.do_rescale:
            return img
        else:
            if c not in self.no_rescale_channel:
                # get rescale_range for channel c
                if isinstance(self.rescale_range, dict):
                    rescale_range = self.rescale_range[c]
                else:
                    rescale_range = self.rescale_range

                # actually read the image with rescaling applied
                return self.rescale(img, rescale_range=rescale_range)
            else:
                return img


class BioformatsMetadataRescale(BioformatsMetadata):
    """Reimplementation of BioformatsMetadata class to provide same parameters as contained in FilePatternReaderRescale"""

    def __init__(self, path):
        super().__init__(path)

    @property
    def channel_map(self):
        n_channels = self._metadata.getChannelCount(0)
        channel_names = []

        for id in range(n_channels):
            channel_names.append(self._metadata.getChannelName(0, id))
        channel_map = dict(zip(list(range(n_channels)), channel_names, strict=False))

        return channel_map


class BioformatsReaderRescale(BioformatsReader):
    """Class for reading images from Bioformats files (e.g. nd2). If desired the images can be rescaled to a certain range while reading."""

    def __init__(self, path, plate=None, well=None, do_rescale=False, no_rescale_channel=None, rescale_range=(1, 99)):
        self.path = path
        self.metadata = BioformatsMetadataRescale(self.path)
        self.metadata.set_active_plate_well(plate, well)
        self.do_rescale = do_rescale
        self.no_rescale_channel = no_rescale_channel
        self.rescale_range = rescale_range

    @staticmethod
    def rescale(img, rescale_range=(1, 99)):
        img = skimage.util.img_as_float32(img)
        cutoff1, cutoff2 = rescale_range

        if img.max() > (40000 / 65535):
            _img = img.copy()
            _img[_img > (10000 / 65535)] = 0
            p1 = np.percentile(_img, cutoff1)
            p99 = np.percentile(_img, cutoff2)
        else:
            p1 = np.percentile(img, cutoff1)
            p99 = np.percentile(img, cutoff2)

        img = skimage.exposure.rescale_intensity(img, in_range=(p1, p99), out_range=(0, 1))
        return (img * 65535).astype("uint16")

    def read(self, series, c):
        self.metadata._reader.setSeries(self.metadata.active_series[series])
        index = self.metadata._reader.getIndex(0, c, 0)
        byte_array = self.metadata._reader.openBytes(index)
        dtype = self.metadata.pixel_dtype
        shape = self.metadata.tile_size(series)
        img = np.frombuffer(byte_array.tostring(), dtype=dtype).reshape(shape)

        if not self.do_rescale:
            return img
        else:
            if c not in self.no_rescale_channel:
                # get rescale_range for channel c
                if isinstance(self.rescale_range, dict):
                    rescale_range = self.rescale_range[c]
                else:
                    rescale_range = self.rescale_range

                # actually read the image with rescaling applied
                return self.rescale(img, rescale_range=rescale_range)
            else:
                return img

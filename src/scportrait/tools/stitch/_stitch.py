"""
stitch
=======

Contains functions to assemble tiled images into fullscale mosaics. Uses out-of-memory computation for the assembly of larger than memory image mosaics.
"""

import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from alphabase.io.tempmmap import (
    create_empty_mmap,
    mmap_array_from_path,
    redefine_temp_location,
)
from tqdm import tqdm

from scportrait.io.daskmmap import dask_array_from_path
from scportrait.processing.images._image_processing import rescale_image
from scportrait.tools.stitch._utils.ashlar_plotting import plot_edge_quality, plot_edge_scatter
from scportrait.tools.stitch._utils.filereaders import (
    BioformatsReaderRescale,
    FilePatternReaderRescale,
)
from scportrait.tools.stitch._utils.filewriters import write_ome_zarr, write_spatialdata, write_tif, write_xml
from scportrait.tools.stitch._utils.parallelized_ashlar import ParallelEdgeAligner, ParallelMosaic


class Stitcher:
    """
    Class for stitching of image tiles to assemble a mosaic.

    Args:
        input_dir (str): Directory containing the input image tiles.
        slidename (str): Name of the slide.
        outdir (str): Output directory to save the stitched mosaic.
        stitching_channel (str): Name of the channel to be used for stitching.
        pattern (str): File pattern to match the image tiles.
        overlap (float, optional): Overlap between adjacent image tiles (default is 0.1).
        max_shift (float, optional): Maximum allowed shift during alignment (default is 30).
        filter_sigma (int, optional): Sigma value for Gaussian filter applied during alignment (default is 0).
        do_intensity_rescale (bool or "full_image", optional): Flag to indicate whether to rescale image intensities (default is True). Alternatively, set to "full_image" to rescale the entire image.
        rescale_range (tuple or dict, optional): If all channels should be rescaled to the same range pass a tuple with the percentiles for rescaling (default is (1, 99)). Alternatively, a dictionary can be passed with the channel names as keys and the percentiles as values if each channel should be rescaled to a different range.
        channel_order (list, optional): Order of channels in the generated output mosaic. If none (default value) the order of the channels is left unchanged.
        reader_type (class, optional): Type of reader to use for reading image tiles (default is FilePatternReaderRescale).
        orientation (dict, optional): Dictionary specifying which dimensions of the slide to flip (default is {'flip_x': False, 'flip_y': True}).
        plot_QC (bool, optional): Flag to indicate whether to plot quality control (QC) figures (default is True).
        overwrite (bool, optional): Flag to indicate whether to overwrite the output directory if it already exists (default is False).
        cache (str, optional): Directory to store temporary files during stitching (default is None). If set to none this directory will be created in the outdir.
    """

    def __init__(
        self,
        input_dir: str,
        slidename: str,
        outdir: str,
        stitching_channel: str,
        pattern: str,
        overlap: float = 0.1,
        max_shift: float = 30,
        filter_sigma: int = 0,
        do_intensity_rescale: bool | str = True,
        rescale_range: tuple = (1, 99),
        channel_order: list[str] = None,
        reader_type=FilePatternReaderRescale,
        orientation: dict = None,
        plot_QC: bool = True,
        overwrite: bool = False,
        cache: str = None,
    ) -> None:
        """
        Initialize the Stitcher object.

        Parameters:
        -----------
        input_dir : str
            Directory containing the input image tiles.
        slidename : str
            Name of the slide.
        outdir : str
            Output directory to save the stitched mosaic.
        stitching_channel : str
            Name of the channel to be used for stitching.
        pattern : str
            File pattern to match the image tiles.
        overlap : float, optional
            Overlap between adjacent image tiles (default is 0.1).
        max_shift : float, optional
            Maximum allowed shift during alignment (default is 30).
        filter_sigma : int, optional
            Sigma value for Gaussian filter applied during alignment (default is 0).
        do_intensity_rescale : bool or "full_image", optional
            Flag to indicate whether to rescale image intensities (default is True). Alternatively, set to "full_image" to rescale the entire image.
        rescale_range : tuple or dictionary, optional
            If all channels should be rescaled to the same range pass a tuple with the percentiles for rescaleing (default is (1, 99)). Alternatively
            a dictionary can be passed with the channel names as keys and the percentiles as values if each channel should be rescaled to a different range.
        channel_order : list, optional
            Order of channels in the generated output mosaic. If none (default value) the order of the channels is left unchanged.
        reader_type : class, optional
            Type of reader to use for reading image tiles (default is FilePatternReaderRescale).
        orientation : dict, optional
            Dictionary specifiying which dimensions of the slide to flip (default is {'flip_x': False, 'flip_y': True}).
        plot_QC : bool, optional
            Flag to indicate whether to plot quality control (QC) figures (default is True).
        overwrite : bool, optional
            Flag to indicate whether to overwrite the output directory if it already exists (default is False).
        cache : str, optional
            Directory to store temporary files during stitching (default is None). If set to none this directory will be created in the outdir.
        """
        self._lazy_imports()

        if orientation is None:
            orientation = {"flip_x": False, "flip_y": True}
        self.input_dir = input_dir
        self.slidename = slidename
        self.outdir = outdir
        self.stitching_channel = stitching_channel

        # stitching settings
        self.pattern = pattern
        self.overlap = overlap
        self.max_shift = max_shift
        self.filter_sigma = filter_sigma

        # image rescaling
        if do_intensity_rescale == "full_image":
            self.rescale_full_image = True
            self.do_intensity_rescale = True

        else:
            self.do_intensity_rescale = do_intensity_rescale  # type: ignore
            self.rescale_full_image = False
        self.rescale_range = rescale_range

        # setup reader for images
        self.orientation = orientation
        self.reader_type = reader_type

        # workflow setup
        self.plot_QC = plot_QC
        self.overwrite = overwrite
        self.channel_order = channel_order
        self.cache = cache

        self._initialize_outdir()

        # initialize variables to default values
        self.reader = None

    def _lazy_imports(self):
        """
        Import necessary packages for stitching.
        """
        from ashlar import thumbnail
        from ashlar.reg import EdgeAligner, Mosaic
        from ashlar.scripts.ashlar import process_axis_flip

        self.ashlar_thumbnail = thumbnail
        self.ashlar_EdgeAligner = EdgeAligner
        self.ashlar_Mosaic = Mosaic
        self.ashlar_process_axis_flip = process_axis_flip

    def __exit__(self):
        self._clear_cache()

    def __del__(self):
        self._clear_cache()

    def _create_cache(self):
        """
        Create a temporary cache directory for storing intermediate files during stitching.
        """
        if self.cache is None:
            TEMP_DIR_NAME = redefine_temp_location(self.outdir)
        else:
            TEMP_DIR_NAME = redefine_temp_location(self.cache)

        self.TEMP_DIR_NAME = TEMP_DIR_NAME

    def _clear_cache(self):
        """
        Clear the temporary cache directory.
        """
        if "TEMP_DIR_NAME" in self.__dict__:
            if os.path.exists(self.TEMP_DIR_NAME):
                shutil.rmtree(self.TEMP_DIR_NAME)

    def _initialize_outdir(self):
        """
        Initialize the output directory for saving the stitched mosaic.
        """
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)
            print("Output directory created at: ", self.outdir)
        else:
            if self.overwrite:
                print(f"Output directory at {self.outdir} already exists, overwriting.")
                shutil.rmtree(self.outdir)
                os.makedirs(self.outdir)
            else:
                raise FileExistsError(
                    f"Output directory at {self.outdir} already exists. Set overwrite to True to overwrite the directory."
                )

    def _get_channel_info(self):
        """
        Get information about the channels in the image tiles.
        """
        # get channel names
        self.channel_lookup = self.reader.metadata.channel_map
        self.channel_names = list(self.reader.metadata.channel_map.values())
        self.channels = list(self.reader.metadata.channel_map.keys())
        self.stitching_channel_id = list(self.channel_lookup.values()).index(self.stitching_channel)
        self.n_channels = len(self.channels)

    def get_stitching_information(self):
        """Print information about the configuration of the stitching process."""
        if self.reader is None:
            self._initialize_reader()
        self._get_channel_info()

        print("Tile positions will be calculated based on channel:", self.stitching_channel)
        print("Channel Names:", self.channel_names)
        print("Overlap of image tiles:", self.overlap)
        print("Max Shift value:", self.max_shift)
        print("Filter Sigma value:", self.filter_sigma)
        print("Output will be written to:", self.outdir)

    def _setup_rescaling(self):
        """
        Setup image rescaling based on the specified rescale_range.
        """
        # set up rescaling
        if self.do_intensity_rescale:
            self.reader.no_rescale_channel = []

            # if all channels should be rescaled to the same range, initialize dictionary with all channels
            if type(self.rescale_range) is tuple:
                self.rescale_range = {k: self.rescale_range for k in self.channel_names}

            # check if all channels are in dictionary for rescaling
            rescale_channels = list(self.rescale_range.keys())

            # make sure all channels provided in lookup dictionary are in the experiment
            if not set(rescale_channels).issubset(set(self.channel_names)):
                raise ValueError("The rescale_range dictionary contains a channel not found in the experiment.")

            # check if we have any channels missing in the rescale_range dictionary
            missing_channels = set.difference(set(self.channel_names), set(rescale_channels))

            if len(missing_channels) > 0:
                Warning(
                    "The rescale_range dictionary does not contain all channels in the experiment. This may lead to unexpected results. For the missing channels rescaling will be turned off."
                )

                missing_channels = set.difference(self.channel_names, rescale_channels)
                for missing_channel in missing_channels:
                    self.rescale_range[missing_channel] = (0, 1)

                self.reader.no_rescale_channel = [
                    list(self.channel_names).index(missing_channel) for missing_channel in missing_channels
                ]

            # lookup channel names and match them with channel ids to return a new dict whose keys are the channel ids
            rescale_range_ids = {list(self.channel_names).index(k): v for k, v in self.rescale_range.items()}
            self.reader.do_rescale = True
            self.reader.rescale_range = rescale_range_ids  # update so that the lookup can occur correctly

        else:
            self.reader.no_rescale_channel = []
            self.reader.do_rescale = False
            self.reader.rescale_range = None

    def _reorder_channels(self):
        """
        Reorder the channels in the mosaic based on the specified channel_order.
        """
        if self.channel_order is None:
            self.channels = self.channels
        else:
            print("current channel order: ", self.channels)

            channels = []
            for channel in self.channel_order:
                channels.append(self.channel_names.index(channel))
            print("new channel order", channels)

            self.channels = channels

    def _initialize_reader(self):
        """
        Initialize the reader for reading image tiles.
        """
        if self.reader_type == FilePatternReaderRescale:
            self.reader = self.reader_type(
                self.input_dir,
                self.pattern,
                self.overlap,
                rescale_range=self.rescale_range,
            )
        elif self.reader_type == BioformatsReaderRescale:
            self.reader = self.reader_type(self.input_dir, rescale_range=self.rescale_range)

        # setup correct orientation of slide (this depends on microscope used to generate the data)
        self.ashlar_process_axis_flip(
            self.reader,
            flip_x=self.orientation["flip_x"],
            flip_y=self.orientation["flip_y"],
        )

        # setup rescaling
        self._get_channel_info()
        self._setup_rescaling()

    def save_positions(self):
        """
        Save the positions of the aligned image tiles.
        """
        positions = self.aligner.positions
        np.savetxt(
            os.path.join(self.outdir, self.slidename + "_tile_positions.tsv"),
            positions,
            delimiter="\t",
        )

    def generate_thumbnail(self, scale=0.05):
        """
        Generate a thumbnail of the stitched mosaic.

        Args:
            scale (float, optional): Scale factor for the thumbnail. Defaults to 0.05.

        """
        self._initialize_reader()
        self.thumbnail = self.ashlar_thumbnail.make_thumbnail(
            self.reader, channel=self.stitching_channel_id, scale=scale
        )

        # rescale thumbnail to 0-1 range
        # if all channels should be rescaled to the same range, initialize dictionary with all channels
        if type(self.rescale_range) is tuple:
            rescale_range = {k: self.rescale_range for k in self.channel_names}
            rescale = True
        elif type(self.rescale_range) is dict:
            rescale_range = self.rescale_range[self.stitching_channel]
            rescale = True
        else:
            if not self.do_intensity_rescale:
                rescale = False  # turn off rescaling

        # rescale generated thumbnail
        if rescale:
            self.thumbnail = rescale_image(self.thumbnail, rescale_range)

    def _initialize_aligner(self):
        """
        Initialize the aligner for aligning the image tiles.

        Returns:
            aligner (EdgeAligner): Initialized EdgeAligner object.
        """
        aligner = self.ashlar_EdgeAligner(
            self.reader,
            channel=self.stitching_channel_id,
            filter_sigma=self.filter_sigma,
            verbose=True,
            do_make_thumbnail=False,
            max_shift=self.max_shift,
        )
        return aligner

    def plot_qc(self):
        """
        Plot quality control (QC) figures for the alignment.
        """
        plot_edge_scatter(self.aligner, self.outdir)
        plot_edge_quality(self.aligner, self.outdir)

    def _perform_alignment(self):
        """
        Perform alignment of the image tiles.
        """
        # intitialize reader for getting individual image tiles
        self._initialize_reader()

        print(f"performing stitching on channel {self.stitching_channel} with id number {self.stitching_channel_id}")
        self.aligner = self._initialize_aligner()
        self.aligner.run()

        if self.plot_QC:
            self.plot_qc()

        if self.save_positions:
            self.save_positions()

        self.aligner.reader._cache = {}  # need to empty cache for some reason

        print("Alignment complete.")

    def _initialize_mosaic(self):
        """
        Initialize the mosaic object for assembling the image tiles.

        Returns:
            mosaic (Mosaic): Initialized Mosaic object.
        """
        mosaic = self.ashlar_Mosaic(
            self.aligner,
            self.aligner.mosaic_shape,
            verbose=True,
            channels=self.channels,
        )
        return mosaic

    def _assemble_mosaic(self):
        """
        Assemble the image tiles into a mosaic.
        """
        # get dimensions of assembled final mosaic
        x, y = self.mosaic.shape
        shape = (self.n_channels, x, y)
        print(f"assembling mosaic with shape {shape}")

        # initialize tempmmap array to save assemled mosaic to
        # if no cache is specified the tempmmap will be created in the outdir

        self._create_cache()

        # create empty mmap array to store assembled mosaic
        hdf5_path = create_empty_mmap(shape, dtype=np.uint16, tmp_dir_abs_path=self.TEMP_DIR_NAME)
        print(f"created tempmmap array for assembled mosaic at {hdf5_path}")
        self.assembled_mosaic = mmap_array_from_path(hdf5_path)
        self.hdf5_path = hdf5_path  # save variable into self for easier access

        # assemble each of the channels
        for i, channel in tqdm(enumerate(self.channels), total=self.n_channels):
            self.assembled_mosaic[i, :, :] = self.mosaic.assemble_channel(
                channel=channel, out=self.assembled_mosaic[i, :, :]
            )

            if self.rescale_full_image:
                # warning this has not been tested for memory efficiency
                print("Rescaling entire input image to 0-1 range using percentiles specified in rescale_range.")
                self.assembled_mosaic[i, :, :] = rescale_image(
                    self.assembled_mosaic[i, :, :], self.rescale_range[channel]
                )

        # convery to dask array
        self.assembled_mosaic = dask_array_from_path(hdf5_path)

    def _generate_mosaic(self):
        # reorder channels
        self._reorder_channels()

        self.mosaic = self._initialize_mosaic()

        # ensure dtype is set correctly
        self.mosaic.dtype = np.uint16
        self._assemble_mosaic()

    def stitch(self):
        """
        Generate the stitched mosaic.
        """
        self._perform_alignment()
        self._generate_mosaic()

    def write_tif(self, export_xml: bool = True) -> None:
        """
        Write the assembled mosaic as TIFF files.

        Args:
            export_xml (bool, optional): Flag to indicate whether to export an XML file for the TIFF files (default is True). This XML file is compatible with loading the generated TIFF files into BIAS.

        Returns:
            The assembled mosaic are written to file as TIFF files in the specified output directory.
        """
        filenames = []
        for i, channel in enumerate(self.channel_names):
            filename = os.path.join(self.outdir, f"{self.slidename}_{channel}.tif")
            filenames.append(filename)
            write_tif(filename, self.assembled_mosaic[i, :, :])

        if export_xml:
            write_xml(filenames, self.channel_names, self.slidename)

    def write_ome_zarr(self, downscaling_size=4, n_downscaling_layers=4, chunk_size=(1, 1024, 1024)):
        """Write the assembled mosaic as an OME-Zarr file.

        Args:
            downscaling_size (int, optional): Downscaling factor for generating lower resolution layers (default is 4).
            n_downscaling_layers (int, optional): Number of downscaling layers to generate (default is 4).
            chunk_size (tuple, optional): Chunk size for the generated OME-Zarr file (default is (1, 1024, 1024)).

        Returns:
            None
        """
        filepath = os.path.join(self.outdir, f"{self.slidename}.ome.zarr")

        write_ome_zarr(
            filepath,
            self.assembled_mosaic,
            self.channel_names,
            self.slidename,
            overwrite=self.overwrite,
            downscaling_size=downscaling_size,
            n_downscaling_layers=n_downscaling_layers,
            chunk_size=chunk_size,
        )

    def write_thumbnail(self):
        """
        Write the generated thumbnail as a TIFF file.
        """
        # calculate thumbnail if this has not already been done
        if "thumbnail" not in self.__dict__:
            self.generate_thumbnail()

        filename = os.path.join(
            self.outdir,
            self.slidename + "_thumbnail_" + self.stitching_channel + ".tif",
        )
        write_tif(filename, self.thumbnail)

    def write_spatialdata(self, scale_factors=None):
        """
        Write the assembled mosaic as a SpatialData object.

        Args:
            scale_factors (list, optional): List of scale factors for the generated SpatialData object.
                Default is [2, 4, 8]. The scale factors are used to generate downsampled versions of the
                image for faster visualization at lower resolutions.
        """
        if scale_factors is None:
            scale_factors = [2, 4, 8]
        filepath = os.path.join(self.outdir, f"{self.slidename}.spatialdata")

        # create spatialdata object
        write_spatialdata(
            filepath,
            image=self.assembled_mosaic,
            channel_names=self.channels,
            scale_factors=scale_factors,
            overwrite=self.overwrite,
        )


class ParallelStitcher(Stitcher):
    """
    Class for parallel stitching of image tiles and generating a mosaic. For applicable steps multi-threading is used for faster processing.

    Args:
        input_dir (str): Directory containing the input image tiles.
        slidename (str): Name of the slide.
        outdir (str): Output directory to save the stitched mosaic.
        stitching_channel (str): Name of the channel to be used for stitching.
        pattern (str): File pattern to match the image tiles.
        overlap (float, optional): Overlap between adjacent image tiles (default is 0.1).
        max_shift (float, optional): Maximum allowed shift during alignment (default is 30).
        filter_sigma (int, optional): Sigma value for Gaussian filter applied during alignment (default is 0).
        do_intensity_rescale (bool or "full_image", optional): Flag to indicate whether to rescale image intensities (default is True). Alternatively, set to "full_image" to rescale the entire image.
        rescale_range (tuple or dict, optional): If all channels should be rescaled to the same range pass a tuple with the percentiles for rescaling (default is (1, 99)). Alternatively, a dictionary can be passed with the channel names as keys and the percentiles as values if each channel should be rescaled to a different range.
        channel_order (list, optional): Order of channels in the generated output mosaic. If none (default value) the order of the channels is left unchanged.
        reader_type (class, optional): Type of reader to use for reading image tiles (default is FilePatternReaderRescale).
        orientation (dict, optional): Dictionary specifying which dimensions of the slide to flip (default is {'flip_x': False, 'flip_y': True}).
        plot_QC (bool, optional): Flag to indicate whether to plot quality control (QC) figures (default is True).
        overwrite (bool, optional): Flag to indicate whether to overwrite the output directory if it already exists (default is False).
        cache (str, optional): Directory to store temporary files during stitching (default is None). If set to none this directory will be created in the outdir.
        threads (int, optional): Number of threads to use for parallel processing (default is 20).
    """

    def __init__(
        self,
        input_dir: str,
        slidename: str,
        outdir: str,
        stitching_channel: str,
        pattern: str,
        overlap: float = 0.1,
        max_shift: float = 30,
        filter_sigma: int = 0,
        do_intensity_rescale: bool = True,
        rescale_range: tuple = (1, 99),
        plot_QC: bool = True,
        WGAchannel: str = None,
        channel_order: list[str] = None,
        overwrite: bool = False,
        reader_type=FilePatternReaderRescale,
        orientation=None,
        cache: str = None,
        threads: int = 20,
    ) -> None:
        if orientation is None:
            orientation = {"flip_x": False, "flip_y": True}
        super().__init__(
            input_dir,
            slidename,
            outdir,
            stitching_channel,
            pattern,
            overlap,
            max_shift,
            filter_sigma,
            do_intensity_rescale,
            rescale_range,
            channel_order,
            reader_type,
            orientation,
            plot_QC,
            overwrite,
            cache,
        )
        # dirty fix to avoide multithreading error with BioformatsReader until this can be fixed
        if self.reader_type == BioformatsReaderRescale:
            threads = 1
            print(
                "BioformatsReaderRescale does not support multithreading for calculating the error threshold currently. Proceeding with 1 thread."
            )
            Warning(
                "BioformatsReaderRescale does not support multithreading for calculating the error threshold currently. Proceeding with 1 thread."
            )

        self.threads = threads

    def _initialize_aligner(self):
        """
        Initialize the aligner for aligning the image tiles.

        Returns:
            aligner (ParallelEdgeAligner): Initialized ParallelEdgeAligner object.
        """
        aligner = ParallelEdgeAligner(
            self.reader,
            channel=self.stitching_channel_id,
            filter_sigma=self.filter_sigma,
            verbose=True,
            do_make_thumbnail=False,
            max_shift=self.max_shift,
            n_threads=self.threads,
        )
        return aligner

    def _initialize_mosaic(self):
        mosaic = ParallelMosaic(
            self.aligner, self.aligner.mosaic_shape, verbose=True, channels=self.channels, n_threads=self.threads
        )
        return mosaic

    def _assemble_channel(self, args):
        hdf5_path = self.hdf5_path
        channel, i, hdf5_path = args
        out = mmap_array_from_path(hdf5_path)
        self.mosaic.assemble_channel_parallel(channel=channel, ch_index=i, hdf5_path=hdf5_path)

        if self.rescale_full_image:
            # warning this has not been tested for memory efficiency
            print("Rescaling entire input image to 0-1 range using percentiles specified in rescale_range.")
            out[i, :, :] = rescale_image(out[i, :, :], self.rescale_range[channel])

    def _assemble_mosaic(self):
        # get dimensions of assembled final mosaic
        x, y = self.mosaic.shape
        shape = (self.n_channels, x, y)

        print(f"assembling mosaic with shape {shape}")

        self._create_cache()

        hdf5_path = create_empty_mmap(shape, dtype=np.uint16, tmp_dir_abs_path=self.TEMP_DIR_NAME)
        print(f"created tempmmap array for assembled mosaic at {hdf5_path}")

        self.assembled_mosaic = mmap_array_from_path(hdf5_path)
        self.hdf5_path = hdf5_path  # save variable to self for easier access

        # assemble each of the channels
        args = []
        for i, channel in enumerate(self.channels):
            args.append((channel, i, hdf5_path))

        tqdm_args = {
            "file": sys.stdout,
            "desc": "assembling mosaic",
            "total": len(self.channels),
        }

        # threading over channels is safe as the channels are written to different postions in the hdf5 file and do not interact with one another
        # threading over the writing of a single channel is not safe and leads to inconsistent results
        workers = np.min([self.threads, self.n_channels])
        print(f"assembling channels with {workers} workers")
        with ThreadPoolExecutor(max_workers=workers) as executor:
            list(tqdm(executor.map(self._assemble_channel, args), **tqdm_args))

        # conver to dask array
        self.assembled_mosaic = dask_array_from_path(hdf5_path)

    def write_tif_parallel(self, export_xml=True):
        """Parallelized version of the write_tif method to write the assembled mosaic as TIFF files.

        Args:
            export_xml (bool, optional): Flag to indicate whether to export an XML file for the TIFF files (default is True). This XML file is compatible with loading the generarted TIFF files into BIAS.

        """

        filenames = []
        args = []
        for i, channel in enumerate(self.channel_names):
            filename = os.path.join(self.outdir, f"{self.slidename}_{channel}.tif")
            filenames.append(filename)
            args.append((filename, i))

        tqdm_args = {
            "file": sys.stdout,
            "desc": "writing tif files",
            "total": len(self.channels),
        }

        # define helper function to execute in threadpooler
        def _write_tif(args):
            filename, ix = args
            write_tif(filename, self.assembled_mosaic[ix, :, :])

        # threading over channels is safe as the channels are written to different files
        workers = np.min([self.threads, self.n_channels])
        with ThreadPoolExecutor(max_workers=workers) as executor:
            list(tqdm(executor.map(_write_tif, args), **tqdm_args))

        # write_tif(filename, self.assembled_mosaic[i, :, :])

        if export_xml:
            write_xml(filenames, self.channel_names, self.slidename)
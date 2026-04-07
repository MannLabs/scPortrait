"""
parse
====================================

Contains functions to parse imaging data acquired on an Opera Phenix or Operetta into usable formats for downstream pipelines.
"""

import os
import platform
import re
import shutil
import sys
import warnings
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import PosixPath

import numpy as np
import pandas as pd
from tifffile import imread, imwrite
from tqdm.auto import tqdm


def _get_child_name(elem):
    return elem.split("}")[1]


class PhenixParser:
    """
    Parse and manage image data exported from Phenix experiments.
    """

    def __init__(
        self,
        experiment_dir: str | PosixPath,
        flatfield_exported: bool = True,
        use_symlinks: bool = True,
        compress_rows: bool = False,
        compress_cols: bool = False,
        overwrite: bool = False,
    ) -> None:
        """
        Args:
            experiment_dir: Directory containing the exported Phenix experiment.
            flatfield_exported: Whether flatfield corrected images were exported.
            use_symlinks: Whether to use symbolic links for parsed images.
            compress_rows: Whether to merge all plate rows into a single parsed row.
            compress_cols: Whether to merge all wells into a single parsed column.
            overwrite: Whether to overwrite existing files during the parsing process.
        """
        self.experiment_dir = experiment_dir
        self.export_symlinks = use_symlinks
        self.flatfield_status = flatfield_exported
        self.compress_rows = compress_rows
        self.compress_cols = compress_cols
        self.overwrite = overwrite

        if self.compress_rows:
            print(
                "The rows found in the phenix layout will be compressed into one row after parsing the images, r and c indicators will be adjusted accordingly."
            )
        if self.compress_cols:
            print(
                "The wells found in the phenix layout will be compressed into one column after parsing the images, r and c indicators will be adjusted accordingly."
            )

        self.xml_path = self._get_xml_path()
        self.image_dir = self._get_input_dir()
        self.channel_lookup = self._get_channel_metadata(self.xml_path)
        self.metadata: None | pd.DataFrame = None
        self.missing_images_copy: list[str] = []
        self.outdirs: dict[str, str] = {}

    def _get_xml_path(self) -> str | PosixPath:
        """Automatically gets the path to the XML file containing metadata."""

        # directory depends on if flatfield images were exported or not
        # these generated folder structures are hard coded during phenix export, do not change

        if self.flatfield_status:
            index_file_names = ["Index.xml", "Index.ref.xml"]
            for index_file_name in index_file_names:
                index_file = os.path.join(self.experiment_dir, "Images", index_file_name)
                if os.path.isfile(index_file):
                    break
        else:
            index_file_names = ["Index.xml", "Index.idx.xml"]
            for index_file_name in index_file_names:
                index_file = os.path.join(self.experiment_dir, "Images", index_file_name)
                if os.path.isfile(index_file):
                    break

        # perform sanity check if file exists else exit
        if not os.path.isfile(index_file):
            sys.exit(f"Can not find index file at path: {index_file}")

        return index_file

    def _get_input_dir(self) -> str | PosixPath:
        """Automatically get the subfolder where the exported image files are located."""
        # directory depends on if flatfield images were exported or not
        # these generated folder structures are hard coded during phenix export, do not change
        if self.flatfield_status:
            input_dir = os.path.join(self.experiment_dir, "Images", "flex")
        else:
            input_dir = os.path.join(self.experiment_dir, "Images")

        # perform sanity check if file exists else exit
        if not os.path.isdir(input_dir):
            sys.exit(f"Can not find directory containing images to parse: {input_dir}")

        return input_dir

    def _define_outdir(self, name: str = "parsed_images") -> None:
        """Create output directory for parsed images.

        Args:
            name: Name of the output directory.
        """

        self.outdirs[name] = f"{self.experiment_dir}/{name}"

        # if output directory did not exist create it
        if not os.path.isdir(self.outdirs[name]):
            os.makedirs(self.outdirs[name])

    def _get_channel_metadata(self, xml_path: str | PosixPath) -> pd.DataFrame:
        """Parse channel metadata from the exported Phenix index XML.

        Args:
            xml_path: Path to the XML file containing metadata.

        Returns:
            A lookup table for channel names and IDs.
        """
        index_file = xml_path

        # Read and parse the XML file
        with open(index_file) as f:
            content = f.read()

        # Use regex to detect pattern
        pattern = r"<ChannelName>.*?</ChannelName>|<ChannelID>.*?</ChannelID>"
        results = re.findall(pattern, content, re.MULTILINE)
        results = [x.strip() for x in results]

        # Extract channel IDs and names
        channel_ids = [x.split(">")[1].split("<")[0] for x in results if x.startswith("<ChannelID")]
        channel_names = [
            x.split(">")[1].split("<")[0].replace(" ", "") for x in results if x.startswith("<ChannelName")
        ]

        # Process
        channel_ids = list(set(channel_ids))
        channel_ids.sort()
        channel_names = channel_names[0 : len(channel_ids)]

        # Create DataFrame
        lookup = pd.DataFrame({"id": list(channel_ids), "label": list(channel_names)})

        print("Experiment contains the following image channels: ")
        print(lookup, "\n")

        # Save lookup file to csv
        lookup.to_csv(f"{self.experiment_dir}/channel_lookuptable.csv")
        print(f"Channel Lookup table saved to file at {self.experiment_dir}/channel_lookuptable.csv\n")

        # Set channel names
        self.channel_names = channel_names

        return lookup

    def _read_phenix_xml(self, xml_path: str | PosixPath) -> pd.DataFrame:
        """Read and parse the XML file containing metadata from a Phenix experiment.

        Args:
            xml_path: Path to the XML file containing metadata.

        Returns:
            A DataFrame containing the metadata of the experiment, including
            floating-point stage positions in ``X`` and ``Y``.
        """
        # initialize lists to save results into
        rows = []
        cols = []
        fields = []
        planes = []
        channel_ids = []
        channel_names = []
        flim_ids = []
        timepoints: list[int] = []
        x_positions: list[float] = []
        y_positions: list[float] = []
        times = []
        url = []

        # extract information from XML file
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # get Harmony Version number
        namespace = root.tag.split("}")[0].strip("{")
        if "HarmonyV5" in namespace:
            version = "HarmonyV5"
            self.harmony_version = version
        elif "HarmonyV7" in namespace:
            version = "HarmonyV7"
            self.harmony_version = version
        else:
            raise ValueError(
                f"Found a currently unsupported version number {namespace}. Please contact the developers with this example."
            )

        print(f"Parsing XML file from {version} version.")

        for i, child in enumerate(root):
            if _get_child_name(child.tag) == "Images":
                images = root[i]

        for image in images:
            for _ix, child in enumerate(image):
                tag = _get_child_name(child.tag)
                if tag == "Row":
                    rows.append(child.text)
                    continue
                elif tag == "Col":
                    cols.append(child.text)
                    continue
                elif tag == "FieldID":
                    fields.append(child.text)
                    continue
                elif tag == "PlaneID":
                    planes.append(child.text)
                    continue
                elif tag == "ChannelID":
                    channel_ids.append(child.text)
                    continue
                elif tag == "FlimID":
                    flim_ids.append(child.text)
                    continue
                elif tag == "TimepointID":
                    timepoints.append(int(child.text))
                    continue
                elif tag == "PositionX":
                    x_positions.append(float(child.text))
                    continue
                elif tag == "PositionY":
                    y_positions.append(float(child.text))
                    continue
                elif tag == "AbsTime":
                    times.append(child.text)
                    continue
                elif tag == "URL":
                    url.append(child.text)
                    continue
                else:
                    continue

        rows = [str(x).zfill(2) for x in rows]
        cols = [str(x).zfill(2) for x in cols]
        fields = [str(x).zfill(2) for x in fields]
        planes = [str(x).zfill(2) for x in planes]
        timepoints = [int(x) + 1 for x in timepoints]

        # get channelnames
        lookup_dict = self.channel_lookup.set_index("id").to_dict()["label"]
        channel_names = [lookup_dict[channel_id] for channel_id in channel_ids]
        channel_names = [
            x.replace(" ", "") for x in channel_names
        ]  # ensure no extra spaces are contained in the generated files

        # get file names of single-tif files on disk
        # the structure of the file names is dependent on the Harmony Version used to export them
        image_names = []
        if version == "HarmonyV5":
            for row, col, field, plane, channel_id, timepoint, flim_id in zip(
                rows, cols, fields, planes, channel_ids, timepoints, flim_ids, strict=False
            ):
                image_names.append(f"r{row}c{col}f{field}p{plane}-ch{channel_id}sk{timepoint}fk1fl{flim_id}.tiff")
        elif version == "HarmonyV7":
            timepoints = [(x - 1) for x in timepoints]
            if self.flatfield_status:
                for (
                    row,
                    col,
                    field,
                    plane,
                    channel_id,
                    timepoint,
                ) in zip(rows, cols, fields, planes, channel_ids, timepoints, strict=False):
                    image_names.append(f"r{row}c{col}f{field}p{plane}-ch{channel_id}t{str(timepoint).zfill(2)}.tiff")
            else:
                for row, col, field, plane, channel_id, timepoint, flim_id in zip(
                    rows, cols, fields, planes, channel_ids, timepoints, flim_ids, strict=False
                ):
                    image_names.append(f"r{row}c{col}f{field}p{plane}-ch{channel_id}sk{timepoint}fk1fl{flim_id}.tiff")

        # convert date/time into useful format
        dates = [x.split("T")[0] for x in times]
        _times = [x.split("T")[1] for x in times]
        _times = [(x.split("+")[0].split(".")[0] + "+" + x.split("+")[1].replace(":", "")) for x in _times]
        time_final: list[str] = [x + " " + y for x, y in zip(dates, _times, strict=False)]

        datetime_format = "%Y-%m-%d %H:%M:%S%z"
        time_unix: list[float] = [datetime.timestamp(datetime.strptime(x, datetime_format)) for x in time_final]

        # update file name if flatfield exported images are to be used
        # syntax here is somewhat complicated because export from harmony is not consistent even within the same version,
        # sometimes flex_ is prepended sometimes not (this has to do with which level of flatfield correction is available due to the number of tiles images)
        if self.flatfield_status:
            _image_names = [f"flex_{x}" for x in image_names]
            status = False
            for img in _image_names:
                path = os.path.join(self.image_dir, img)
                if os.path.exists(path):
                    status = True
                    break
            if status:
                image_names = _image_names

        df = pd.DataFrame(
            {
                "filename": image_names,
                "Row": rows,
                "Well": cols,
                "Zstack": planes,
                "Timepoint": timepoints,
                "X": x_positions,
                "Y": y_positions,
                "date": dates,
                "time": _times,
                "unix_time": time_unix,
                "Channel": channel_names,
            }
        )

        # define path where to find raw image files
        df["source"] = self.image_dir
        df["filename"] = df["filename"].astype(str)  # ensure this is a string otherwise it can cause issues later on
        df["date"] = df["date"].astype(str)
        df["time"] = df["time"].astype(str)
        df["Channel"] = df["Channel"].astype(str)

        return df

    def _get_phenix_metadata(self) -> pd.DataFrame:
        """Helper function to get metadata from Phenix XML file."""
        return self._read_phenix_xml(self.xml_path)

    @staticmethod
    def _cluster_axis_positions(values: np.ndarray) -> list[tuple[np.ndarray, int]]:
        """Group nearly-identical stage positions into one tile coordinate."""

        def _estimate_tolerance(sorted_values: np.ndarray) -> float:
            """Estimate a clustering tolerance that separates jitter from tile spacing."""
            diffs = np.diff(sorted_values)
            positive_diffs = diffs[diffs > 0]
            if len(positive_diffs) == 0:
                return 0.0

            step_size = float(np.quantile(positive_diffs, 0.9))
            if step_size <= 0:
                return 0.0

            jitter_limit = step_size * 0.2
            jitter_diffs = positive_diffs[positive_diffs <= jitter_limit]
            tolerance_floor = step_size * 0.01

            if len(jitter_diffs) == 0:
                return max(step_size * 0.05, tolerance_floor)

            jitter_scale = float(np.quantile(jitter_diffs, 0.95))
            return max(jitter_scale * 2, tolerance_floor)

        unique_values = np.sort(np.unique(values))
        if len(unique_values) == 0:
            return []
        if len(unique_values) == 1:
            return [(np.array([unique_values[0]]), 0)]

        tolerance = _estimate_tolerance(unique_values)

        clusters = [[unique_values[0]]]
        for value in unique_values[1:]:
            if value - clusters[-1][-1] <= tolerance:
                clusters[-1].append(value)
            else:
                clusters.append([value])

        return [(np.array(cluster), index) for index, cluster in enumerate(clusters)]

    def _assign_tile_positions(self, metadata: pd.DataFrame) -> pd.DataFrame:
        """Annotate metadata with per-well tile coordinates derived from stage positions."""
        metadata = metadata.copy()

        # convert position values to numeric to ensure proper sorting
        metadata["X"] = [float(x) for x in metadata.X]
        metadata["Y"] = [float(x) for x in metadata.Y]

        # convert stage positions into tile coordinates relative to each well
        metadata["X_pos"] = None
        metadata["Y_pos"] = None
        for (_, _), group in metadata.groupby(["Row", "Well"], sort=False):
            for x_cluster, i in self._cluster_axis_positions(group.X.to_numpy()):
                metadata.loc[group.index[group.X.isin(x_cluster)], "X_pos"] = i

            for y_cluster, i in self._cluster_axis_positions(group.Y.to_numpy()):
                metadata.loc[group.index[group.Y.isin(y_cluster)], "Y_pos"] = i

        metadata["X_pos"] = metadata["X_pos"].astype(int)
        metadata["Y_pos"] = metadata["Y_pos"].astype(int)

        return metadata

    def _generate_new_filenames(self, metadata: pd.DataFrame) -> pd.DataFrame:
        """Generate parsed output file names for each image.

        Args:
            metadata: Metadata generated by ``_get_phenix_metadata()``.

        Returns:
            The input metadata with relative tile coordinates and parsed file names added.
        """
        if "X_pos" not in metadata.columns or "Y_pos" not in metadata.columns:
            metadata = self._assign_tile_positions(metadata)
        else:
            metadata = metadata.copy()
            metadata["X_pos"] = metadata["X_pos"].astype(int)
            metadata["Y_pos"] = metadata["Y_pos"].astype(int)

        # get number of rows and wells and adjust labelling if specific entries need to be compressed
        wells = metadata.Well.value_counts().index.to_list()
        rows = metadata.Row.value_counts().index.to_list()

        wells.sort()
        rows.sort(reverse=True)  # invert because we need to start assembling from bottom left

        if self.compress_rows:
            metadata["orig_Row"] = metadata["Row"]
            select_row_name = rows[0]
            for i, row in enumerate(rows):
                for well in wells:
                    if i == 0:
                        continue
                    else:
                        # get current highest index
                        max_y = metadata.loc[((metadata.Well == well) & (metadata.Row == select_row_name))].Y_pos.max()

                        # add current highest index to existing index
                        metadata.loc[(metadata.Well == well) & (metadata.Row == row), "Y_pos"] = (
                            metadata.loc[(metadata.Well == well) & (metadata.Row == row), "Y_pos"] + int(max_y) + 1
                        )
                        # update row name
                        metadata.loc[(metadata.Well == well) & (metadata.Row == row), "Row"] = select_row_name

            metadata.loc[:, "Row"] = rows[-1]  # update nomenclature to start with the row 01

        if self.compress_cols:
            metadata["orig_Well"] = metadata["Well"]
            for i, well in enumerate(wells):
                if i == 0:
                    continue
                else:
                    max_x = metadata.loc[(metadata.Well == wells[0])].X_pos.max()
                    metadata.loc[(metadata.Well == well), "X_pos"] = (
                        metadata.loc[(metadata.Well == well), "X_pos"] + int(max_x) + 1
                    )
                    metadata.loc[(metadata.Well == well), "Well"] = wells[0]

        metadata["X_pos"] = metadata["X_pos"].astype(int)
        metadata["Y_pos"] = metadata["Y_pos"].astype(int)
        metadata["X_pos"] = metadata["X_pos"].map(lambda x: str(x).zfill(3))
        metadata["Y_pos"] = metadata["Y_pos"].map(lambda x: str(x).zfill(3))
        metadata.Timepoint = [str(x).zfill(3) for x in metadata.Timepoint]
        metadata.Zstack = [str(x).zfill(2) for x in metadata.Zstack]

        # generate new file names
        for i in range(metadata.shape[0]):
            _row = metadata.loc[i, :]
            name = (
                f"Timepoint{_row.Timepoint}_Row{_row.Row}_Well{_row.Well}_{_row.Channel}"
                f"_zstack{_row.Zstack}_r{_row.Y_pos}_c{_row.X_pos}.tif"
            )
            name = name
            metadata.loc[i, "new_file_name"] = name

        return metadata

    def _get_tile_id(self, string: str) -> str:
        """Helper function to extract tile id from filename.

        Args:
            string: Filename of the image.

        Returns:
            The tile id extracted from the filename in the format r<row>_c<col>.
        """
        pattern = r"_r(\d+)_c(\d+)\.tif"
        match = re.search(pattern, string)
        if match:
            row = match.group(1)
            col = match.group(2)
            return f"r{row}_c{col}"
        else:
            return None

    def generate_metadata(self) -> pd.DataFrame:
        """Generate metadata for the Phenix experiment, including parsed file names."""
        metadata = self._get_phenix_metadata()
        metadata_new = self._generate_new_filenames(metadata)

        # save results to self for easy access
        self.metadata = metadata_new

        return metadata_new

    def check_for_missing_files(self, metadata: pd.DataFrame = None, return_values: bool = False) -> list | None:
        """Check for missing images in the experiment.

        Stitching requires a full rectangular tile grid, so missing images
        are identified and can later be replaced with black images.

        Args:
            metadata: Metadata for the experiment. If omitted, cached or freshly
                generated metadata is used.
            return_values: Whether to return the list of missing images instead
                of only storing it on ``self.missing_images``.

        Returns:
            A list of missing images if ``return_values`` is ``True``, otherwise ``None``.
        """

        def _generate_missing_file_names(x_positions, y_positions, timepoint, row, well, channels, zstacks):
            """Helper function to generate missing file names given x_positions and y_positions."""

            _missing_tiles: list[str] = []

            for channel in channels:
                for zstack in zstacks:
                    for x_pos in x_positions:
                        for y_pos in y_positions:
                            _missing_tiles.append(
                                f"Timepoint{timepoint}_Row{row}_Well{well}_{channel}_zstack{zstack}_r{y_pos}_c{x_pos}.tif"
                            )
            return _missing_tiles

        # check if metadata has been passed or is already calculated, else repeat calculation
        if metadata is None:
            if "metadata" in self.__dict__:
                metadata = self.metadata
                if "new_file_name" not in metadata.columns:
                    metadata = self._generate_new_filenames(metadata)
            else:
                metadata = self.generate_metadata()
                metadata = self._generate_new_filenames(metadata)

        # get unique values for each category describing the imaging experiment
        channels = np.unique(metadata.Channel)
        zstacks = np.unique(metadata.Zstack)
        rows = np.unique(metadata.Row)
        wells = np.unique(metadata.Well)
        timepoints = np.unique(metadata.Timepoint)

        # all X and Y pos values need to be there
        y_range = np.unique(metadata.Y_pos)
        x_range = np.unique(metadata.X_pos)

        # check to ensure that no y_range or x_range value is missing
        _y_range = [int(x) for x in y_range]
        y_range = [str(x).zfill(3) for x in np.arange(min(_y_range), max(_y_range) + 1)]
        _x_range = [int(x) for x in x_range]
        x_range = [str(x).zfill(3) for x in np.arange(min(_x_range), max(_x_range) + 1)]

        # this will not catch missing tiles were an entire row or column is missing
        missing_tiles: list[str] = []
        print("Checking for missing images...")

        for timepoint in timepoints:
            _df = metadata[metadata.Timepoint == timepoint]
            for row in rows:
                __df = _df[_df.Row == row]
                if __df.empty:
                    Warning(f"Entire row {row} is missing for timepoint {timepoint}.")
                    for well in wells:
                        missing_tiles = missing_tiles + _generate_missing_file_names(
                            x_range, y_range, timepoint, row, well, channels, zstacks
                        )

                for well in wells:
                    ___df = __df[__df.Well == well]

                    if ___df.empty:
                        Warning(f"Entire well {well} is missing for timepoint {timepoint}.")
                        missing_tiles = missing_tiles + _generate_missing_file_names(
                            x_range, y_range, timepoint, row, well, channels, zstacks
                        )
                        continue

                    for x_pos in x_range:
                        _check = ___df[___df.X_pos == x_pos]
                        _y_pos = [y_pos for y_pos in y_range if y_pos not in set(_check.Y_pos)]

                        if len(_y_pos) > 0:
                            missing_tiles = missing_tiles + _generate_missing_file_names(
                                [x_pos], _y_pos, timepoint, row, well, channels, zstacks
                            )

                    for y_pos in y_range:
                        _check = ___df[___df.Y_pos == y_pos]
                        _x_pos = [x_pos for x_pos in x_range if x_pos not in set(_check.X_pos)]

                        if len(_y_pos) > 0:
                            missing_tiles = missing_tiles + _generate_missing_file_names(
                                _x_pos, [y_pos], timepoint, row, well, channels, zstacks
                            )

        if self.harmony_version == "HarmonyV7":
            missing_tiles.extend(self.missing_images_copy)  # add additional missing images from copy process
            self.missing_images = missing_tiles

        elif self.harmony_version == "HarmonyV5":
            self.missing_images = missing_tiles

        if len(self.missing_images) == 0:
            print("No missing tiles found.")
        else:
            # get size of missing images that need to be replaced
            image = None
            for source, file in zip(metadata["source"], metadata["filename"], strict=True):
                path = os.path.join(source, file)
                if os.path.exists(path):
                    image = imread(path)
                    break
            if image is None:
                raise ValueError(
                    f"Could not read any input images to determine size. Tried finding images at the following path: {path}"
                )
            image[:] = 0
            self.black_image = image

            print(f"The found missing tiles that need to be replaced with black images of the size {image.shape}.")

        if return_values:
            return missing_tiles
        else:
            return None

    def replace_missing_images(self) -> None:
        """Replace missing images with black images of the same size."""
        # calculate missing images if not already done
        if "missing_images" not in self.__dict__:
            self.check_for_missing_files()

        # initialize output directory if not already done
        if self.outdirs["parsed_images"] not in self.__dict__:
            self._define_outdir(name="parsed_images")

        # if there are missing images replace them with black images
        if len(self.missing_images) > 0:
            for missing_image in self.missing_images:
                print(f"Creating black image with name: {missing_image}")
                imwrite(os.path.join(self.outdirs["parsed_images"], missing_image), self.black_image)

            print(
                f"All missing images successfully replaced with black images of the dimension {self.black_image.shape}"
            )

    def _define_copy_functions(self) -> None:
        """Define the file copy/link function based on the configured export mode."""
        if self.export_symlinks:

            def copyfunction(input, output):
                try:
                    if platform.system() == "Windows":
                        warnings.warn(
                            "\n\nWindows detected as platform. Symlinks cannot be used on Windows. Using hard links instead.\n\n",
                            stacklevel=2,
                        )
                        # On Windows, use hard links when symlinks are requested
                        if not os.path.exists(output):
                            os.link(input, output)
                    else:
                        # Unix symlink behavior
                        os.symlink(input, output)
                except OSError as e:
                    print("Error: ", e)
                    return ()
        else:

            def copyfunction(input, output):
                try:
                    # Create destination directory if it doesn't exist
                    os.makedirs(os.path.dirname(output), exist_ok=True)
                    shutil.copyfile(input, output)
                except OSError as e:
                    print("Error: ", e)
                    return ()

        self.copyfunction = copyfunction

    def _copy_files(self, metadata: pd.DataFrame) -> None:
        """
        Copy files from the source directory to the output directory.

        Args:
            metadata: Parsed metadata with at least ``filename``, ``new_file_name``,
                ``source``, and ``dest`` columns.
        """
        print("Starting copy process...")
        self._define_copy_functions()
        # actually perform the copy process
        for old, new, source, dest in tqdm(
            zip(
                metadata.filename.tolist(),
                metadata.new_file_name.tolist(),
                metadata.source.tolist(),
                metadata.dest.tolist(),
                strict=False,
            ),
            total=len(metadata.new_file_name.tolist()),
            desc="Copying files",
        ):
            # define old and new paths for copy process
            old_path = os.path.join(source, old)
            new_path = os.path.join(dest, new)
            # check if old path exists
            if os.path.exists(old_path):
                if os.path.exists(new_path):
                    if self.overwrite:
                        os.remove(new_path)
                        self.copyfunction(old_path, new_path)
                    else:
                        self.copyfunction(old_path, new_path)
                else:
                    self.copyfunction(old_path, new_path)

            else:
                if self.harmony_version == "HarmonyV5":
                    print("Error: ", old_path, "not found.")
                elif self.harmony_version == "HarmonyV7":
                    self.missing_images_copy.append(new)
        print("Copy process completed.")

    def _save_metadata(self, metadata: pd.DataFrame) -> None:
        """Save metadata used to parse images to a csv file."""
        metadata.to_csv(f"{self.experiment_dir}/metadata_image_parsing.csv")
        print(f"Metadata used to parse images saved to file {self.experiment_dir}/metadata_image_parsing.csv")

    def parse(self, check_missing_tiles: bool = True) -> None:
        """Complete parsing of the Phenix experiment.

        Args:
            check_missing_tiles: Whether to check for and generate missing tiles.
        """
        # create output directory
        self._define_outdir(name="parsed_images")

        # get metadata for the images we want to parse
        metadata = self.generate_metadata()

        # set destination for copying
        metadata["dest"] = self.outdirs["parsed_images"]

        # copy/link the images to their new names
        self._copy_files(metadata=metadata)

        # check for missing images and replace them
        if check_missing_tiles:
            self.check_for_missing_files(metadata=metadata)
            self.replace_missing_images()
        self._save_metadata(metadata)

    def sort_wells(self, sort_tiles: bool = False) -> None:
        """Sort parsed images by well.

        Generates a folder tree where each well has its own folder containing all images from that well.
        If ``sort_tiles`` is ``True``, an additional layer is added where all
        images from the same FOV are sorted into a dedicated subfolder.

        Args:
            sort_tiles: Whether to sort images into per-tile directories within each well.
        """

        # create output directory
        self._define_outdir(name="sorted_wells")

        # get all new file names
        if "metadata" in self.__dict__:
            metadata = self.metadata
        else:
            metadata = self.generate_metadata()

        metadata["tiles"] = [self._get_tile_id(x) for x in metadata.new_file_name.to_list()]

        # get unique rows, wells and tiles
        timepoints = list(set(metadata.Timepoint.to_list()))
        rows = list(set(metadata.Row.to_list()))
        wells = list(set(metadata.Well.to_list()))
        tiles = list(set(metadata.tiles.to_list()))

        print("Found the following image specs: ")
        print("\t Timepoints: ", timepoints)
        print("\t Rows: ", rows)
        print("\t Wells: ", wells)
        if sort_tiles:
            print("\t Tiles: ", tiles)  # only print if these folders should be created
            # update metadata to include destination for each tile
            metadata["dest"] = [
                os.path.join(self.outdirs["sorted_wells"], f"row{row}_well{well}", tile)
                for row, well, tile in zip(metadata.Row, metadata.Well, metadata.tiles, strict=False)
            ]
        else:
            metadata["dest"] = [
                os.path.join(self.outdirs["sorted_wells"], f"row{row}_well{well}")
                for row, well in zip(metadata.Row, metadata.Well, strict=False)
            ]

        # unique directories for each tile
        unique_dirs = list(set(metadata.dest.to_list()))

        for _dir in unique_dirs:
            if not os.path.exists(_dir):
                os.makedirs(_dir)

        # copy/link the images to their new names
        self._copy_files(metadata=metadata)

    def sort_timepoints(self, sort_wells: bool = False) -> None:
        """Sort parsed images by timepoint.

        Generates a folder tree where each timepoint has its own folder containing all images captured at that timepoint.
        If ``sort_wells`` is ``True``, an additional layer is added where images
        from the same well are grouped within each timepoint.

        Args:
            sort_wells: Whether to sort images into per-well directories within each timepoint.
        """

        # create output directory
        self._define_outdir(name="sorted_timepoints")

        # get all new file names
        if "metadata" in self.__dict__:
            metadata = self.metadata
        else:
            metadata = self.generate_metadata()

        metadata["tiles"] = [self._get_tile_id(x) for x in metadata.new_file_name.to_list()]

        # get unique rows, wells and tiles
        rows = list(set(metadata.Row.to_list()))
        wells = list(set(metadata.Well.to_list()))
        tiles = list(set(metadata.tiles.to_list()))
        timepoints = list(set(metadata.Timepoint.to_list()))

        print("Found the following image specs: ")
        print("\t Timepoints: ", timepoints)
        print("\t Rows: ", rows)
        print("\t Wells: ", wells)
        print("\t Tiles: ", tiles)

        if sort_wells:
            # update metadata to include destination for each tile
            metadata["dest"] = [
                os.path.join(self.outdirs["sorted_timepoints"], timepoint, f"{row}_{well}")
                for row, well, timepoint in zip(metadata.Row, metadata.Well, metadata.Timepoint, strict=False)
            ]
        else:
            metadata["dest"] = [
                os.path.join(self.outdirs["sorted_timepoints"], timepoint) for timepoint in metadata.Timepoint
            ]

        # unique directories for each tile
        unique_dirs = list(set(metadata.dest.to_list()))

        for _dir in unique_dirs:
            if not os.path.exists(_dir):
                os.makedirs(_dir)

        # copy/link the images to their new names
        self._copy_files(metadata=metadata)


class CombinedPhenixParser(PhenixParser):
    """
    Parse Phenix experiments where multiple exports should be combined into one dataset.

    This is typically used when individual tiles were not imaged during
    acquisition, for example because of a focus failure. Instead of repeating
    the entire measurement, the missing tiles can be acquired in a separate
    experiment and combined with the original dataset.

    This class inherits from the PhenixParser class and extends it by adding the functionality to combine multiple experiments into one dataset.
    These individual experiments need to be placed together in the following structure:

        .. code-block:: none

            <experiment_name>/
            └── experiments_to_combine/
                ├── experiment_1/
                ├── experiment_2/
                ├── experiment_3/
                └── ...

        - *<experiment_name>* can be chosen freely.
        - *experiments_to_combine* is a folder containing all experiments that should be combined. This folder should be placed in the main experiment directory.
        - *experiment_n* always refers to the complete folder as generated by Harmony when exporting Phenix data without any further modifications.

    The experiments will be combined in the order of their creation date and time.
    If two experiments contain images in the same position, the parser will keep the images from the first experiment.
    """

    directory_combined_measurements = "experiments_to_combine"

    def __init__(
        self,
        experiment_dir: str,
        flatfield_exported: bool = True,
        use_symlinks: bool = True,
        compress_rows: bool = False,
        compress_cols: bool = False,
        overwrite: bool = False,
    ) -> None:
        """
        Args:
            experiment_dir: Directory containing the exported Phenix experiment.
            flatfield_exported: Whether flatfield corrected images were exported.
            use_symlinks: Whether to use symbolic links for parsed images.
            compress_rows: Whether to merge all plate rows into a single parsed row.
            compress_cols: Whether to merge all wells into a single parsed column.
            overwrite: Whether to overwrite existing files during the parsing process.
        """
        self.experiment_dir = experiment_dir
        self.get_datasets_to_combine()
        super().__init__(
            experiment_dir, flatfield_exported, use_symlinks, compress_rows, compress_cols, overwrite=overwrite
        )

    def _get_xml_path(self):
        """Automatically get the XML files from all phenix experiments that should be combined."""
        # directory depends on if flatfield images were exported or not
        # these generated folder structures are hard coded during phenix export, do not change
        # get index file of the first phenix dir(this is our main experiment!)
        if self.flatfield_status:
            index_file_names = ["Index.xml", "Index.ref.xml"]
            for index_file_name in index_file_names:
                index_file = os.path.join(self.phenix_dirs[0], "Images", index_file_name)
                if os.path.isfile(index_file):
                    break
        else:
            index_file_names = ["Index.xml", "Index.idx.xml"]
            for index_file_name in index_file_names:
                index_file = os.path.join(self.phenix_dirs[0], "Images", index_file_name)
                if os.path.isfile(index_file):
                    break

        # perform sanity check if file exists else exit
        if not os.path.isfile(index_file):
            sys.exit(f"Can not find index file at path: {index_file}")

        return index_file

    def _get_input_dir(self) -> str:
        """Automatically get the subfolder where the exported image files are located."""

        # directory depends on if flatfield images were exported or not
        # these generated folder structures are hard coded during phenix export, do not change
        # for the combined exported the first experiment is always used (they should have the same exported XML file anyways for reading)
        if self.flatfield_status:
            input_dir = f"{self.phenix_dirs[0]}/Images/flex"
        else:
            input_dir = f"{self.phenix_dirs[0]}/Images"

        # perform sanity check if file exists else exit
        if not os.path.isdir(input_dir):
            sys.exit(f"Can not find directory containing images to parse: {input_dir}")

        return input_dir

    def get_datasets_to_combine(self) -> None:
        """Get all Phenix experiments from subdirectories that should be combined."""
        input_path = f"{self.experiment_dir}/{self.directory_combined_measurements}"

        # get phenix directories that need to be combined together
        phenix_dirs = os.listdir(input_path)

        # only get the phenix dirs that match the pattern of a phenix experiment (ie they contain a time stamp)
        pattern = r"\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2}"
        phenix_dirs = [x for x in phenix_dirs if re.search(pattern, x)]

        # Extract the date and time information from each file name
        dates_times = [
            re.search(pattern, file_name).group(0)
            for file_name in phenix_dirs
            if re.search(pattern, file_name)  # Ensure it matched before accessing .group()
        ]

        # Sort the file names based on the extracted date and time information
        sorted_phenix_dirs = [file_name for _, file_name in sorted(zip(dates_times, phenix_dirs, strict=False))]

        self.phenix_dirs = [f"{input_path}/{phenix_dir}" for phenix_dir in sorted_phenix_dirs]

    def _get_phenix_metadata(self) -> pd.DataFrame:
        """Read combined metadata from all Phenix experiments.

        If multiple exports contain the same logical tile within a well, the
        earliest export is preserved after stage positions are clustered into
        shared tile coordinates.
        """
        ###
        # read metadata from all experiments and merge into one file
        # note: if more than one image exists at the same logical tile position
        # then the first image acquired will be preserved based on the timestamps
        # in the exported phenix measurement names
        ####

        # define under what path the actual exported images will be found
        # this is hard coded through phenix export script, do not change
        if self.flatfield_status:
            xml_path = "Images/Index.ref.xml"
            append_string = "Images/flex"
        else:
            append_string = "Images"
            xml_path = "Index.idx.xml"

        # read all metadata
        metadata = {}
        for phenix_dir in self.phenix_dirs:
            df = self._read_phenix_xml(f"{phenix_dir}/{xml_path}")
            df.loc[:, "source"] = f"{phenix_dir}/{append_string}"  # update source with the correct strings
            metadata[phenix_dir] = df

        metadata_combined = pd.concat(metadata.values(), ignore_index=True)
        metadata_combined = self._assign_tile_positions(metadata_combined)

        # merge generated metadata files together (order of what is preserved is according to calcualted creation times above)
        for i, key in enumerate(metadata.keys()):
            df = metadata_combined[metadata_combined["source"] == metadata[key]["source"].iloc[0]]
            df = df.set_index(["Row", "Well", "Zstack", "Timepoint", "X_pos", "Y_pos", "Channel"])
            if i == 0:
                metadata_merged = df
            else:
                metadata_merged = metadata_merged.combine_first(df)

        metadata_merged = metadata_merged.reset_index()

        # return generated dataframe
        print("merged metadata generated from all passed phenix experiments.")
        return metadata_merged

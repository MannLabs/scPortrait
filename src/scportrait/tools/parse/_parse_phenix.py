"""
parse
====================================

Contains functions to parse imaging data aquired on an OperaPhenix or Operetta into a usable formats for downstream pipelines.
"""

import os
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime

import numpy as np
import pandas as pd
from tifffile import imread, imwrite
from tqdm import tqdm


def _get_child_name(elem):
    return elem.split("}")[1]


class PhenixParser:
    def __init__(
        self,
        experiment_dir,
        flatfield_exported=True,
        export_symlinks=True,
        compress_rows=False,
        compress_cols=False,
    ) -> None:
        self.experiment_dir = experiment_dir
        self.export_symlinks = export_symlinks
        self.flatfield_status = flatfield_exported
        self.compress_rows = compress_rows
        self.compress_cols = compress_cols

        if self.compress_rows:
            print(
                "The rows found in the phenix layout will be compressed into one row after parsing the images, r and c indicators will be adjusted accordingly."
            )
        if self.compress_cols:
            print(
                "The wells found in the phenix layout will be compressed into one column after parsing the images, r and c indicators will be adjusted accordingly."
            )

        self.xml_path = self.get_xml_path()
        self.image_dir = self.get_input_dir()
        self.channel_lookup = self.get_channel_metadata(self.xml_path)
        self.metadata = None

    def get_xml_path(self):
        # directory depends on if flatfield images were exported or not
        # these generated folder structures are hard coded during phenix export, do not change
        if self.flatfield_status:
            index_file = os.path.join(self.experiment_dir, "Images", "Index.ref.xml")
        else:
            index_file = os.path.join(self.experiment_dir, "Index.idx.xml")

        # perform sanity check if file exists else exit
        if not os.path.isfile(index_file):
            sys.exit(f"Can not find index file at path: {index_file}")

        return index_file

    def get_input_dir(self):
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

    def define_outdir(self, name="parsed_images"):
        setattr(self, f"outdir_{name}", f"{self.experiment_dir}/{name}")

        # if output directory did not exist create it
        if not os.path.isdir(getattr(self, f"outdir_{name}")):
            os.makedirs(getattr(self, f"outdir_{name}"))

    def get_channel_metadata(self, xml_path) -> pd.DataFrame:
        index_file = xml_path

        # get channel names and ids and generate a lookup table
        cmd = """grep -E -m 20 '<ChannelName>|<ChannelID>' '""" + index_file + """'"""
        results = (
            subprocess.check_output(cmd, shell=True)
            .decode("utf-8")
            .strip()
            .split("\r\n")
        )

        results = [x.strip() for x in results]
        channel_ids = [
            x.split(">")[1].split("<")[0] for x in results if x.startswith("<ChannelID")
        ]
        channel_names = [
            x.split(">")[1].split("<")[0].replace(" ", "")
            for x in results
            if x.startswith("<ChannelName")
        ]

        channel_ids = list(set(channel_ids))
        channel_ids.sort()
        channel_names = channel_names[0 : len(channel_ids)]

        lookup = pd.DataFrame({"id": list(channel_ids), "label": list(channel_names)})

        print("Experiment contains the following image channels: ")
        print(lookup, "\n")

        # save lookup file to csv
        lookup.to_csv(f"{self.experiment_dir}/channel_lookuptable.csv")
        print(
            f"Channel Lookup table saved to file at {self.experiment_dir}/channel_lookuptable.csv\n"
        )

        self.channel_names = channel_names
        return lookup

    def read_phenix_xml(self, xml_path):
        # initialize lists to save results into
        rows = []
        cols = []
        fields = []
        planes = []
        channel_ids = []
        channel_names = []
        flim_ids = []
        timepoints = []
        x_positions = []
        y_positions = []
        times = []

        # extract information from XML file
        tree = ET.parse(xml_path)
        root = tree.getroot()

        for i, child in enumerate(root):
            if _get_child_name(child.tag) == "Images":
                images = root[i]

        for i, image in enumerate(images):
            for ix, child in enumerate(image):
                tag = _get_child_name(child.tag)
                if tag == "Row":
                    rows.append(child.text)
                if tag == "Col":
                    cols.append(child.text)
                if tag == "FieldID":
                    fields.append(child.text)
                if tag == "PlaneID":
                    planes.append(child.text)
                if tag == "ChannelID":
                    channel_ids.append(child.text)
                if tag == "ChannelName":
                    channel_names.append(child.text)
                if tag == "FlimID":
                    flim_ids.append(child.text)
                if tag == "TimepointID":
                    timepoints.append(child.text)
                if tag == "PositionX":
                    x_positions.append(child.text)
                if tag == "PositionY":
                    y_positions.append(child.text)
                if tag == "AbsTime":
                    times.append(child.text)

        rows = [str(x).zfill(2) for x in rows]
        cols = [str(x).zfill(2) for x in cols]
        fields = [str(x).zfill(2) for x in fields]
        planes = [str(x).zfill(2) for x in planes]
        timepoints = [int(x) + 1 for x in timepoints]

        image_names = []
        for row, col, field, plane, channel_id, timepoint, flim_id in zip(
            rows, cols, fields, planes, channel_ids, timepoints, flim_ids
        ):
            image_names.append(
                f"r{row}c{col}f{field}p{plane}-ch{channel_id}sk{timepoint}fk1fl{flim_id}.tiff"
            )

        # remove extra spaces from channel names
        channel_names = [x.replace(" ", "") for x in channel_names]

        # convert date/time into useful format
        dates = [x.split("T")[0] for x in times]
        _times = [x.split("T")[1] for x in times]
        _times = [
            (x.split("+")[0].split(".")[0] + "+" + x.split("+")[1].replace(":", ""))
            for x in _times
        ]
        time_final = [x + " " + y for x, y in zip(dates, _times)]

        datetime_format = "%Y-%m-%d %H:%M:%S%z"
        time_unix = [datetime.strptime(x, datetime_format) for x in time_final]
        time_unix = [datetime.timestamp(x) for x in time_unix]

        # update file name if flatfield exported images are to be used
        if self.flatfield_status:
            image_names = [f"flex_{x}" for x in image_names]

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
        df["filename"] = df["filename"].astype(
            str
        )  # ensure this is a string otherwise it can cause issues later on
        df["date"] = df["date"].astype(str)
        df["time"] = df["time"].astype(str)
        df["Channel"] = df["Channel"].astype(str)

        return df

    def get_phenix_metadata(self):
        return self.read_phenix_xml(self.xml_path)

    def generate_new_filenames(self, metadata):
        # convert position values to numeric to ensure proper sorting
        metadata["X"] = [float(x) for x in metadata.X]
        metadata["Y"] = [float(x) for x in metadata.Y]

        # convert X_positions into row and col values
        metadata["X_pos"] = None
        X_values = metadata.X.value_counts().index.to_list()
        X_values = np.sort(X_values)
        for i, x in enumerate(X_values):
            metadata.loc[metadata.X == x, "X_pos"] = i

        # get y positions
        metadata["Y_pos"] = None
        Y_values = metadata.Y.value_counts().index.to_list()
        Y_values = np.sort(
            Y_values
        )  # ensure that the values are numeric and not string
        for i, y in enumerate(Y_values):
            metadata.loc[metadata.Y == y, "Y_pos"] = i

        # get number of rows and wells and adjust labelling if specific entries need to be compressed
        wells = metadata.Well.value_counts().index.to_list()
        rows = metadata.Row.value_counts().index.to_list()

        wells.sort()
        rows.sort(
            reverse=True
        )  # invert because the image quadrant beginns in the bottom left

        if self.compress_rows:
            for well in wells:
                for i, row in enumerate(rows):
                    if i == 0:
                        continue
                    else:
                        max_y = metadata.loc[
                            ((metadata.Well == well) & (metadata.Row == rows[0]))
                        ].Y_pos.max()
                        metadata.loc[
                            (metadata.Well == well) & (metadata.Row == row), "Y_pos"
                        ] = (
                            metadata.loc[
                                (metadata.Well == well) & (metadata.Row == row), "Y_pos"
                            ]
                            + int(max_y)
                            + int(1)
                        )
                        metadata.loc[
                            (metadata.Well == well) & (metadata.Row == row), "Row"
                        ] = rows[0]

        if self.compress_cols:
            for i, well in enumerate(wells):
                if i == 0:
                    continue
                else:
                    max_x = metadata.loc[(metadata.Well == wells[0])].X_pos.max()
                    metadata.loc[(metadata.Well == well), "X_pos"] = (
                        metadata.loc[(metadata.Well == well), "X_pos"]
                        + int(max_x)
                        + int(1)
                    )
                    metadata.loc[(metadata.Well == well), "Well"] = wells[0]

        metadata.X_pos = [str(int(x)).zfill(3) for x in metadata.X_pos]
        metadata.Y_pos = [str(int(x)).zfill(3) for x in metadata.Y_pos]
        metadata.Timepoint = [str(x).zfill(3) for x in metadata.Timepoint]
        metadata.Zstack = [str(x).zfill(2) for x in metadata.Zstack]

        # generate new file names
        for i in range(metadata.shape[0]):
            _row = metadata.loc[i, :]
            name = "Timepoint{}_Row{}_Well{}_{}_zstack{}_r{}_c{}.tif".format(
                _row.Timepoint,
                _row.Row,
                _row.Well,
                _row.Channel,
                _row.Zstack,
                _row.Y_pos,
                _row.X_pos,
            )
            name = name
            metadata.loc[i, "new_file_name"] = name

        return metadata

    def get_tile_id(self, string):
        pattern = r"_r(\d+)_c(\d+)\.tif"
        match = re.search(pattern, string)
        if match:
            row = match.group(1)
            col = match.group(2)
            return f"r{row}_c{col}"
        else:
            return None

    def generate_metadata(self):
        metadata = self.get_phenix_metadata()
        metadata_new = self.generate_new_filenames(metadata)

        # save results to self for easy access
        self.metadata = metadata_new

        return metadata_new

    def check_for_missing_files(self, metadata=None, return_values=False):
        def _generate_missing_file_names(
            x_positions, y_positions, timepoint, row, well, channels, zstacks
        ):
            """Helper function to generate missing file names given x_positions and y_positions."""

            _missing_tiles = []

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
            if "metdata" in self.__dict__:
                metadata = self.metadata
            else:
                metadata = self.generate_metadata()

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
        missing_tiles = []
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
                        Warning(
                            f"Entire well {well} is missing for timepoint {timepoint}."
                        )
                        missing_tiles = missing_tiles + _generate_missing_file_names(
                            x_range, y_range, timepoint, row, well, channels, zstacks
                        )
                        continue

                    for x_pos in x_range:
                        _check = ___df[___df.X_pos == x_pos]
                        _y_pos = [
                            y_pos for y_pos in y_range if y_pos not in set(_check.Y_pos)
                        ]

                        if len(_y_pos) > 0:
                            missing_tiles = (
                                missing_tiles
                                + _generate_missing_file_names(
                                    [x_pos],
                                    _y_pos,
                                    timepoint,
                                    row,
                                    well,
                                    channels,
                                    zstacks,
                                )
                            )

                    for y_pos in y_range:
                        _check = ___df[___df.Y_pos == y_pos]
                        _x_pos = [
                            x_pos for x_pos in x_range if x_pos not in set(_check.X_pos)
                        ]

                        if len(_y_pos) > 0:
                            missing_tiles = (
                                missing_tiles
                                + _generate_missing_file_names(
                                    _x_pos,
                                    [y_pos],
                                    timepoint,
                                    row,
                                    well,
                                    channels,
                                    zstacks,
                                )
                            )

        if len(missing_tiles) == 0:
            print("No missing tiles found.")
        else:
            # get size of missing images that need to be replaced
            image = imread(os.path.join(metadata["source"][0], metadata["filename"][0]))
            image[:] = int(0)
            self.black_image = image

            print(
                f"The found missing tiles need to be replaced with black images of the size {image.shape}."
            )

        self.missing_images = missing_tiles

        if return_values:
            return missing_tiles

    def replace_missing_images(self):
        # calculate missing images if not already done
        if "missing_images" not in self.__dict__:
            self.check_for_missing_files()

        # initialize output directory if not alreadt done
        if "outdir_parsed_images" not in self.__dict__:
            self.define_outdir(name="parsed_images")

        # if there are missing images replace them with black images
        if len(self.missing_images) > 0:
            for missing_image in self.missing_images:
                print(f"Creating black image with name: {missing_image}")
                imwrite(
                    os.path.join(self.outdir_parsed_images, missing_image),
                    self.black_image,
                )

            print(
                f"All missing images successfully replaced with black images of the dimension {self.black_image.shape}"
            )

    def define_copy_functions(self):
        # define function for copying depending on if symlinks should be used or not
        if self.export_symlinks:

            def copyfunction(input, output):
                try:
                    os.symlink(input, output)
                except OSError:
                    return ()
        else:

            def copyfunction(input, output):
                shutil.copyfile(input, output)

        self.copyfunction = copyfunction

    def copy_files(self, metadata):
        """
        Copy files from the source directory to the output directory. The new file names are defined in the metadata.

        Parameters
        ----------

        metadata : pd.DataFrame
            Expected columns are: filename, new_file_name, source, dest

        Returns
        -------
        None

        """
        print("Starting copy process...")
        self.define_copy_functions()

        # actually perform the copy process
        for old, new, source, dest in tqdm(
            zip(
                metadata.filename.tolist(),
                metadata.new_file_name.tolist(),
                metadata.source.tolist(),
                metadata.dest.tolist(),
            ),
            total=len(metadata.new_file_name.tolist()),
            desc="Copying files",
        ):
            # define old and new paths for copy process
            old_path = os.path.join(source, old)
            new_path = os.path.join(dest, new)

            # check if old path exists
            if os.path.exists(old_path):
                self.copyfunction(old_path, new_path)
            else:
                print("Error: ", old_path, "not found.")
        print("Copy process completed.")

    def save_metadata(self, metadata):
        # save to csv file
        metadata.to_csv(f"{self.experiment_dir}/metadata_image_parsing.csv")
        print(
            f"Metadata used to parse images saved to file {self.experiment_dir}/metadata_image_parsing.csv"
        )

    def parse(self):
        """Complete parsing of phenix experiment including checking for and replacing missing images."""
        # create output directory
        self.define_outdir(name="parsed_images")

        # get metadata for the images we want to parse
        metadata = self.generate_metadata()

        # set destination for copying
        metadata["dest"] = getattr(self, "outdir_parsed_images")

        # copy/link the images to their new names
        self.copy_files(metadata=metadata)

        # check for missing images and replace them
        self.check_for_missing_files(metadata=metadata)
        self.replace_missing_images()
        self.save_metadata(metadata)

    def sort_wells(self, sort_tiles=False):
        """Sorts parsed images according to their well.

        Generates a folder tree where each well has its own folder containing all images from that well.
        If sort_tiles = True an additional layer will be added to the tree where all images obtained from the same FOV are sorted into a unique subfolder.

        Parameters
        ----------
        sort_tiles : bool, optional
            if the images should be sorted into individual directories according to FOV in addition to well, by default False
        """

        # create output directory
        self.define_outdir(name="sorted_wells")

        # get all new file names
        if "metdata" in self.__dict__:
            metadata = self.metadata
        else:
            metadata = self.generate_metadata()

        metadata["tiles"] = [
            self.get_tile_id(x) for x in metadata.new_file_name.to_list()
        ]

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
                os.path.join(
                    getattr(self, "outdir_sorted_wells"), f"row{row}_well{well}", tile
                )
                for row, well, tile in zip(metadata.Row, metadata.Well, metadata.tiles)
            ]
        else:
            metadata["dest"] = [
                os.path.join(
                    getattr(self, "outdir_sorted_wells"), f"row{row}_well{well}"
                )
                for row, well in zip(metadata.Row, metadata.Well)
            ]

        # unique directories for each tile
        unique_dirs = list(set(metadata.dest.to_list()))

        for _dir in unique_dirs:
            if not os.path.exists(_dir):
                os.makedirs(_dir)

        # copy/link the images to their new names
        self.copy_files(metadata=metadata)

    def sort_timepoints(self, sort_wells=False):
        """Sorts parsed images according to their timepoint.

        Generates a folder tree where each timepoint has its own folder containing all images captured at that timepoint.
        If sort_wells = True an additional layer will be added to the tree where all images obtained from the same well are sorted into a unique subfolder according to timepoint.

        Parameters
        ----------
        sort_wells : bool, optional
            if the images should be sorted into individual directories according to well in addition to timepoint, by default False
        """

        # create output directory
        self.define_outdir(name="sorted_timepoints")

        # get all new file names
        if "metdata" in self.__dict__:
            metadata = self.metadata
        else:
            metadata = self.generate_metadata()

        metadata["tiles"] = [
            self.get_tile_id(x) for x in metadata.new_file_name.to_list()
        ]

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
                os.path.join(
                    getattr(self, "outdir_sorted_timepoints"),
                    timepoint,
                    f"{row}_{well}",
                )
                for row, well, timepoint in zip(
                    metadata.Row, metadata.Well, metadata.Timepoint
                )
            ]
        else:
            metadata["dest"] = [
                os.path.join(getattr(self, "outdir_sorted_timepoints"), timepoint)
                for timepoint in metadata.Timepoint
            ]

        # unique directories for each tile
        unique_dirs = list(set(metadata.dest.to_list()))

        for _dir in unique_dirs:
            if not os.path.exists(_dir):
                os.makedirs(_dir)

        # copy/link the images to their new names
        self.copy_files(metadata=metadata)


class CombinedPhenixParser(PhenixParser):
    directory_combined_measurements = "experiments_to_combine"

    def __init__(
        self,
        experiment_dir,
        flatfield_exported=True,
        export_symlinks=True,
        compress_rows=False,
        compress_cols=False,
    ) -> None:
        self.experiment_dir = experiment_dir
        self.get_datasets_to_combine()

        super().__init__(
            experiment_dir,
            flatfield_exported,
            export_symlinks,
            compress_rows,
            compress_cols,
        )

    def get_xml_path(self):
        # directory depends on if flatfield images were exported or not
        # these generated folder structures are hard coded during phenix export, do not change
        # get index file of the first phenix dir(this is our main experiment!)
        if self.flatfield_status:
            index_file = f"{self.phenix_dirs[0]}/Images/Index.ref.xml"
        else:
            index_file = f"{self.phenix_dirs[0]}/Index.idx.xml"

        # perform sanity check if file exists else exit
        if not os.path.isfile(index_file):
            sys.exit(f"Can not find index file at path: {index_file}")

        return index_file

    def get_input_dir(self):
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

    def get_datasets_to_combine(self):
        input_path = f"{self.experiment_dir}/{self.directory_combined_measurements}"

        # get phenix directories that need to be comined together
        phenix_dirs = os.listdir(input_path)

        # only get the phenix dirs that match the pattern of a phenix experiment (ie they contain a time stamp)
        pattern = r"\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2}"
        phenix_dirs = [x for x in phenix_dirs if re.search(pattern, x)]

        # Extract the date and time information from each file name
        dates_times = [
            re.search(pattern, file_name).group(0)
            for file_name in phenix_dirs
            if re.search(
                pattern, file_name
            )  # Ensure it matched before accessing .group()
        ]

        # Sort the file names based on the extracted date and time information
        sorted_phenix_dirs = [
            file_name for _, file_name in sorted(zip(dates_times, phenix_dirs))
        ]

        self.phenix_dirs = [
            f"{input_path}/{phenix_dir}" for phenix_dir in sorted_phenix_dirs
        ]

    def get_phenix_metadata(self):
        ###
        # read metadata from all experiments and merge into one file
        # note: if more than one image exists at a specific position then the first image aquired will be preserved based on the timestamps in the exported phenix measurement names
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
            df = self.read_phenix_xml(f"{phenix_dir}/{xml_path}")
            df = df.set_index(
                ["Row", "Well", "Zstack", "Timepoint", "X", "Y", "Channel"]
            )
            df.loc[:, "source"] = (
                f"{phenix_dir}/{append_string}"  # update source with the correct strings
            )
            metadata[phenix_dir] = df

        # merge generated metadata files together (order of what is preserved is according to calcualted creation times above)
        for i, key in enumerate(metadata.keys()):
            if i == 0:
                metadata_merged = metadata[key]
            else:
                metadata_merged = metadata_merged.combine_first(metadata[key])

        metadata_merged = metadata_merged.reset_index()

        # return generated dataframe
        print("merged metadata generated from all passed phenix experiments.")
        return metadata_merged

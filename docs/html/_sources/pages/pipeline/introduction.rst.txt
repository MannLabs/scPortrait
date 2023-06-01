Introduction
#############









This is old stuff here that we will want to put somehwere else

.. code-block:: python
    
    from vipercore.pipeline.project import Project
    from vipercore.pipeline.workflows import ShardedWGASegmentation
    from vipercore.pipeline.extraction import HDF5CellExtraction
    from vipercore.pipeline.selection import LMDSelection
    from vipercore.processing.utils import download_testimage

    import numpy as np
    import os
    import csv
    
.. code-block:: python

    location = download_testimage("/mnt/dss_fast/datasets/2021_09_03_hdf5_extraction_developement")
    print(location)
 
.. code-block:: python
    
    project_location = "/Users/georgwallmann/Documents/testdaten/viper_library_test"

    project = Project(project_location, 
                        config_path="settings.yaml",
                        overwrite=True,
                        segmentation_f=ShardedWGASegmentation,
                        extraction_f=HDF5CellExtraction,
                        selection_f=LMDSelection)

    project.load_input_from_file(location)
    project.segment()
    project.extract()

    csv_location = os.path.join(project_location, "segmentation/classes.csv")

    with open(csv_location) as f:
        reader = csv.reader(f)
        data = list(reader)
        data = np.array([int(el[0]) for el in data])

    cells_to_select = [{"name": "dataset1", "classes": list(data), "well": "A1"}]

    # calibration marker should be defined as (row, column)
    marker_0 = np.array([-10,-10])
    marker_1 = np.array([-10,3010])
    marker_2 = np.array([3010,1500])

    calibration_marker = np.array([marker_0, marker_1, marker_2])

    project.select(cells_to_select, calibration_marker)
    
.. code-block:: yaml
    
    ---
    name: "WGA confluent segmentation"
    input_channels: 2

    # Define remapping of channels. 
    # For example use 1, 0, 2 to change the order of the first and the second channel.
    channel_remap: 0,1
    
    ShardedWGASegmentation:
        input_channels: 2
        # average number of pixel per shard. 
        # shards of size 10.000 * 10.000 pixel are recommended which equals a 100,000,000 pixel.
        # shard_size: 100000000  
        # can be adapted to memory and computation needs.
        shard_size: 2250000  

        # number of shards / tiles segmented at the same size. 
        # should be adapted to the maximum amount allowed by memory.
        threads: 4

        # upper and lower percentile for the normalization by percentiles.
        lower_quantile_normalization:   0.001
        upper_quantile_normalization:   0.999

        # median filter size in pixel
        median_filter_size:   4

        # parameters specific to the nucleus segmentation 
        nucleus_segmentation:
            # quantile normalization of dapi channel before local tresholding. 
            # strong normalization (0.05,0.95) can help with nuclear speckles.
            lower_quantile_normalization:   0.03 
            upper_quantile_normalization:   0.92 

            # Size of pixel disk used for median, should be uneven
            median_block: 41

            # The image can be downsampled before calculating the median which speeds up calculation tremendously.
            # If median step is set, only ever nth pixel is considered, similar to image[::median_step, ::median_step]
            # Please note, that the median_block can be halved if a median_step of 2 is choosen. 
            median_step: 3 

            # Threshold above local median for nuclear segmentation.
            # If threshold or median_block is missing, otsu global tresholding will be used
            threshold: 0.10 

            # minimum distance between two nucleis in pixel
            min_distance: 10 

            # minimum distance between two nucleis in pixel
            peak_footprint: 7 

            # Erosion followed by Dilation to remove speckels, size in pixels, should be uneven
            speckle_kernel: 5 

            # final dilation of pixel mask
            dilation: 0        

            # minimum nucleus area in pixel
            min_size: 150 

            # maximum nucleus area in pixel
            max_size: 1000 

            # minimum nucleus contact with background
            contact_filter: 0.8 

        # parameters specific to the nucleus segmentation 
        wga_segmentation:

            lower_quantile_normalization: 0.2
            upper_quantile_normalization: 0.98

            # erosion and dilation are used for speckle removal and shrinking / dilation
            # for no change in size choose erosion = dilation, for larger masks, increase the mask erosion
            erosion: 3 
            dilation: 3 

            # Threshold for wga segmentation.
            # If threshold is missing, otsu global tresholding will be used
            #threshold: 0.15 

            min_clip: 0.5
            max_clip: 0.9

            min_size: 200
            max_size: 4000

        # chunk size for chunked HDF5 storage. is needed for correct caching and high performance reading. should be left at 50.
        chunk_size: 50

    HDF5CellExtraction:

        compression: True

        threads: 80 # threads used in multithreading

        image_size: 128 # image size in pixel

        cache: "/Users/georgwallmann/Documents/testdaten/temp"

        # Define remapping of channels. 
        # For example use 1, 0, 2 to change the order of the first and the second channel.
        channel_remap: 0,1,2,3

        hdf5_rdcc_nbytes: 5242880000 # 5gb 1024 * 1024 * 5000 

        hdf5_rdcc_w0: 1

        hdf5_rdcc_nslots: 50000

    LMDSelection:
        processes: 10

        # defines the channel used for generating cutting masks
        # segmentation.hdf5 => labels => segmentation_channel
        # When using WGA segmentation:
        #    0 corresponds to nuclear masks
        #    1 corresponds to cytosolic masks.
        segmentation_channel: 0

        # dilation of the cutting mask in pixel 
        shape_dilation: 2

        # number of datapoints which are averaged for smoothing
        # the number of datapoints over an distance of n pixel is 2*n
        smoothing_filter_size: 10

        # fold reduction of datapoints for compression
        poly_compression_factor: 30
    
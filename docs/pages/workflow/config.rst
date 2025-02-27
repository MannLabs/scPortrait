.. _config:

Config Files
============

The configuration file is a ``.yml`` file which specifies all of the parameters for each of the
methods chosen in a specific scPortrait Project run.

.. code:: yaml
    :caption: Example Configuration File

    ---
    name: "Cellpose Segmentation"
    ShardedCytosolSegmentationCellpose:
        shard_size: 2000000 # maxmimum number of pixel per tile
        overlap_px: 100
        nGPUs: 1
        threads: 2 # number of shards / tiles segmented at the same size. should be adapted to the maximum amount allowed by memory.
        cache: "."
        nucleus_segmentation:
            model: "nuclei"
        cytosol_segmentation:
            model: "cyto2"
        match_masks: True
        filter_masks_size: False
    HDF5CellExtraction:
        threads: 80 # threads used in multithreading
        image_size: 128 # image size in pixel
        normalize_output: True
        normalization_range: (0.01, 0.99)
        cache: "."
    CellFeaturizer:
        batch_size: 900
        dataloader_worker_number: 10 #needs to be 0 if using cpu
        inference_device: "cpu"
    LMDSelection:
        threads: 20
        cache: "."
        processes_cell_sets: 10
        # defines the channel used for generating cutting masks
        # segmentation.hdf5 => labels => segmentation_channel
        # When using WGA segmentation:
        #    0 corresponds to nuclear masks
        #    1 corresponds to cytosolic masks.
        segmentation_channel: "seg_all_nucleus"
        shape_dilation: 16
        smoothing_filter_size: 25
        rdp: 0.6
        path_optimization: "hilbert"
        greedy_k: 15
        hilbert_p: 7

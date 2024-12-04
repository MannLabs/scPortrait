.. _config:

Config Files
============

The configuration file is a ``.yml`` file which specifies all of the parameters for each of the 
methods chosen in a specific scPortrait Project run.

.. code:: yaml
    :caption: Example Configuration File

    ---
    name: "Cellpose Segmentation"
    CytosolSegmentationCellpose:
        cache: "."
        nucleus_segmentation:
            model: "nuclei"
        cytosol_segmentation:
            model: "cyto2"
    HDF5CellExtraction:
        threads: 80 # threads used in multithreading
        image_size: 400 # image size in pixel
        normalization_range: None #turn of percentile normalization for cells -> otherwise normalise out differences for the alexa647 channel
        cache: "."
    CellFeaturizer:
        channel_selection: 4
        batch_size: 900
        dataloader_worker_number: 0 #needs to be 0 if using cpu
        inference_device: "cpu"
        label: "Ch3_Featurization"
    LMDSelection:
        processes: 20
        segmentation_channel: 0
        shape_dilation: 16
        smoothing_filter_size: 25
        poly_compression_factor: 30
        path_optimization: "hilbert"
        greedy_k: 15
        hilbert_p: 7

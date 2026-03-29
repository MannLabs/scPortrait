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
        normalization_range: [0.01, 0.99]
        cache: "."
        target_ram_utilization: 0.85 # fraction of total system RAM the extraction job should target
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
        rdp_epsilon: 0.6
        path_optimization: "hilbert"
        greedy_k: 15
        hilbert_p: 7

For ``HDF5CellExtraction``, scPortrait can run multiple worker processes to prepare
single-cell image batches while the main process writes results to the output HDF5
file. On large datasets, preparing batches can be faster than writing them to
disk, which would otherwise allow completed batch results to accumulate in
memory. To keep this manageable, the extraction workflow can limit how many
completed batch results are buffered in memory at the same time.

When ``max_inflight_result_batches`` is not provided explicitly, scPortrait
calibrates it automatically from the first wave of worker batches together with the
configured ``target_ram_utilization``. This calibration estimates returned batch
payload size together with the parent-process RSS, then chooses an in-flight batch limit
that aims to stay within the requested RAM budget for the job.

If the RAM budget would imply a value smaller than the active worker count,
scPortrait keeps the in-flight batch limit at least as large as the number of
workers and emits a warning in the log. In that case, the correct way to reduce
memory further is to lower ``threads``.

``flush_every`` controls how often the output HDF5 file is flushed and garbage
collection is run during extraction. If it is not configured explicitly,
scPortrait derives it automatically from the effective in-flight batch limit.

Normalization settings are also important because they directly affect the
extracted single-cell image values used downstream. If you need guidance on
choosing ``normalize_output`` or ``normalization_range``, refer to the tutorial
notebook "Fine-tuning the single-cell image extraction" in the tutorials
section.

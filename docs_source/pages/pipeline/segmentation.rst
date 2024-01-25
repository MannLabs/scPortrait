.. _segmentation:

Segmentation
============

Segmentation is an essential step in the SPARCSpy workflow. The goal of segmentation is to generate a mask which maps the pixels of the input image to individual cells, which are then assigned a unique ``cell id``. Background pixels are assigned the value ``0``. In SPARCSpy we distinguish two different types of segmentation masks: nuclear and cytosolic. Nuclear masks <...> Cytosolic masks <...>.

.. |pic1| image:: ../images/input_image.png
   :width: 100%

.. |pic2| image:: ../images/nucleus_mask.png
   :width: 100%

.. |pic3| image:: ../images/cytosol_mask.png
   :width: 100%

+-----------------------+-----------------------+-----------------------+
| Input Image           | Nucleus Mask          | Cytosol Mask          |
+-----------------------+-----------------------+-----------------------+
| |pic1|                | |pic2|                | |pic3|                |
+-----------------------+-----------------------+-----------------------+

To ensure overall flexibility, SPARCSpy seperates the segmentation code framework (i.e. loading input data, calling a segmentation method or saving results) from the actual segmentation algorithm (i.e. how the segmentation mask is calculated for a given input).

The segmentation code framework is implemented through so called segmentation classes. Each class is optimized for a given input data format and level of parallelization. The segmentation algorithms themselves are implemented by so called segmentation workflows. Each workflow implements a different segmentation algorithm (e.g. thresholding based segmentation or deep learning based segmentation). 

Using class inheritance each segmentation workflow inherits from a segmentation class to provide the segmentation code framework, but updates the segmentation generation method with the desired algorithm. This way you can easily exchange one segmentation algorithm for another while retaining the rest of the code framework.

Segmentation classes
--------------------

1. Segmentation
+++++++++++++++

The :func:`Segmentation <sparcscore.pipeline.segmentation.Segmentation>` class is optimized for processing input images of the format CXY within the context of a base SPARCSpy :func:`Project <sparcscore.pipeline.project.Project>`. It loads the input image into memory and then segments the image using the provided segmentation workflow. The resulting segmentation mask is then saved to disk.

2. ShardedSegmentation
++++++++++++++++++++++

The :func:`ShardedSegmentation <sparcscore.pipeline.segmentation.ShardedSegmentation>` class is an extension of the :func:`Segmentation <sparcscore.pipeline.segmentation.Segmentation>` class which is optimized for processing large input images in the format CXY in a parallelized fashion. When loading the input image, the :func:`ShardedSegmentation <sparcscore.pipeline.segmentation.ShardedSegmentation>` class splits the provided image into smaller tiles, called shards, which can then be processed individually in a parallelized fashion. After segmentation of the individual shards is completed, the :func:`ShardedSegmentation <sparcscore.pipeline.segmentation.ShardedSegmentation>` class merges the individual tiles back together to generate a final segmentation mask which extends over the complete input image.

Using a shardings approach has two main advantages:

    1. the possibility to segment images larger than the available memory the segmentation of images
    2. the parallelized segmentation of shards over mutiple threads to better utilize the available hardware

To determine how many shards should be generated, the user specifies the maximum number of pixels that can be allocated to one shard via the configuration file (``shard_size``). SPARCSpy then dynamically calculates a so-called `sharding plan` which splits the input image into the minimum number of equally sized shards. If desired, the user can also specify a pixel overlap (``overlap_px``) which determines how far the shards should overlap. This can be useful to ensure that cells which are located on the border between two shards are still fully segmented. 

The :func:`ShardedSegmentation <sparcscore.pipeline.segmentation.ShardedSegmentation>` class then segments each of the calculated shards individually using the designated number of parallel processes (``threads``). The intermediate segmentation results from each shard are saved to disk  before proceeding with the next shard. This ensures that memory usage during the segmentation process is kept to a minimum as only the required data to calculate the current shard segmentation are retained in memory.

After segmentation of each individual shard is completed, the :func:`ShardedSegmentation <sparcscore.pipeline.segmentation.ShardedSegmentation>` class merges the individual segmentation masks back together to generate a final segmentation mask which extends over the complete input image. During this process the ``cell ids`` are adjusted on each shard so that they remain unique throughout the final segmentation mask. After this process is completed the final segmentation mask is saved to disk and all intermediate results are deleted. 

Configuration parameters
^^^^^^^^^^^^^^^^^^^^^^^^

The following parameters for a sharded segmentation need to be specified in the configuration file:

.. code:: yaml

    ShardedSegmentationWorkflow:
        shard_size: 1000000000 # maximum number of pixels that can be allocated to one shard
        overlap_px: 0 # number of pixels by which the shards should overlap
        threads: 2 # number of threads to be used for parallelized segmentation of shards
        ... additional workflow specific parameters...
    

3. TimecourseSegmentation
+++++++++++++++++++++++++

The :func:`TimecourseSegmentation <sparcscore.pipeline.segmentation.TimecourseSegmentation>` class is optimized for processing input images of the format NCXY within the context of a SPARCSpy :func:`Timecourse Project <sparcscore.pipeline.project.TimecourseProject>`. It loads the input images into memory and segments them sequentially using the provided segmentation workflow. The resulting segmentation masks are then saved to disk.

4. MultithreadedSegmentation
++++++++++++++++++++++++++++

The :func:`MultithreadedSegmentation <sparcscore.pipeline.segmentation.MultithreadedSegmentation>` class is an extension of the :func:`TimecourseSegmentation <sparcscore.pipeline.segmentation.TimecourseSegmentation>` class and segments input images in the format NCYX in a parallelized fashion. The parallelization is achieved by splitting the input images along the N axis and processing each imagestack individually. The number of parallel processes can be specified by the user via the configuration file (``threads``).

Configuration parameters
^^^^^^^^^^^^^^^^^^^^^^^^

The following parameters for a multithreaded segmentation need to be specified in the configuration file:

.. code:: yaml

    MultithreadedSegmentationWorkflow:
        threads: 2 # number of threads to be used for parallelized segmentation of shards
        ... additional workflow specific parameters...
    
Segmentation Workflows
----------------------
Within SPARCSpy a segmentation workflow refers to a specific segmentation algorithm that can be called by one of the segmentation classes described above. Currently the following segmentation workflows are available for each of the different segmentation classes. They are explained in more detail below:

- :ref:`WGA_segmentation`
- :ref:`DAPI_segmentation`
- :ref:`Cytosol_segmentation_cellpose`
- :ref:`DAPI_segmentation_cellpose`

If none of these segmentation approaches suit your particular needs you can easily implement your own workflow. In case you need help, please open a git issue.

.. _WGA_segmentation:

WGA segmentation
++++++++++++++++

This segmentation workflow aims to segment mononucleated cells, i.e. cells that contain exactly one nucleus. Based on a nuclear stain and a cellmembrane stain, it first uses a thresholding approach to identify nuclei which are assumed to be the center of each cell. Then in a second step, the center of the identified nuclei are used as a starting point to generate a potential map using the cytosolic stain. This potential map is then used to segment the cytosol using a watershed approach. At the end of the workflow the user obtains both a nuclear and a cytosolic segmentation mask where each cytosol is matched to exactly one nucleus as kann be identified by the matching ``cell id``. 

This segmentation workflow is implemented to only run on the CPU. As such it can easily be scaled up to run on large datasets using parallel processing over multiple cores using either the :func:`ShardedSegmentation <sparcscore.pipeline.segmentation.ShardedSegmentation>` class or the :func:`MultithreadedSegmentation <sparcscore.pipeline.segmentation.MultithreadedSegmentation>` class respectively. However, it has a lot of parameters that need to be adjusted for different datasets to obtain an optimal segmentation.

..  code-block:: yaml
    :caption: Example configuration for  WGASegmentation

    WGASegmentation:
        input_channels: 3
        chunk_size: 50 # chunk size for chunked HDF5 storage. is needed for correct caching and high performance reading. should be left at 50.
        lower_quantile_normalization:   0.001
        upper_quantile_normalization:   0.999
        median_filter_size:   4 # Size in pixels
        nucleus_segmentation:
            lower_quantile_normalization:   0.01 # quantile normalization of dapi channel before local tresholding. Strong normalization (0.05,0.95) can help with nuclear speckles.
            upper_quantile_normalization:   0.99 # quantile normalization of dapi channel before local tresholding. Strong normalization (0.05,0.95) can help with nuclear speckles.
            median_block: 41 # Size of pixel disk used for median, should be uneven
            median_step: 4
            threshold: 0.2 # threshold above which nucleus is detected, if not specified a global threshold is calcualted using otsu
            min_distance: 8 # minimum distance between two nucleis in pixel
            peak_footprint: 7 # 
            speckle_kernel: 9 # Erosion followed by Dilation to remove speckels, size in pixels, should be uneven
            dilation: 0 # final dilation of pixel mask       
            min_size: 200 # minimum nucleus area in pixel
            max_size: 5000 # maximum nucleus area in pixel
            contact_filter: 0.5 # minimum nucleus contact with background
        wga_segmentation:
            threshold: 0.05 # threshold above which cytosol is detected, if not specified a global threshold is calcualted using otsu
            lower_quantile_normalization: 0.01
            upper_quantile_normalization: 0.99
            erosion: 2 # erosion and dilation are used for speckle removal and shrinking / dilation
            dilation: 7 # for no change in size choose erosion = dilation, for larger cells increase the mask erosion
            min_clip: 0
            max_clip: 0.2
            min_size: 200
            max_size: 30000
        chunk_size: 50

Nucleus Segmentation Algorithm
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. image:: ../images/WGA_segmentation_nucleus.pdf
   :width: 100%
   :align: left
   :alt: Nuclear segmentation algorithm steps


Cytosol Segmentation Algorithm
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. image:: ../images/WGA_segmentation_cytosol.pdf
   :width: 100%
   :align: left
   :alt: Cytosol segmentation algorithm steps


.. _DAPI_segmentation:

DAPI segmentation
+++++++++++++++++

This segmentation workflow aims to only segment nuclei. Based on a nuclear stain, it uses the same thresholding approach used during the WGA segmentation to identify nuclei. To ensure compatability with the downstream extraction workflow which assumes the presence of both a nuclear and a cytosolic segmentation mask the nuclear mask is duplicated and also used as the cytosolic mask. The generated single cell datasets using this segmentation method only focus on signals contained within the nuclear region.

..  code-block:: yaml
    :caption: Example configuration for  WGASegmentation

    DAPISegmentation:
        input_channels: 3
        chunk_size: 50 # chunk size for chunked HDF5 storage. is needed for correct caching and high performance reading. should be left at 50.
        lower_quantile_normalization:   0.001
        upper_quantile_normalization:   0.999
        median_filter_size:   4 # Size in pixels
        nucleus_segmentation:
            lower_quantile_normalization:   0.01 # quantile normalization of dapi channel before local tresholding. Strong normalization (0.05,0.95) can help with nuclear speckles.
            upper_quantile_normalization:   0.99 # quantile normalization of dapi channel before local tresholding. Strong normalization (0.05,0.95) can help with nuclear speckles.
            median_block: 41 # Size of pixel disk used for median, should be uneven
            median_step: 4
            threshold: 0.2 # threshold above which nucleus is detected, if not specified a global threshold is calcualted using otsu
            min_distance: 8 # minimum distance between two nucleis in pixel
            peak_footprint: 7 # 
            speckle_kernel: 9 # Erosion followed by Dilation to remove speckels, size in pixels, should be uneven
            dilation: 0 # final dilation of pixel mask       
            min_size: 200 # minimum nucleus area in pixel
            max_size: 5000 # maximum nucleus area in pixel
            contact_filter: 0.5 # minimum nucleus contact with background
        chunk_size: 50

Nucleus Segmentation Algorithm
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. image:: ../images/WGA_segmentation_nucleus.pdf
   :width: 100%
   :align: center
   :alt: Nuclear segmentation algorithm steps

.. _Cytosol_segmentation_cellpose:

Cytosol Cellpose segmentation
+++++++++++++++++++++++++++++

This segmentation workflow is built around the cellular segmentation algorithm `cellpose <https://cellpose.readthedocs.io/en/latest/>`_ . Cellpose is a deep neural network with a U-net style architecture that was trained on large datasets of microscopy images of cells. It provides very accurate out of the box segmentation models for both nuclei and cytosols but also allows you to fine-tune models using your own data. 

The SPARCSpy implementation of the cellpose segmenation algorithm allows you to perform both a nuclear and cytosolic segmentation and align the ``cellids`` between the two resulting masks. This means that the nucleus and the cytosol belonging to the same cell have the same ``cellids``. Furthermore, it performs some filtering steps to remove the masks from multi-nucleated cells or those with only a nuclear or cytosolic mask. This ensures that only cells which show a normal physiology are retained for further analysis.

While this segmentation workflow is also capable of running on a CPU it is highly recommended to utilize a GPU for better performance. 

If you utilize this segmentation workflow please also consider citing the `cellpose paper <https://www.nature.com/articles/s41592-022-01663-4#Sec8>`_. 

..  code-block:: yaml
    :caption: Example configuration for  Sharded Cytosol Cellpose Segmentation

    ShardedCytosolSegmentationCellpose:
        #segmentation class specific
        input_channels: 2
        output_masks: 2
        shard_size: 120000000 # maxmimum number of pixel per tile
        overlap_px: 100
        chunk_size: 50 # chunk size for chunked HDF5 storage. is needed for correct caching and high performance reading. should be left at 50.
        threads: 1 # number of shards / tiles segmented at the same size. should be adapted to the maximum amount allowed by memory.
        cache: "/fs/pool/pool-mann-maedler-shared/temp"
        #segmentation workflow specific
        nGPUs: 2
        lower_quantile_normalization:   0.001
        upper_quantile_normalization:   0.999
        median_filter_size: 6 # Size in pixels
        nucleus_segmentation:
            model: "nuclei"
        cytosol_segmentation:
            model: "cyto2"
        chunk_size: 50
        filtering_threshold: 0.95

.. _DAPI_segmentation_cellpose:

DAPI Cellpose segmentation
++++++++++++++++++++++++++

This segmentation workflow is also built around the cellular segmentation algorithm `cellpose <https://cellpose.readthedocs.io/en/latest/>`_  but only performs a nuclear segmentation. To ensure compatability with the downstream extraction workflow which assumes the presence of both a nuclear and a cytosolic segmentation mask the nuclear mask is duplicated and also used as the cytosolic mask. The generated single cell datasets using this segmentation method only focus on signals contained within the nuclear region.

As for the :ref:`cytosol segmentation cellpose <Cytosol_segmentation_cellpose>` workflow it is highly recommended to utilize a GPU.

If you utilize this segmentation workflow please also consider citing the `cellpose paper <https://www.nature.com/articles/s41592-022-01663-4#Sec8>`_. 

..  code-block:: yaml
    :caption: Example configuration for  DAPI Cellpose segmentation

    ShardedDAPISegmentationCellpose:
        #segmentation class specific
        input_channels: 2
        output_masks: 2
        shard_size: 120000000 # maxmimum number of pixel per tile
        overlap_px: 100
        chunk_size: 50 # chunk size for chunked HDF5 storage. is needed for correct caching and high performance reading. should be left at 50.
        cache: "/fs/pool/pool-mann-maedler-shared/temp"
        # segmentation workflow specific
        nGPUs: 2
        lower_quantile_normalization:   0.001
        upper_quantile_normalization:   0.999
        median_filter_size: 6 # Size in pixels
        nucleus_segmentation:
            model: "nuclei"
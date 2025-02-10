.. _single_cell_image_datasets:

Single-cell image datasets
--------------------------

For each cell, the extracted single-cell image dataset consists of a collection of images containing the segmentation masks and imaging channels of that particular cell. The segmentation masks are saved as binary masks, while the imaging channels are saved as float images normalized to the range ``[0, 1]``. During extraction the nuclear channel is masked using the nucleus mask, while all other imaging channels are masked using the cytosol mask. During this procedure the input mask is expanded slighly and a gaussian blur is applied to ensure that the entire cell is captured. Aggregated across all cells in a scPortrait dataset, the image collections for each cell are saved to ``HDF5``, a container file format that enables the retrieval of individual cells without loading the entire dataset. These ``HDF5`` datasets are the result of the extraction step and we refer to them as single-cell image datasets.

.. image:: ../images/single_cell_dataset.png
   :width: 100%

Besides containing the images themselves, the single-cell image datasets also contain annotation information for each cell within the dataset. In the minimal form this consists of a ``cellID``, which is a unique numerical identifier assigned to each cell during segmentation. By directly linking single-cell images to the ``cellID`` of the extracted cell this allows you to trace individual extracted cells back to their original position in the input image to e.g. select them for subsequent laser microdissection or look at their localization. Depending on the extraction method used, the single-cell image dataset can also contain additional labelling information.

.. image:: ../images/HDF5_data_containers.png
    :width: 100%
    :align: center

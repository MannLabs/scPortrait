Classification
==============

After extracting individual cells into a collection of images, each depicting a single cell isolated from its neighbours, the phenotypes of all cells in this dataset can be analyzed. scPortrait is compatible with various phenotyping techniques ranging from classical image analysis methods, for example those provided by `scikit-image <https://scikit-image.org/>`_ to recent deep learning-based computer vision models. scPortrait provides a pytorch dataloader for its HDF5-based dataset format, enabling inference with existing pytorch models and facilitating training new or finetuning existing models with your own data.

You can find more information on running an inference within a scPortrait Project in this `notebook <https://mannlabs.github.io/scPortrait/html/pages/notebooks/example_scPortrait_project.html#Classification-of-extracted-single-cells>`_.

Classification
==============

After extracting individual cells into a collection of images, each depicting a single cell isolated from its neighbours, the phenotypes of all cells in this dataset can be analyzed. SPARCSpy is compatible with various phenotyping techniques ranging from classical image analysis methods, for example those provided by `scikit-image <https://scikit-image.org/>`_ to recent deep learning-based computer vision models. SPARCSpy provides a pytorch dataloader for its HDF5-based dataset format, enabling inference with existing pytorch models and facilitating training new or finetuning existing models with your own data. 

You can find more information on running an inference within a SPARCSpy Project in this `notebook <https://mannlabs.github.io/SPARCSpy/html/pages/notebooks/example_sparcspy_project.html#Classification-of-extracted-single-cells>`_.
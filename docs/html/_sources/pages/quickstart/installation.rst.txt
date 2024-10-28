.. _installation:

************
Installation
************

We recommended installing the library within a separate conda environment running Python 3.10 or 3.11. Python 3.12 is currently **not** supported.

.. code::

   conda create -n "scPortrait"
   conda activate scPortrait

For utilizing the stitching capabilities of scPortrait, a working java installation is required. If not already installed, you can download the latest version of Java from the `official website <https://www.java.com/en/download/>`_ or install it via mamba or conda:

.. Important::
   Java needs to be installed before installing scPortrait. Otherwise when trying to access the stitching capabilities of scPortrait, an error will be raised that Java is not found at the indicated path.

.. code::

   conda install -c conda-forge openjdk

If you wish to utilize the accelerated stitching backend you need to install the `graph-tool library <https://graph-tool.skewed.de>`_. This library is not available via pip and needs to be installed seperately via conda.

.. code::

   conda install -c conda-forge graph-tool==2.68

To install the latest release of scPortrait:

.. code::

   pip install git+https://github.com/MannLabs/scPortrait

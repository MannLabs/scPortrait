.. _installation:

************
Installation
************

We recommend installing the library within a separate conda environment running Python 3.11, 3.12, 3.13 or 3.14.

.. code::

   conda create -n "scPortrait" python=3.11
   conda activate scPortrait


.. dropdown:: Optional: Installing the stitching capabilities of scPortrait
   :chevron: down-up

   The stitching capabilities of scPortrait require a working Java installation. If not already installed, you can download the latest version of Java from the `official website <https://www.java.com/en/download/>`_ or install it via mamba or conda:

   .. Important::

      Java needs to be installed before installing scPortrait. Otherwise when trying to access the stitching capabilities of scPortrait, an error will be raised that Java is not found at the indicated path.

   .. code::

      conda install -c conda-forge openjdk

   If you wish to utilize the accelerated stitching backend you need to install the `graph-tool library <https://graph-tool.skewed.de>`_. This library is not available via pip and needs to be installed separately via conda.

   .. code::

      conda install -c conda-forge graph-tool==2.68

   The stitching capabilities require optional dependencies which you can install along with scPortrait by running:

   .. code::

      pip install "scportrait[stitching]"

To install the latest release of scPortrait:

.. code::

   pip install scportrait

Optional feature extras are available for dependencies that are only needed by specific workflows:

.. code::

   pip install "scportrait[segmentation]"  # scikit-fmm-backed segmentation helpers and workflows
   pip install "scportrait[zstack]"        # extended depth-of-field z-stack compression

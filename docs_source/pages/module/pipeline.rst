*******************
pipeline 
*******************

.. toctree::
   :maxdepth: 3

project
#######

Within scPortrait, all operations are centered around the concept of a ``Project``. A ``Project`` is a python class which manages all of the scPortrait processing steps and is the central element through which all operations are performed. Each ``Project`` directly maps to a directory on the file system which contains all of the inputs to a specific scPortrait run as well as the generated outputs. Depending on the structure of the data that is to be processed a different Project class is required. Please see :ref:`here <projects>` for more information.

Project 
=========
.. autoclass:: scportrait.pipeline.project.Project
    :members:
    :show-inheritance:

segmentation
#############

Segmentation
==============
.. autoclass:: scportrait.pipeline.segmentation.Segmentation
    :members:
    :show-inheritance:

ShardedSegmentation
=====================
.. autoclass:: scportrait.pipeline.segmentation.ShardedSegmentation
    :members:
    :show-inheritance:

segmentation workflows
######################
.. automodule:: scportrait.pipeline.segmentation.workflows
    :members:

extraction
###########

HDF5CellExtraction
===================
.. autoclass:: scportrait.pipeline.extraction.HDF5CellExtraction
    :members:
    :show-inheritance:


classification
##############

MLClusterClassifier
===================
.. autoclass:: scportrait.pipeline.classification.MLClusterClassifier
    :members:

    .. automethod:: __call__

CellFeaturizer
==============
.. autoclass:: scportrait.pipeline.classification.CellFeaturizer
    :members:

    .. automethod:: __call__

selection
###########

LMDSelection
==============
.. autoclass:: scportrait.pipeline.selection.LMDSelection
    :members:
    :show-inheritance:


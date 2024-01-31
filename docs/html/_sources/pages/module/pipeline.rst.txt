*******************
pipeline 
*******************

.. toctree::
   :maxdepth: 3

base
#######

A collection of base classes from which other classes in the SPARCSpy environment can inherit that manage base functionalities like logging or directory creation.

Logable
==============

Base Class which generates framework for logging. 

.. autoclass:: sparcscore.pipeline.base.Logable
    :members:

ProcessingStep
==============

Starting point for all processing steps. Reads a config file that contains the parameters used to set up a processing method and generates the folder structure necessary for saving the generated outputs. 

.. autoclass:: sparcscore.pipeline.base.ProcessingStep
    :members:
    :show-inheritance:

project
#######

Within SPARCSpy, all operations are centered around the concept of a ``Project``. A ``Project`` is a python class which manages all of the SPARCSpy processing steps and is the central element through which all operations are performed. Each ``Project`` directly maps to a directory on the file system which contains all of the inputs to a specific SPARCSpy run as well as the generated outputs. Depending on the structure of the data that is to be processed a different Project class is required. Please see :ref:`here <projects>` for more information.

Project 
=========
.. autoclass:: sparcscore.pipeline.project.Project
    :members:
    :show-inheritance:


TimecourseProject 
=================
.. autoclass:: sparcscore.pipeline.project.TimecourseProject
    :members:
    :show-inheritance:


segmentation
#############

Segmentation
==============
.. autoclass:: sparcscore.pipeline.segmentation.Segmentation
    :members:
    :show-inheritance:

ShardedSegmentation
=====================
.. autoclass:: sparcscore.pipeline.segmentation.ShardedSegmentation
    :members:
    :show-inheritance:

TimecourseSegmentation
======================
.. autoclass:: sparcscore.pipeline.segmentation.TimecourseSegmentation
    :members:
    :show-inheritance:

MultithreadedTimecourseSegmentation
===================================
.. autoclass:: sparcscore.pipeline.segmentation.MultithreadedSegmentation
    :members:
    :show-inheritance:


workflows
##########
.. automodule:: sparcscore.pipeline.workflows
    :members:

extraction
###########

HDF5CellExtraction
===================
.. autoclass:: sparcscore.pipeline.extraction.HDF5CellExtraction
    :members:
    :show-inheritance:

TimecourseHDF5CellExtraction
============================
.. autoclass:: sparcscore.pipeline.extraction.TimecourseHDF5CellExtraction
    :members:
    :show-inheritance:


classification
##############

MLClusterClassifier
===================
.. autoclass:: sparcscore.pipeline.classification.MLClusterClassifier
    :members:

    .. automethod:: __call__

CellFeaturizer
==============
.. autoclass:: sparcscore.pipeline.classification.CellFeaturizer
    :members:

    .. automethod:: __call__

selection
###########

LMDSelection
==============
.. autoclass:: sparcscore.pipeline.selection.LMDSelection
    :members:
    :show-inheritance:


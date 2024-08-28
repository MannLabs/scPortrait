*******************
pipeline 
*******************

.. toctree::
   :maxdepth: 3

base
#######

A collection of base classes from which other classes in the scPortrait environment can inherit that manage base functionalities like logging or directory creation.

Logable
==============

Base Class which generates framework for logging. 

.. autoclass:: scportrait.pipeline.base.Logable
    :members:

ProcessingStep
==============

Starting point for all processing steps. Reads a config file that contains the parameters used to set up a processing method and generates the folder structure necessary for saving the generated outputs. 

.. autoclass:: scportrait.pipeline.base.ProcessingStep
    :members:
    :show-inheritance:

project
#######

Within scPortrait, all operations are centered around the concept of a ``Project``. A ``Project`` is a python class which manages all of the scPortrait processing steps and is the central element through which all operations are performed. Each ``Project`` directly maps to a directory on the file system which contains all of the inputs to a specific scPortrait run as well as the generated outputs. Depending on the structure of the data that is to be processed a different Project class is required. Please see :ref:`here <projects>` for more information.

Project 
=========
.. autoclass:: scportrait.pipeline.project.Project
    :members:
    :show-inheritance:


TimecourseProject 
=================
.. autoclass:: scportrait.pipeline.project.TimecourseProject
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

TimecourseSegmentation
======================
.. autoclass:: scportrait.pipeline.segmentation.TimecourseSegmentation
    :members:
    :show-inheritance:

MultithreadedTimecourseSegmentation
===================================
.. autoclass:: scportrait.pipeline.segmentation.MultithreadedSegmentation
    :members:
    :show-inheritance:


workflows
##########
.. automodule:: scportrait.pipeline.workflows
    :members:

extraction
###########

HDF5CellExtraction
===================
.. autoclass:: scportrait.pipeline.extraction.HDF5CellExtraction
    :members:
    :show-inheritance:

TimecourseHDF5CellExtraction
============================
.. autoclass:: scportrait.pipeline.extraction.TimecourseHDF5CellExtraction
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


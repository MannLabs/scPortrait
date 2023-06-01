*******************
pipeline 
*******************

.. toctree::
   :maxdepth: 3

base
#######

Logable
==============
.. autoclass:: sparcscore.pipeline.base.Logable
    :members:

ProcessingStep
==============
.. autoclass:: sparcscore.pipeline.base.ProcessingStep
    :members:
    :show-inheritance:

project
#######

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
    :show-inheritance:

    .. automethod:: __call__

selection
###########

LMDSelection
==============
.. autoclass:: sparcscore.pipeline.selection.LMDSelection
    :members:
    :show-inheritance:


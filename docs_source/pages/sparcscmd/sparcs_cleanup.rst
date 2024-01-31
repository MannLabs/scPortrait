.. _sparcs-cleanup:

sparcs-cleanup
====================
.. argparse::
   :module: sparcscmd.sparcs-stat
   :func: generate_parser
   :prog: sparcs-cleanup
   
List intermediate files contained in sparcs projects that can be deleted to free up disk-space.
Can be run in dry-run to only list found files before deleteting.

Examples
--------

    Show found files in a folder containing multiple datasets
    ::
        sparcs-cleanup -n True .
        
    Result:
    ::
        Searching for intermediate files that can be deleted, this may take a moment...
        
        ProjectA
        Found the following files to delete:
        ('~/ProjectA/segmentation/input_image.h5', '42.6GiB')

        Found the following directories to delete:
        ('~/ProjectA/segmentation/tiles/7', '7.7GiB')
        ('~/ProjectA/segmentation/tiles/6', '7.7GiB')
        ('~/ProjectA/segmentation/tiles/3', '7.7GiB')
        ('~/ProjectA/segmentation/tiles/11', '7.7GiB')
        ('~/ProjectA/segmentation/tiles/19', '7.7GiB')
        ('~/ProjectA/segmentation/tiles/14', '7.7GiB')
        ('~/ProjectA/segmentation/tiles/17', '7.7GiB')
        ('~/ProjectA/segmentation/tiles/5', '7.7GiB')
        ('~/ProjectA/segmentation/tiles/2', '7.7GiB')
        ('~/ProjectA/segmentation/tiles/4', '7.7GiB')
        ('~/ProjectA/segmentation/tiles/1', '7.7GiB')
        ('~/ProjectA/segmentation/tiles/13', '7.7GiB')
        ('~/ProjectA/segmentation/tiles/0', '7.7GiB')
        ('~/ProjectA/segmentation/tiles/16', '7.7GiB')
        ('~/ProjectA/segmentation/tiles/20', '7.7GiB')
        ('~/ProjectA/segmentation/tiles/9', '7.7GiB')
        ('~/ProjectA/segmentation/tiles/12', '7.7GiB')
        ('~/ProjectA/segmentation/tiles/15', '7.7GiB')
        ('~/ProjectA/segmentation/tiles/10', '7.7GiB')
        ('~/ProjectA/segmentation/tiles/8', '7.7GiB')
        ('~/ProjectA/segmentation/tiles/18', '7.7GiB')
        Rerun with -n False to remove these files
        
        ProjectB
        Found the following files to delete:
        ('~/ProjectA/segmentation/input_image.h5', '42.6GiB')

        Found the following directories to delete:
        ('~/ProjectA/segmentation/tiles/7', '7.7GiB')
        ('~/ProjectA/segmentation/tiles/6', '7.7GiB')
        ('~/ProjectA/segmentation/tiles/3', '7.7GiB')
        ('~/ProjectA/segmentation/tiles/11', '7.7GiB')
        ('~/ProjectA/segmentation/tiles/19', '7.7GiB')
        ('~/ProjectA/segmentation/tiles/14', '7.7GiB')
        ('~/ProjectA/segmentation/tiles/17', '7.7GiB')
        ('~/ProjectA/segmentation/tiles/5', '7.7GiB')
        ('~/ProjectA/segmentation/tiles/2', '7.7GiB')
        ('~/ProjectA/segmentation/tiles/4', '7.7GiB')
        ('~/ProjectA/segmentation/tiles/1', '7.7GiB')
        ('~/ProjectA/segmentation/tiles/13', '7.7GiB')
        ('~/ProjectA/segmentation/tiles/0', '7.7GiB')
        ('~/ProjectA/segmentation/tiles/16', '7.7GiB')
        ('~/ProjectA/segmentation/tiles/20', '7.7GiB')
        ('~/ProjectA/segmentation/tiles/9', '7.7GiB')
        ('~/ProjectA/segmentation/tiles/12', '7.7GiB')
        ('~/ProjectA/segmentation/tiles/15', '7.7GiB')
        ('~/ProjectA/segmentation/tiles/10', '7.7GiB')
        ('~/ProjectA/segmentation/tiles/8', '7.7GiB')
        ('~/ProjectA/segmentation/tiles/18', '7.7GiB')
        Rerun with -n False to remove these files
    
    Delete intermediate files in folder containing multiple datasets
    ::
        sparcs-cleanup -n False .
        
    Result:
    ::
        Searching for intermediate files that can be deleted, this may take a moment...

        ProjectA
        Deleting files...
        Deleted files with a total storage size of 200.6GiB

        ProjectB
        Deleting files...
        Deleted files with a total storage size of 200.6GiB
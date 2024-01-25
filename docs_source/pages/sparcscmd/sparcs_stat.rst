.. _sparcs-stat:

sparcs-stat
====================
.. argparse::
   :module: sparcscmd.sparcs-stat
   :func: generate_parser
   :prog: sparcs-stat
   
get information on the status of sparcs projects.

Examples
--------

    Show progress in a folder containing multiple datasets
    ::
        sparcs-stat
        
    Result:
    ::
        sparcs-stat collecting information. This can take some time...
       slide000           True        731,468        72.6GiB
               single_cells.h5        729,775        30.5GiB
       slide001           True        755,358        69.3GiB
               single_cells.h5        753,277        30.4GiB


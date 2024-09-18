.. _sparcs-split:

sparcs-split
====================
.. argparse::
   :module: sparcscmd.sparcs-split
   :func: _generate_parser
   :prog: sparcs-split


Manipulate existing scPortrait single cell hdf5 datasets.
sparcs-split can be used for splitting, shuffleing and compression / decompression.

Examples
--------

    Splitting with shuffle and compression:
    ::
        sparcs-split single_cells.h5 -r -c -o train.h5 0.9 -o test.h5 0.05 -o validate.h5 0.05

    Shuffle
    ::
        sparcs-split single_cells.h5 -r -o single_cells.h5 1.0

    Compression
    ::
        sparcs-split single_cells.h5 -c -o single_cells.h5 1.0

    Decompression
    ::
        sparcs-split single_cells.h5 -o single_cells.h5 1.0

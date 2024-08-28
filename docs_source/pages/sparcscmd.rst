Command Line Tools
=====================

scPortrait comes equipped with several command line tools for easy handling of scPortrait datasets. 

1. :ref:`sparcs-split <sparcs-split>`: Split, shuffle and compress scPortrait datasets.
2. :ref:`sparcs-stat <sparcs-stat>`: Get information on the status of scPortrait datasets.
3. :ref:`sparcs-cleanup <sparcs-cleanup>`: List intermediate files contained in scPortrait datasets that can be deleted to free up disk-space.

To effectively utilize these tools it is recommended to add the following aliases to your ``.bashrc`` file.

.. code:: bash

    alias sparcs-split="python /path/to/repo/clone/scPortrait/src/sparcscmd/sparcs-cleanup.py"
    alias sparcs-cleanup="python /path/to/repo/clone/scPortrait/src/sparcscmd/sparcs-cleanup.py"
    alias sparcs-stats="python /path/to/repo/clone/scPortrait/src/sparcscmd/sparcs-stat.py"


.. toctree::
   :maxdepth: 2
   :hidden:

   sparcscmd/sparcs_split
   sparcscmd/sparcs_stat
   sparcscmd/sparcs_cleanup
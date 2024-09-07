.. _installation:

*******************
Installation
*******************

scPortrait has been tested with **Python 3.8, 3.9 and 3.10**. We recommended installing the library within a separate conda environment. 

We recommend installing the dependencies with mamba or conda before installing scPortrait.

.. note:: **Using Mamba instead of Conda**
   
   We have encountered issues with resolving dependencies when trying to install prerequisites using conda. In our experience, using mamba instead of conda leads to reliable results. To install mamba check out the documentation `here <https://mamba.readthedocs.io/en/latest/index.html>`_. After installation, you can use mamba instead of conda simply by replacing the command ``conda`` with ``mamba`` in the following instructions.

.. code::

   git clone https://github.com/MannLabs/SPARCSspatial
   cd scPortrait

   conda create -n "scPortrait"
   conda activate scPortrait
   conda install python=3.9 scipy 'scikit-image>=0.19' scikit-fmm cellpose opencv numba -c conda-forge

In case you wish to utilize the ML capabilities of scPortrait (either for segmentation or classification) please follow the instructions `here <https://pytorch.org/get-started/locally/>`__ to install pytorch correctly for your operating system. After you can verify your installation by executing the following python code:

.. code:: python

   import torch
   x = torch.rand(5, 3)
   print(x)

You can access the Python console by typing ``python`` and exit it when you are finished by entering ``exit()``.
The output should look similar to this:

.. code:: python

   tensor([[0.3380, 0.3845, 0.3217],
         [0.8337, 0.9050, 0.2650],
         [0.2979, 0.7141, 0.9069],
         [0.1449, 0.1132, 0.1375],
         [0.4675, 0.3947, 0.1426]])

Once you have installed `pytorch` according to the instructions you still need to install `pytorch lightning`. To do this run:

.. code::

      conda install -c pytorch pytorch-lightning

.. note:: **pytorch and pytorch lightning installation from pytorch channel**
   
   We have encountered issues with installed pytorch dependencies when they are not installed from the ``pytorch`` channel but from ``conda-forge``. We highly recommend only installing pytorch and its dependencies from the ``pytorch`` channel. You can check your installation by importing ``import pytorch_lightning`` in Python. If this import fails please double check what channel your packages are installed from and reinstall them from the ``pytorch`` channel if necessary.

Currently scPortrait depends on a developer version of alphabase so please install the package from source by doing the following:

.. code::

   pip install git+https://github.com/MannLabs/alphabase

scPortrait also depends on the `py-lmd <https://github.com/MannLabs/py-lmd>`_ library. Before proceeding please install the py-lmd libray into the same conda environment following the installation instructions `here <https://mannlabs.github.io/py-lmd/html/pages/quickstart.html#installation-from-github>`__.
Once these steps are completed you can proceed to install the scPortrait package.

Clone the `Github repository <https://github.com/MannLabs/SPARCSspatial>`_ and use pip to install the library in your current environment. Please make sure that the package is installed editable (with the `-e` flag). Otherwise pretrained models might not be available:

.. code:: 

   pip install -e .

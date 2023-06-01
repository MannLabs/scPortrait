.. _installation:

*******************
Installation
*******************

SPARCSpy has been tested with **Python 3.8 and 3.9**. We recommended installing the library within a conda environment. 

To install the SPARCSpy library clone the Github repository and use pip to install the library in your current environment.
Please make sure that the package is installed editable (with the `-e` flag). Otherwise pretrained models might not be available.

We recommend installing the non-python dependencies with conda before installing SPARCSpy (especially if running on an M1 Silicon Mac):

.. code::

   git clone https://github.com/MannLabs/SPARCSpy
   cd SPARCSpy

   conda create -n "SPARCSpy"
   conda activate SPARCSpy
   conda install python=3.9 scipy 'scikit-image>=0.19' scikit-fmm cellpose opencv numba -c conda-forge


In case you wish to utilize the ML capabilities of SPARCSpy (either for segmentation or classification) please follow the instructions `here <https://pytorch.org/get-started/locally/>`_ to install pytorch correctly for your operating system. Once this has been installed you can verify that pytorch is installed correctly by executing the following python code:

.. code:: python

   import torch
   x = torch.rand(5, 3)
   print(x)

You can access the python console by typing `python` and exit it when you are finished by entering `exit()`.
The output should look something like this:

.. code:: python

   tensor([[0.3380, 0.3845, 0.3217],
         [0.8337, 0.9050, 0.2650],
         [0.2979, 0.7141, 0.9069],
         [0.1449, 0.1132, 0.1375],
         [0.4675, 0.3947, 0.1426]])

Once you have installed pytorch according to the instructions we still need to install pytorch lightning. To do this run:

.. code:: 

   conda install -c conda-forge pytorch-lightning

Currently SPARCSpy depends on a developer version of alphabase so please install the package from source by doing the following:

.. code::

   pip install git+https://github.com/MannLabs/alphabase

Once these steps are completed you can proceed to install the SPARCSpy package via pip:

.. code:: 

   pip install -e .

In case you wish to export shapes for excision on a Leica LMD please also install the `py-lmd <https://github.com/MannLabs/py-lmd>`_ library into the same conda environment
following the installation instructions `here <https://mannlabs.github.io/py-lmd/html/pages/quickstart.html#installation-from-github>`_.
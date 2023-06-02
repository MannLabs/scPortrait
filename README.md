[![Python package](https://img.shields.io/badge/version-v1.0.0-blue)](https://github.com/MannLabs/SPARCSpy/actions/workflows/python-package.yml) [![Python package](https://img.shields.io/badge/license-MIT-blue)](https://github.com/MannLabs/SPARCSpy/actions/workflows/python-package.yml)
[![website](https://img.shields.io/website?url=https%3A%2F%2Fmannlabs.github.io/SPARCSpy/html/index.html)](https://mannlabs.github.io/SPARCSpy/html/index.html)

SPARCSpy is a scalable toolkit to analyse SPARCS datasets. The python implementation efficiently segments individual cells, generates single-cell datasets and provides tools for the efficient deep learning classification of their phenotypes and subsequent excision using Laser Microdissection.

## Installation from Github

SPARCSpy has been tested with **Python 3.8 and 3.9**.
To install the SPARCSpy library clone the Github repository and use pip to install the library in your current environment. It is recommended to use the library with a conda environment. Please make sure that the package is installed editable like described. Otherwise pretrained models might not be available.

We recommend installing the non-python dependencies with conda before installing SPARCSpy (especially if running on an M1 Silicon Mac):

```
git clone https://github.com/MannLabs/SPARCSpy
cd SPARCSpy

conda create -n "SPARCSpy"
conda activate SPARCSpy
conda install python=3.9 scipy 'scikit-image>=0.19' scikit-fmm cellpose opencv numba -c conda-forge
```

In case you wish to utilize the ML capabilities of SPARCSpy (either for segmentation or classification) please follow the instructions [here](<https://pytorch.org/get-started/locally/>) to install pytorch correctly for your operating system. Once completed you can verify that pytorch is installed correctly by executing the following python code (you can access the python console by typing `python` and exit it when you are finished by entering `exit()`):

```
import torch
x = torch.rand(5, 3)
print(x)
```

The output should look something like this:

```
tensor([[0.3380, 0.3845, 0.3217],
        [0.8337, 0.9050, 0.2650],
        [0.2979, 0.7141, 0.9069],
        [0.1449, 0.1132, 0.1375],
        [0.4675, 0.3947, 0.1426]])
```

Once you have installed pytorch according to the instructions we still need to install pytorch lightning. To do this run:

```
conda install -c conda-forge pytorch-lightning
```

Once these steps are completed you can proceed to install the SPARCSpy package via pip:

```
pip install -e .
```
  
## Documentation

The current documentation can be found under https://mannlabs.github.io/SPARCSpy/html/index.html.

## Citing our Work

py-lmd was developed by Georg Wallmann, Sophia Mädler and Niklas Schmacke in the labs of Veit Hornung and Matthias Mann. If you use our code please cite the [following manuscript](https://www.biorxiv.org/content/10.1101/2023.06.01.542416v1):

SPARCS, a platform for genome-scale CRISPR screening for spatial cellular phenotypes
Niklas Arndt Schmacke, Sophia Clara Maedler, Georg Wallmann, Andreas Metousis, Marleen Berouti, Hartmann Harz, Heinrich Leonhardt, Matthias Mann, Veit Hornung
bioRxiv 2023.06.01.542416; doi: https://doi.org/10.1101/2023.06.01.542416
# sparcspy

[![Tests][badge-tests]][link-tests]
[![Documentation][badge-docs]][link-docs]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/MannLabs/sparcspy/test.yaml?branch=main
[link-tests]: https://github.com/MannLabs/SPARCSspatial/actions/workflows/test.yml
[badge-docs]: https://img.shields.io/readthedocs/sparcspy

![Graphical Abstract](https://github.com/MannLabs/SPARCSspatial/assets/15019107/47461f35-3dec-4aa6-ba51-ee1b631ddab9)

SPARCSpy is a scalable toolkit to analyse SPARCS datasets. The python implementation efficiently segments individual cells, generates single-cell datasets and provides tools for the efficient deep learning classification of their phenotypes for downstream applications.

To better understand what SPARCSpy can do please checkout our [documentation](https://mannlabs.github.io/SPARCSpy/html/index.html), the [paper](https://www.biorxiv.org/content/10.1101/2023.06.01.542416v1) or our [tweetorial on the computation tools](https://twitter.com/SophiaMaedler/status/1665816840726085634?s=20) and our [tweetorial on the science](https://twitter.com/niklas_a_s/status/1664538053744947203?s=20).

## Getting started

Please refer to the [documentation][link-docs]. In particular, the

-   [API documentation][link-api].

## Installation

You need to have Python 3.10 or newer installed on your system. If you don't have
Python installed, we recommend installing [Mambaforge](https://github.com/conda-forge/miniforge#mambaforge).

There are several alternative options to install sparcspy:

<!--
1) Install the latest release of `sparcspy` from [PyPI][link-pypi]:

```bash
pip install sparcspy
```
-->

1. Install the latest development version:

<!-- TODO update this - most dependencies should be inside the toml and only weird stuff from conda-->

```bash
pip install git+https://github.com/MannLabs/SPARCSspatial.git@main
```

We recommend installing the following dependencies from conda:

```bash
conda install -c conda-forge cellpose scikit-fmm
```

## Release notes

See the [changelog][changelog].

## Contact

For questions and help requests, you can reach out in the [scverse discourse][scverse-discourse].
If you found a bug, please use the [issue tracker][issue-tracker].

## Citation

SPARCSpy was created by Georg Wallmann, Sophia Mädler and Niklas Schmacke in the labs of [Matthias Mann](https://www.biochem.mpg.de/de/mann) and [Veit Hornung](https://www.genzentrum.uni-muenchen.de/research-groups/hornung/index.html) in 2023.
SPARCSpy is actively developed with support from the labs of Matthias Mann, Veit Hornung and [Fabian Theis](https://www.helmholtz-munich.de/en/icb/research-groups/theis-lab).

If you use our code please cite [this manuscript](https://www.biorxiv.org/content/10.1101/2023.06.01.542416v1):

SPARCS, a platform for genome-scale CRISPR screening for spatial cellular phenotypes
Niklas Arndt Schmacke, Sophia Clara Maedler, Georg Wallmann, Andreas Metousis, Marleen Berouti, Hartmann Harz, Heinrich Leonhardt, Matthias Mann, Veit Hornung
bioRxiv 2023.06.01.542416; doi: https://doi.org/10.1101/2023.06.01.542416

[scverse-discourse]: https://discourse.scverse.org/
[issue-tracker]: https://github.com/MannLabs/sparcspy/issues
[changelog]: https://sparcspy.readthedocs.io/latest/changelog.html
[link-docs]: https://sparcspy.readthedocs.io
[link-api]: https://sparcspy.readthedocs.io/latest/api.html
[link-pypi]: https://pypi.org/project/sparcspy

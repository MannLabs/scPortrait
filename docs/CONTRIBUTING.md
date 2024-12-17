# Contributing guide

Scanpy provides extensive [developer documentation][scanpy developer guide], most of which applies to this repo, too.
This document will not reproduce the entire content from there. Instead, it aims at summarizing the most important
information to get you started on contributing.

We assume that you are already familiar with git and with making pull requests on GitHub. If not, please refer
to the [scanpy developer guide][].

## Installing dev dependencies

In addition to the packages needed to _use_ this package, you need additional python packages to _run tests_ and _build
the documentation_. It's easy to install them using `pip`:

```bash
git clone https://github.com/MannLabs/scPortrait
cd scPortrait
pip install -e '.[dev]'
```

## Code-style

This project uses [pre-commit][] to enforce consistent code-styles. On every commit, pre-commit checks will either
automatically fix issues with the code, or raise an error message.

To enable pre-commit locally, simply run

```bash
pre-commit install
```

in the root of the repository. Pre-commit will automatically download all dependencies when it is run for the first time.

Alternatively, you can rely on the [pre-commit.ci][] service enabled on GitHub. If you didn't run `pre-commit` before
pushing changes to GitHub it will automatically commit fixes to your pull request, or show an error message.

If pre-commit.ci added a commit on a branch you still have been working on locally, simply use

```bash
git pull --rebase
```

to integrate the changes into yours.

While the [pre-commit.ci][] is useful, we strongly encourage installing and running pre-commit locally first to understand its usage.

## Writing documentation

Please write documentation for new or changed features and use-cases. This project uses [sphinx][] with the following features:

-   Google-style docstrings, where the __init__ method should be documented in its own docstring, not at the class level.
-   example code
-   automatic building with Sphinx

Refer to [sphinx google style docstrings][] for detailed information on writing documentation.

### Tutorials with jupyter notebooks

The tutorials for this project are hosted in a dedicated repository: [scPortrait Notebooks](https://github.com/MannLabs/scPortrait-notebooks).

<!-- Links -->

[scanpy developer guide]: https://scanpy.readthedocs.io/en/latest/dev/index.html
[cookiecutter-scverse-instance]: https://cookiecutter-scverse-instance.readthedocs.io/en/latest/template_usage.html
[github quickstart guide]: https://docs.github.com/en/get-started/quickstart/create-a-repo?tool=webui
[pre-commit.ci]: https://pre-commit.ci/
[pre-commit]: https://pre-commit.com/
[sphinx]: https://www.sphinx-doc.org/en/master/
[sphinx autodoc typehints]: https://github.com/tox-dev/sphinx-autodoc-typehints
[sphinx google style docstrings]: https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html
[scPortrait Notebooks]: (https://github.com/MannLabs/scPortrait-notebooks)

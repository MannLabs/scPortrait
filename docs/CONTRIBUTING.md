# Contributing guide

Contributions to scPortrait are welcome! This section provides some guidelines and tips to follow when contributing.

## Installing dev dependencies

In addition to the packages needed to _use_ this package, you need additional python packages to _run tests_ and _build the documentation_. It's easy to install them using `pip`:

```bash
git clone https://github.com/MannLabs/scPortrait
cd scPortrait
pip install -e '.[dev]'
```

## Code-style

This project uses [pre-commit][] to enforce consistent code-styles. On every commit, pre-commit checks will either automatically fix issues with the code, or raise an error message.

To enable pre-commit locally, simply run

```console
pre-commit install
```

in the root of the repository. Pre-commit will automatically download all dependencies when it is run for the first time.

Alternatively, you can rely on the [pre-commit.ci][] service enabled on GitHub. If you didn't run `pre-commit` before pushing changes to GitHub it will automatically commit fixes to your pull request, or show an error message.

If pre-commit.ci added a commit on a branch you still have been working on locally, simply use

```console
git pull --rebase
```
to integrate the changes into yours.

While the [pre-commit.ci][] is useful, we strongly encourage installing and running pre-commit locally first to understand its usage.

## Writing documentation

Please write documentation for new or changed features and use-cases. This project uses [sphinx][] with the following features:

-   Google-style docstrings, where the __init__ method should be documented in its own docstring, not at the class level.
-   example code
-   automatic building with Sphinx
-   add type hints for use with mypy

Refer to [sphinx google style docstrings][] for detailed information on writing documentation.

## Writing Tests

If you are adding a new function to scPortrait please also include a unit test.

## Running Tests

We use [pytest][] to test scProtrait. To run the tests, simply run `pytest .` in your local clone of the scPortrait repo.

A lot of warnings can be thrown while running the test files. It’s often easier to read the test results with them hidden via the `--disable-pytest-warnings`  argument.

## Building the Documentation

The docs for scPortrait are updated and built automatically whenever code is merged or commited to the main branch. To test documentation locally you need to do the following:

1. navigate into the `docs` folder in your local scPortrait clone
2. ensure you have a functional development environment where the additional dev dependencies (these incldue those required to build the documentation) are installed
3. execute:

```console
make clean
make html
```
4. open the file `scportriat/docs/_build/html/index.html` in your favorite browser

## Tutorials with jupyter notebooks

Indepth tutorials using jupyter notebooks are hosted in a dedicated repository: [scPortrait Notebooks](https://github.com/MannLabs/scPortrait-notebooks).

Please update and/or add new tutorials there.

## Commiting Code Changes

We assume some familiarity with `git`. For more detailed information we recommend checking out these tutorials:

[Atlassian's git tutorial][]: Beginner friendly introductions to the git command line interface
[Setting up git for GitHub][]: Configuring git to work with your GitHub user account

### Forking and cloning

To get the code, and be able to push changes back to the main project, you'll need to (1) fork the repository on github and (2) clone the repository to your local machine.

This is very straight forward if you're using [GitHub's CLI][]:

```console
$ gh repo fork mannlabs/scPortrait --clone --remote
```

This will fork the repo to your github account, create a clone of the repo on your current machine, add our repository as a remote, and set the `main` development branch to track our repository.

To do this manually, first make a fork of the repository by clicking the "fork" button on our main github package. Then, on your machine, run:

```console
$ # Clone your fork of the repository (substitute in your username)
$ git clone https://github.com/{your-username}/scPortrait.git
$ # Enter the cloned repository
$ cd scanpy
$ # Add our repository as a remote
$ git remote add upstream https://github.com/mannlabs/scPortrait.git
$ # git branch --set-upstream-to "upstream/main"
```

### setting up `pre-commit`

We use [pre-commit][] to run some styling checks in an automated way.
We also test against these checks, so make sure you follow them!

You can install pre-commit with:

```console
$ pip install pre-commit
```

You can then install it to run while developing here with:

```console
$ pre-commit install
```

From the root of the repo.

If you choose not to run the hooks on each commit, you can run them manually with `pre-commit run --files={your files}`.

### Creating a branch for your feature

All development should occur in branches dedicated to the particular work being done.
Additionally, unless you are a maintainer, all changes should be directed at the `main` branch.
You can create a branch with:

```console
$ git checkout main                 # Starting from the main branch
$ git pull                          # Syncing with the repo
$ git switch -c {your-branch-name}  # Making and changing to the new branch
```

### Open a pull request

When you're ready to have your code reviewed, push your changes up to your fork:

```console
$ # The first time you push the branch, you'll need to tell git where
$ git push --set-upstream origin {your-branch-name}
$ # After that, just use
$ git push
```

And open a pull request by going to the main repo and clicking *New pull request*.
GitHub is also pretty good about prompting you to open PRs for recently pushed branches.

We'll try and get back to you soon!

## Creating a development Environment

It's recommended to do development work in an isolated environment.
There are number of ways to do this, including virtual environments, conda environments, and virtual machines.

We use conda environments. To setup a conda environment please do the following:

```console
conda create -n "{dev_environment_name}" python=3.12
conda activate {dev_environment_name}
pip install scportrait[dev]
```

<!-- Links -->

[Atlassian's git tutorial]: https://www.atlassian.com/git/tutorials
[Setting up git for GitHub]: https://docs.github.com/en/free-pro-team@latest/github/getting-started-with-github/set-up-git
[pre-commit.ci]: https://pre-commit.ci/
[pre-commit]: https://pre-commit.com/
[GitHub's CLI]: https://cli.github.com
[sphinx]: https://www.sphinx-doc.org/en/master/
[sphinx autodoc typehints]: https://github.com/tox-dev/sphinx-autodoc-typehints
[sphinx google style docstrings]: https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html
[scPortrait Notebooks]: (https://github.com/MannLabs/scPortrait-notebooks)
[pytest]: https://docs.pytest.org/en/stable/

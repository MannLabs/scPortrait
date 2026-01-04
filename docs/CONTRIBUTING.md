# Contributing guide

Contributions to scPortrait are welcome! This section provides some guidelines and tips to follow when contributing.

## Creating a development Environment and installing dev dependencies

It's recommended to do development work in an isolated environment.
There are number of ways to do this, including virtual environments, conda environments, and virtual machines.

We use conda environments. To setup a conda environment please do the following:

```console
conda create -n "{dev_environment_name}" python=3.12
conda activate {dev_environment_name}
```

In addition to the packages needed to _use_ this package, you need additional python packages to _run tests_ and _build the documentation_. It's easy to install them using `pip` by adding the `dev` tag:

### Interactive Installation
```console
git clone https://github.com/MannLabs/scPortrait
cd scPortrait
pip install -e '.[dev]'
```

### Installation from PyPi
```console
pip install scportrait[dev]
```

## Committing Code Changes

We assume some familiarity with `git`. For more detailed information, we recommend the following introductions:

- [Atlassian's git tutorial][] — beginner-friendly introduction to the git command line
- [Setting up git for GitHub][] — configuring git to work with your GitHub account

In short, contributing changes to scPortrait follows this workflow:

1. **Fork and clone the repository**
   See: [Forking and cloning](#forking-and-cloning)

2. **Set up pre-commit hooks**
   Ensures consistent formatting and linting across contributions.
   See: [Setting up `pre-commit`](#setting-up-pre-commit)

3. **Create a new branch for your changes**
   Development should occur on a separate branch.
   See: [Creating a branch for your feature](#creating-a-branch-for-your-feature)

4. **Commit your changes with clear commit messages**
   See: [Commit Message Guidelines](#commit-message-guidelines)

5. **Open a pull request**
   Submit your branch to the `main` branch of the main repository.
   See: [Open a pull request](#open-a-pull-request)

We encourage small, focused pull requests, as these are easier to review and discuss.

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
$ cd scPortrait
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

### Commit Message Guidelines

To keep the commit history clear and easy to navigate, we use short, descriptive commit messages with a conventional prefix indicating the type of change:

`[TAG] Short, clear description in imperative form`

Following this style is encouraged for all contributions, but do not worry if you forget — maintainers may adjust commit messages during PR squash-merge.

#### Format
- **Use the imperative mood** (“Fix bug”, “Add feature”), not past tense.
- **Keep it brief** (ideally under 60 characters).
- **Do not include author names or PR numbers** in the commit message itself (GitHub tracks that automatically).

#### Recommended Tags

| Tag | Purpose | Examples |
|---|---|---|
| `FEATURE` | Adding new functionality | `[FEATURE] Add ConvNextFeaturizer` |
| `FIX` | Bug fixes and corrections | `[FIX] Ensure sharding is resolved correctly` |
| `IMPROVE` | Enhancements to existing code or performance | `[IMPROVE] Handling of empty SpatialData files` |
| `DOCS` | Documentation updates | `[DOCS] Update cellpose segmentation guide` |
| `REFactor` | Code restructuring without changing behavior | `[REFACTOR] Simplify project status tracking` |
| `TEST` | New or updated test coverage | `[TEST] Add tests for HDF5 extraction workflow` |
| `CI` | Continuous integration / workflow updates | `[CI] Run tests on pull requests` |
| `VERSION` | Version updates performed by automation | `[VERSION] Bump version to 1.5.0` |

#### Examples Based on Previous Commits

| Original Commit | Improved Commit Message |
|---|---|
| fix some small bugs | `[FIX] Resolve minor segmentation edge cases` |
| improve spatialdata file handling | `[IMPROVE] Robust handling of backed SpatialData stores` |
| implement automatic workflow for bumping version numbers | `[CI] Add automated version bump workflow` |
| Update docs | `[DOCS] Expand documentation for project setup` |
| Ensure dtypes are consistent over all image tiles during stitching | `[FIX] Ensure dtype consistency across stitched tiles` |

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
-   there should be no specification of type in the docstrings (this should be done in the function call with mypy style type hints instead)
-   add type hints for use with mypy
-   example code
-   automatic building with Sphinx

Refer to [sphinx google style docstrings][] for detailed information on writing documentation.

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

## Writing Tests

If you introduce a new function or modify existing functionality, be sure to include corresponding **unit tests**. Tests help ensure correctness, stability, and maintainability across the codebase.

## Running Tests

scPortrait uses [pytest][] for testing and maintains two categories of test suites:
1.  Unit Tests
2.  End-to-End (E2E) Tests

Unit Tests are run whenever a commit is made in a PR. Before opening a pull request, ensure **unit tests pass locally**.

The E2E tests are computationally more expensive and run the entire scPortrait processing pipeline and as such are much slower. These tests are only run on commits made in the main branch e.g. upon merging a PR. Before merging into `main`, please ensure **E2E tests pass**.

A lot of warnings can be thrown while running the test files. It’s often easier to read the test results with them hidden via the `--disable-pytest-warnings`  argument.

### Unit Tests

Unit tests validate individual functions or modules in isolation (e.g., file readers, plotting utilities, segmentation helpers). These tests are designed to be fast and run frequently during development.

Run only unit tests:
```console
pytest tests/unit_tests --disable-warnings
```

### End-to-End (E2E) Tests

E2E tests run larger workflow scenarios to ensure that multiple components of the pipeline work correctly together (e.g., segmentation → extraction → featurization → selection).
These tests are **much slower** and may require larger example data.

Run only E2E tests:
```console
pytest tests/e2e_tests --disable-warnings
```

### Running All Tests

To run both unit and E2E tests together:
```console
pytest .
```

## Tutorials with jupyter notebooks

Indepth tutorials using jupyter notebooks are hosted in a dedicated repository: [scPortrait Notebooks](https://github.com/MannLabs/scPortrait-notebooks).

Please update and/or add new tutorials there.

## Creating a New Release

Before creating a release, ensure that **all tests pass** for the code currently on `main` (unit tests and end-to-end tests). If any tests fail, fix the issues before proceeding.

Releases are versioned and published through automated GitHub workflows. The release process consists of the following steps:

### 1. Bump the version

The version is incremented using the **Bump version** workflow:

1. Open: <https://github.com/MannLabs/scPortrait/actions/workflows/bump_version.yml>
2. Run the workflow and specify the version increment (`patch`, `minor`, or `major`).

This will create **two pull requests**:
   - **`[VERSION] Bump version to X.Y.Z`** — *merge this now*
   - **`[VERSION] Bump version to X.Y.Z-dev0`** — *merge this after the release*

### 2. Merge the version bump PR

1. Merge the PR titled: `[VERSION] Bump version to X.Y.Z`

This sets the release version in the codebase.

### 3. Create a draft release

1. Open: <https://github.com/MannLabs/scPortrait/actions/workflows/create_release.yml>
2. Run the workflow and specify:
   - **Branch:** `main`

This workflow will create a **draft release** automatically.

### 4. Finalize the GitHub release

1. Go to the Releases page: <https://github.com/MannLabs/scPortrait/releases>
2. Select the **draft** release that was generated.
3. Click **"Generate release notes"** to automatically populate the changelog from commit messages.
4. Review and publish the release.

### 5. Publish the release to PyPI

1. Open: <https://github.com/MannLabs/scPortrait/actions/workflows/publish_on_pypi.yml>
2. Run the **Publish on PyPI** workflow.
3. Enter the version number you just released.
4. Wait for the workflow to complete successfully.

This workflow:
- Builds and uploads the package to PyPI
- Runs validation tests against the published package

### 6. Merge the post-release development version bump

Once PyPI publication is confirmed, merge the second PR: `[VERSION] Bump version to X.Y.Z-dev0`

This transitions the project back into a development state.


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

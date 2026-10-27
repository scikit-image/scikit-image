# Installation

Your system needs a:

- C compiler,
- C++ compiler, and
- a version of Python supported by `scikit-image` (see
  [pyproject.toml](https://github.com/scikit-image/scikit-image/blob/main/pyproject.toml#L14)).

First, [fork the scikit-image repository on GitHub](https://github.com/scikit-image/scikit-image/fork).
Then clone your fork locally and set an `upstream` remote to point to the original scikit-image repository:

:::{note}
We use `git@github.com` below; if you don't have SSH keys setup, use
`https://github.com` instead.
:::

```sh
git clone git@github.com:YOURUSERNAME/scikit-image
cd scikit-image
git remote add upstream git@github.com:scikit-image/scikit-image
```

All commands below are run from within the cloned `scikit-image` directory.

(build-env-setup)=

## Build environment setup

Set up a Python development environment tailored for scikit-image.
Here we provide instructions for two popular environment managers:
`venv` (pip) and `conda` (miniforge).

### venv

```sh
# Create a virtualenv named ``skimage-dev`` that lives outside of the repository.
# One common convention is to place it inside an ``envs`` directory under your home directory:
mkdir ~/envs
python -m venv ~/envs/skimage-dev

# Activate it
# (On Windows, use ``skimage-dev\Scripts\activate``)
source ~/envs/skimage-dev/bin/activate

# Install development dependencies
pip install -r requirements.txt

# Install scikit-image in editable mode. In editable mode,
# scikit-image will be recompiled, as necessary, on import.
spin install -v
```

:::{tip}
The above installs scikit-image into your environment, which makes
it accessible to IDEs, IPython, etc.
This is not strictly necessary; you can also build with:

```sh
spin build
```

In that case, the library is not installed, but is accessible via
`spin` commands, such as `spin test`, `spin ipython`, `spin run`,
etc.
:::

### conda

We recommend installing conda using
[miniforge](https://github.com/conda-forge/miniforge),
an alternative to Anaconda without licensing costs.

After installing miniforge:

```sh
# Create a conda environment with required dependencies
conda env create -f environment.yml

# Activate it
conda activate skimage-dev

# Install scikit-image in editable mode. In editable mode,
# scikit-image will be recompiled, as necessary, on import.
spin install -v
```

:::{tip}
The above installs scikit-image into your environment, which makes
it accessible to IDEs, IPython, etc.
This is not strictly necessary; you can also build with:

```sh
spin build
```

In that case, the library is not installed, but is accessible via
`spin` commands, such as `spin test`, `spin ipython`, `spin run`,
etc.
:::

## Adding a feature branch

When contributing a new feature, do so via a feature branch.

First, fetch the latest source:

```sh
git switch main
git pull upstream main
```

Create your feature branch:

```sh
git switch --create my-feature-name
```

Using an editable install, `scikit-image` will rebuild itself as
necessary.
If you are building manually, rebuild with:

```sh
spin build
```

Repeated, incremental builds usually work just fine, but if you notice build
problems, rebuild from scratch using:

```sh
spin build --clean
```

## Platform-specific notes

**Windows**

Building `scikit-image` on Windows is done as part of our continuous
integration testing; the steps are shown in this [Azure Pipeline].

**Debian and Ubuntu**

Install suitable compilers prior to library compilation:

```sh
sudo apt-get install build-essential
```

## Full requirements list

**Build Requirements**

```{eval-rst}
.. include:: ../../../requirements/build.txt
   :literal:
```

**Runtime Requirements**

```{eval-rst}
.. include:: ../../../requirements/default.txt
   :literal:
```

**Test Requirements**

```{eval-rst}
.. include:: ../../../requirements/test.txt
   :literal:
```

**Documentation Requirements**

```{eval-rst}
.. include:: ../../../requirements/docs.txt
   :literal:
```

**Developer Requirements**

```{eval-rst}
.. include:: ../../../requirements/developer.txt
   :literal:
```

**Data Requirements**

The full selection of demo datasets is only available with the
following installed:

```{eval-rst}
.. include:: ../../../requirements/data.txt
   :literal:
```

**Optional Requirements**

You can use `scikit-image` with the basic requirements listed above, but some
functionality is only available with the following installed:

- [Matplotlib](https://matplotlib.org)
  Used in various functions, e.g., for drawing, segmenting, reading images.
- [Dask](https://dask.org/)
  The `dask` module is used to parallelize certain functions.

More rarely, you may also need:

- [PyAMG](https://pyamg.org/)
  The `pyamg` module is used for the fast `cg_mg` mode of random
  walker segmentation.
- [Astropy](https://www.astropy.org)
  Provides FITS I/O capability.
- [SimpleITK](http://www.simpleitk.org/)
  Optional I/O plugin providing a wide variety of [formats](https://itk.org/Wiki/ITK_File_Formats).
  including specialized formats used in biomedical imaging.

```{eval-rst}
.. include:: ../../../requirements/optional.txt
  :literal:
```

## Help with contributor installation

See {ref}`additional-help`.

[azure pipeline]: https://github.com/scikit-image/scikit-image/blob/main/azure-pipelines.yml

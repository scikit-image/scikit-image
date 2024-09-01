.. _installing-scikit-image:

Installing scikit-image
==============================================================================

- First, you need to have the Python language installed.
  Two popular routes are the pip-based
  `Python.org installers <https://www.python.org/downloads/>`_
  and the conda-based
  `miniforge <https://github.com/conda-forge/miniforge>`_.

- Install ``scikit-image`` via `pip <#install-via-pip>`_ or `conda
  <#install-via-conda>`_, as appropriate.

- Or, `build the package from source
  <#installing-scikit-image-for-contributors>`_.
  Do this if you'd like to contribute to development.

Supported platforms
------------------------------------------------------------------------------

- Windows 64-bit on x86 processors
- macOS on x86 and ARM (M1, etc.) processors
- Linux 64-bit on x86 and ARM processors

While we do not officially support other platforms, you could still
try `building from source <#building-from-source>`_.

Version check
------------------------------------------------------------------------------

To see whether ``scikit-image`` is already installed or to check if an install has
worked, run the following in a Python shell or Jupyter notebook:

.. code-block:: python

  import skimage as ski
  print(ski.__version__)

or, from the command line:

.. code-block:: sh

   python -c "import skimage; print(skimage.__version__)"

(Try ``python3`` if ``python`` is unsuccessful.)

You'll see the version number if ``scikit-image`` is installed and
an error message otherwise.

Installation via pip and conda
------------------------------------------------------------------------------

.. _install-via-pip:

pip
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Prerequisites to a pip install: you must be able to use ``pip`` on
your command line to install packages.

We strongly recommend the use of a
`virtual environment
<https://towardsdatascience.com/virtual-environments-104c62d48c54?gi=2532aa12906#ee81>`_.
A virtual environment creates a clean Python environment that does not interfere
with the existing system installation, can be easily removed, and contains only
the package versions your application needs.

To install the current ``scikit-image`` you'll need at least Python 3.10. If
your Python is older, pip will find the most recent compatible version.

.. code-block:: sh

  # Update pip
  python -m pip install -U pip

  # Install scikit-image
  python -m pip install -U scikit-image

Some additional dependencies are required to access all example
datasets in ``skimage.data``. Install them using:

.. code-block:: sh

   python -m pip install -U scikit-image[data]

To install optional scientific Python packages that expand
``scikit-image``'s capabilities to include, e.g., parallel processing,
use:

.. code-block:: sh

    python -m pip install -U scikit-image[optional]

.. warning::

    Do not use the command ``sudo`` and ``pip`` together as ``pip`` may
    overwrite critical system libraries.


.. _install-via-conda:

conda
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We recommend `miniforge <https://github.com/conda-forge/miniforge>`_, a minimal
distribution that makes use of `conda-forge <https://conda-forge.org>`_.
It installs Python and provides virtual environments.

Once you have your conda environment set up, install ``scikit-image`` with:

.. code-block:: sh

    conda install scikit-image


System package managers
------------------------------------------------------------------------------

Using a package manager (``apt``, ``dnf``, etc.) to install ``scikit-image``
or other Python packages is not your best option, since you're likely
to get an older version. It also becomes harder to install other Python packages
not provided by the package manager.


Downloading all demo datasets
------------------------------------------------------------------------------

Some of our example images (in ``skimage.data``) are hosted online and are
not installed by default. These images are downloaded upon first
access. If you prefer to download all demo datasets, so they can be
accessed offline, ensure that ``pooch`` is installed, then run:

.. code-block:: sh

    python -c 'import skimage as ski; ski.data.download_all()'


Additional help
------------------------------------------------------------------------------

If you still have questions, reach out through

- our `user forum <https://forum.image.sc/tags/scikit-image>`_
- our `developer forum <https://discuss.scientific-python.org/c/contributor/skimage>`_
- our `chat channel <https://skimage.zulipchat.com/>`_

To suggest a change in these instructions,
`please open an issue on GitHub <https://github.com/scikit-image/scikit-image/issues/new>`_.


Installing scikit-image for contributors
========================================

Your system needs a:

- C compiler,
- C++ compiler, and
- a version of Python supported by ``scikit-image`` (see
  `pyproject.toml <https://github.com/scikit-image/scikit-image/blob/main/pyproject.toml#L14>`_).

First, `fork the scikit-image repository on GitHub <https://github.com/scikit-image/scikit-image/fork>`_.
Then clone your fork locally and set an ``upstream`` remote to point to the original scikit-image repository:

.. note::

    We use ``git@github.com`` below; if you don't have SSH keys setup, use
    ``https://github.com`` instead.

.. code-block:: sh

   git clone git@github.com:YOURUSERNAME/scikit-image
   cd scikit-image
   git remote add upstream git@github.com:scikit-image/scikit-image

All commands below are run from within the cloned ``scikit-image`` directory.

.. _build-env-setup:

Build environment setup
------------------------------------------------------------------------------

Set up a Python development environment tailored for scikit-image.
Here we provide instructions for two popular environment managers:
``venv`` (pip) and ``conda`` (miniforge).

venv
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: sh

  # Create a virtualenv named ``skimage-dev`` that lives outside of the repository.
  # One common convention is to place it inside an ``envs`` directory under your home directory:
  mkdir ~/envs
  python -m venv ~/envs/skimage-dev

  # Activate it
  # (On Windows, use ``skimage-dev\Scripts\activate``)
  source ~/envs/skimage-dev/bin/activate

  # Install development dependencies
  pip install -r requirements.txt
  pip install -r requirements/build.txt

  # Install scikit-image in editable mode. In editable mode,
  # scikit-image will be recompiled, as necessary, on import.
  spin install -v

.. tip::

    The above installs scikit-image into your environment, which makes
    it accessible to IDEs, IPython, etc.
    This is not strictly necessary; you can also build with:

    .. code-block:: sh

        spin build

    In that case, the library is not installed, but is accessible via
    ``spin`` commands, such as ``spin test``, ``spin ipython``, ``spin run``,
    etc.

conda
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We recommend installing conda using
`miniforge <https://github.com/conda-forge/miniforge>`_,
an alternative to Anaconda without licensing costs.

After installing miniforge:

.. code-block:: sh

  # Create a conda environment named ``skimage-dev``
  conda create --name skimage-dev

  # Activate it
  conda activate skimage-dev

  # Install development dependencies
  conda install -c conda-forge --file requirements/default.txt
  conda install -c conda-forge --file requirements/test.txt
  conda install -c conda-forge pre-commit ipython
  conda install -c conda-forge --file requirements/build.txt

  # Install scikit-image in editable mode. In editable mode,
  # scikit-image will be recompiled, as necessary, on import.
  spin install -v

.. tip::

    The above installs scikit-image into your environment, which makes
    it accessible to IDEs, IPython, etc.
    This is not strictly necessary; you can also build with:

    .. code-block:: sh

        spin build

    In that case, the library is not installed, but is accessible via
    ``spin`` commands, such as ``spin test``, ``spin ipython``, ``spin run``,
    etc.


Testing
-------

Run the complete test suite:

.. code-block:: sh

   spin test

Or run a subset of tests:

.. code-block:: sh

   # Run tests in a given file
   spin test skimage/morphology/tests/test_gray.py

   # Run tests in a given directory
   spin test skimage/morphology

   # Run tests matching a given expression
   spin test -- -k local_maxima


Adding a feature branch
------------------------------------------------------------------------------

When contributing a new feature, do so via a feature branch.

First, fetch the latest source:

.. code-block:: sh

   git switch main
   git pull upstream main

Create your feature branch:

.. code-block:: sh

   git switch --create my-feature-name

Using an editable install, ``scikit-image`` will rebuild itself as
necessary.
If you are building manually, rebuild with::

.. code-block:: sh

   spin build

Repeated, incremental builds usually work just fine, but if you notice build
problems, rebuild from scratch using:

.. code-block:: sh

   spin build --clean

Platform-specific notes
------------------------------------------------------------------------------

**Windows**

Building ``scikit-image`` on Windows is done as part of our continuous
integration testing; the steps are shown in this `Azure Pipeline`_.

.. _Azure Pipeline: https://github.com/scikit-image/scikit-image/blob/main/azure-pipelines.yml

**Debian and Ubuntu**

Install suitable compilers prior to library compilation:

.. code-block:: sh

  sudo apt-get install build-essential


Full requirements list
----------------------
**Build Requirements**

.. include:: ../../../requirements/build.txt
   :literal:

**Runtime Requirements**

.. include:: ../../../requirements/default.txt
   :literal:

**Test Requirements**

.. include:: ../../../requirements/test.txt
   :literal:

**Documentation Requirements**

.. include:: ../../../requirements/docs.txt
   :literal:

**Developer Requirements**

.. include:: ../../../requirements/developer.txt
   :literal:

**Data Requirements**

The full selection of demo datasets is only available with the
following installed:

.. include:: ../../../requirements/data.txt
   :literal:

**Optional Requirements**

You can use ``scikit-image`` with the basic requirements listed above, but some
functionality is only available with the following installed:

* `Matplotlib <https://matplotlib.org>`__
  Used in various functions, e.g., for drawing, segmenting, reading images.

* `Dask <https://dask.org/>`__
  The ``dask`` module is used to parallelize certain functions.

More rarely, you may also need:

* `PyAMG <https://pyamg.org/>`__
  The ``pyamg`` module is used for the fast ``cg_mg`` mode of random
  walker segmentation.

* `Astropy <https://www.astropy.org>`__
  Provides FITS I/O capability.

* `SimpleITK <http://www.simpleitk.org/>`__
  Optional I/O plugin providing a wide variety of `formats <https://itk.org/Wiki/ITK_File_Formats>`__.
  including specialized formats used in biomedical imaging.

.. include:: ../../../requirements/optional.txt
  :literal:


Help with contributor installation
------------------------------------------------------------------------------

See `Additional help <#additional-help>`_ above.

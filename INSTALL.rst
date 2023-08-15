.. _installing-scikit-image:

Installing scikit-image
==============================================================================

How you should install ``scikit-image`` depends on your needs and skills:

- First, ensure that you have Python installed.
  Two popular alternatives are the pip-based
  `Python.org installers <https://www.python.org/downloads/>`_
  and the conda-based
  `miniforge <https://github.com/conda-forge/miniforge>`_.

- Install `scikit-image` via `pip <#install-via-pip>`_ or `conda
  <#install-via-conda>`_, as appropriate.

- Or, `build the package from source
  <#installing-scikit-image-for-contributors>`_.
  Do this if you'd like to contribute to development.

Supported platforms
------------------------------------------------------------------------------

- Windows 64-bit on x86 processors
- macOS on x86 and M (ARM) processors
- Linux 64-bit on x86 processors

While we do not officially support other platforms, you could still
try `building from source <#building-from-source>`_.

Version check
------------------------------------------------------------------------------

To see whether ``scikit-image`` is already installed or to check if an install has
worked, run the following in a Python shell or Jupyter notebook:

.. code-block:: python

  import skimage
  print(skimage.__version__)

or, from the command line:

.. code-block:: sh

   python -c "import skimage; print(skimage.__version__)"

(Try ``python3`` if ``python`` is unsuccessful.)

You'll see the version number if ``scikit-image`` is installed and
an error message otherwise.

Installation via pip and conda
------------------------------------------------------------------------------

These install only ``scikit-image`` and its dependencies; pip has an option to
include related packages.

.. _install-via-pip:

pip
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Prerequisites to a pip install: You're able to use your system's command line to
install packages and are using a
`virtual environment
<https://towardsdatascience.com/virtual-environments-104c62d48c54?gi=2532aa12906#ee81>`_
(any of
`several
<https://stackoverflow.com/questions/41573587/what-is-the-difference-between-venv-pyvenv-pyenv-virtualenv-virtualenvwrappe>`_\
).

While it is possible to use pip without a virtual environment, it is not advised:
virtual environments create a clean Python environment that does not interfere
with any existing system installation, can be easily removed, and contain only
the package versions your application needs. They help avoid a common
challenge known as
`dependency hell <https://en.wikipedia.org/wiki/Dependency_hell>`_.

To install the current ``scikit-image`` you'll need at least Python 3.6. If
your Python is older, pip will find the most recent compatible version.

.. code-block:: sh

  # Update pip
  python -m pip install -U pip
  # Install scikit-image
  python -m pip install -U scikit-image

To access the full selection of demo datasets, use ``scikit-image[data]``.
To include a selection of other scientific Python packages that expand
``scikit-image``'s capabilities to include, e.g., parallel processing, you
can install the package ``scikit-image[optional]``:

.. code-block:: sh

    python -m pip install -U scikit-image[optional]

.. warning::

    Please do not use the command ``sudo`` and ``pip`` together as ``pip`` may
    overwrite critical system libraries which may require you to reinstall your
    operating system.

.. _install-via-conda:

conda
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Miniconda is a bare-essentials version of the Anaconda package; you'll need to
install packages like ``scikit-image`` yourself. Like Anaconda, it installs
Python and provides virtual environments.

- `conda documentation <https://docs.conda.io>`_
- `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_
- `conda-forge <https://conda-forge.org>`_, a conda channel maintained
  with the latest ``scikit-image`` package

Once you have your conda environment set up, you can install ``scikit-image``
with the command:

.. code-block:: sh

    conda install scikit-image

System package managers
------------------------------------------------------------------------------

Using a package manager (``yum``, ``apt-get``, etc.) to install ``scikit-image``
or other Python packages is not your best option:

- You're likely to get an older version.

- You'll probably want to make updates and add new packages outside of
  the package manager, leaving you with the same kind of
  dependency conflicts you see when using pip without a virtual environment.

- There's an added risk because operating systems use Python, so if you
  make system-wide Python changes (installing as root or using sudo),
  you can break the operating system.


Downloading all demo datasets
------------------------------------------------------------------------------

Some of the data used in our examples is hosted online and is not installed
by default by the procedures explained above. Data are downloaded once, at the
first call, but this requires an internet connection. If you prefer downloading
all the demo datasets to be able to work offline, ensure that package ``pooch``
is installed and then run this command:

.. code-block:: sh

    python -c 'from skimage.data import download_all; download_all()'

or call ``download_all()`` in your favourite interactive Python environment
(IPython, Jupyter notebook, ...).


Additional help
------------------------------------------------------------------------------

If you still have questions, reach out through

- our `user forum <https://forum.image.sc/tags/scikit-image>`_
- our `developer forum <https://discuss.scientific-python.org/c/contributor/skimage>`_
- our `chat channel <https://skimage.zulipchat.com/>`_
- `Stack Overflow <https://stackoverflow.com/questions/tagged/scikit-image>`_


To suggest a change in these instructions,
`please open an issue on GitHub <https://github.com/scikit-image/scikit-image/issues/new>`_.


Installing scikit-image for contributors
========================================

We are assuming that you have a default Python environment already configured on
your computer and that you intend to install ``scikit-image`` inside of it.

We also make a few more assumptions about your system:

- You have a C compiler set up.
- You have a C++ compiler set up.
- You are running a version of Python compatible with our system as listed
  in our `pyproject.toml <https://github.com/scikit-image/scikit-image/blob/main/pyproject.toml#L14>`_.
- You've cloned the git repository into a directory called ``scikit-image``.
  You have set up the `upstream` remote to point to our repository and `origin`
  to point to your fork.


This directory contains the following files:

.. code-block::

    scikit-image
    ├── asv.conf.json
    ├── azure-pipelines.yml
    ├── benchmarks/
    ├── CITATION.bib
    ├── CODE_OF_CONDUCT.md
    ├── CONTRIBUTING.rst
    ├── CONTRIBUTORS.txt
    ├── doc/
    ├── INSTALL.rst
    ├── LICENSE.txt
    ├── MANIFEST.in
    ├── meson.build
    ├── meson.md
    ├── pyproject.toml
    ├── README.md
    ├── RELEASE.txt
    ├── requirements/
    ├── requirements.txt
    ├── skimage/
    ├── TODO.txt
    └── tools/

All commands below are assumed to be running from the ``scikit-image``
directory containing the files above.


.. _build-env-setup:

Build environment setup
------------------------------------------------------------------------------

Once you've cloned your fork of the scikit-image repository,
you should set up a Python development environment tailored for scikit-image.
You may choose the environment manager of your choice.
Here we provide instructions for two popular environment managers:
``venv`` (pip based) and ``conda`` (Anaconda or Miniconda).

venv
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: sh

  # Create a virtualenv named ``skimage-dev`` that lives outside of the repository.
  # One common convention is to place it inside an ``envs`` directory under your home directory:
  mkdir ~/envs
  python -m venv ~/envs/skimage-dev
  # Activate it
  # (On Windows, please use ``skimage-dev\Scripts\activate``)
  source ~/envs/skimage-dev/bin/activate
  # Install main development and runtime dependencies
  pip install -r requirements.txt
  # Install build dependencies of scikit-image
  pip install -r requirements/build.txt
  # Build scikit-image from source
  spin build
  # Test your installation
  spin test
  # Build docs
  spin docs
  # Try the new version in IPython
  spin ipython

conda
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When using conda for development, we
recommend adding the conda-forge channel for the most up-to-date version
of many dependencies.
Some dependencies we use (for testing and documentation) are not available
from the default Anaconda channel. Please follow the official
`conda-forge installation instructions <https://conda-forge.org/#about>`_
before you get started.

.. code-block:: sh

  # Create a conda environment named ``skimage-dev``
  conda create --name skimage-dev
  # Activate it
  conda activate skimage-dev
  # Install main development and runtime dependencies
  conda install -c conda-forge --file requirements/default.txt
  conda install -c conda-forge --file requirements/test.txt
  conda install -c conda-forge pre-commit
  # Install build dependencies of scikit-image
  pip install -r requirements/build.txt
  # Build scikit-image from source
  spin build
  # Test your installation
  spin test
  # Build docs
  spin docs
  # Try the new version
  spin python

For more information about building and using the ``spin`` package, see ``meson.md``.

Testing
-------

Test your installation for correct behavior using:

.. code-block:: sh

   pytest skimage

Updating the installation
------------------------------------------------------------------------------

When updating your installation, it is often necessary to recompile submodules
that have changed. Do so with the following commands:

.. code-block:: sh

    # Grab the latest source
    git checkout main
    git pull upstream main
    # Update the installation
    pip install -e . -vv

Platform-specific notes
------------------------------------------------------------------------------

**Windows**

A run-through of the compilation process for Windows is included in
our `setup of Azure Pipelines`_ (a continuous integration service).

.. _setup of Azure Pipelines: https://github.com/scikit-image/scikit-image/blob/main/azure-pipelines.yml

**Debian and Ubuntu**

Install suitable compilers:

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

* `SimpleITK <http://www.simpleitk.org/>`__
    Optional I/O plugin providing a wide variety of `formats <https://itk.org/Wiki/ITK_File_Formats>`__.
    including specialized formats using in medical imaging.

* `Astropy <https://www.astropy.org>`__
    Provides FITS I/O capability.

* `PyAMG <https://pyamg.org/>`__
    The ``pyamg`` module is used for the fast ``cg_mg`` mode of random
    walker segmentation.

* `Dask <https://dask.org/>`__
    The ``dask`` module is used to speed up certain functions.


.. include:: ../../../requirements/optional.txt
  :literal:


Help with contributor installation
------------------------------------------------------------------------------

See `Additional help <#additional-help>`_ above.

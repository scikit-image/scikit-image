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

- Or, :doc:`build the package from source </development/contribute>`.
  Do this if you'd like to contribute to development.

Supported platforms
------------------------------------------------------------------------------

- Windows 64-bit on x86 processors
- macOS on x86 and ARM (M1, etc.) processors
- Linux 64-bit on x86 and ARM processors

While we do not officially support other platforms, you could still
try :doc:`building from source </development/contribute>`.

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

To install the current ``scikit-image`` you'll need at least Python 3.12. If
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


.. _additional-help:

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

See the :doc:`contributing guide </development/contribute>` for instructions
on setting up a development environment and contributing to scikit-image.

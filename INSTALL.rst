Installing scikit-image
=======================

We highlight four different ways to install and use ``scikit-image`` for
Python3:

1. Pre-installed scientific python distributions (recommended)
2. Isolated environment based installation (recommended)
3. Source based installation for development purposes
4. Using the operating system's package manager (not recommended)

1. Pre-installed scientific python distributions
------------------------------------------------

The easiest way to get going with scikit-image is to use one of several
scientific python distributions that come with scikit-image (and many other
useful packages) pre-installed.

  - `Anaconda <https://www.anaconda.com/download/>`_
  - `Enthought Canopy <https://www.enthought.com/product/canopy/>`_
  - `Python(x,y) <https://python-xy.github.io/>`_
  - `WinPython <https://winpython.github.io/>`_

2. Isolated environment based installation
------------------------------------------

While the pre-installed versions above give you an easy way to start, if you
wish to have more control over the installation procedure, you may want to use
a virtual environment based approach. Two common approaches use technologies
such as ``venv`` (also known as virtual environments or ``pip`` based) and
``conda`` (installed through Anaconda or Miniconda).

venv
====
When using ``venv``, you can get started with the following commands::

  # Take note of your current directory
  pwd
  # Create a virtualenv named ``skimage`` that lives in the directory of
  # the same name
  python -m venv skimage
  # Activate it
  source skimage/bin/activate
  # Install a precompiled scikit-image wheel from PyPi
  pip install scikit-image
  # Install all other packages you need
  pip install ...

Now, whenever you want to use this specific version of scikit-image, make sure
you activate it by::

  # Change to the directory where you created the virtual environment
  cd YOUR_DIRECTORY
  # Activate it
  source skimage/bin/activate
  # Use python as you wish


Should you need to have access to the wheels offline, they can be downloaded
manually from `PyPi <https://pypi.org/project/scikit-image/>`__.

conda
=====

When using conda, you may find the following bash commands useful::

  # Create a conda environment named ``skimage``
  conda create --name skimage
  # Activate it
  conda activate skimage
  conda install scikit-image

Now, whenver you want to use this specific version of scikit-image, make sure
you activate your conda environment by typing::

  # Activate the specific conda environment
  conda activate skimage
  # Use python as you wish

While the default Anaconda installation provides scikit-image, you may wish to
explore `conda-forge <https://conda-forge.org/>`__ for a more updated version.
If you are new to conda, you may also find the following
`introduction to conda <https://kaust-vislab.github.io/introduction-to-conda-for-data-scientists/>`__
useful.

3. Source based installation for development purposes
-----------------------------------------------------

Development installation instructions can be found in the Contribution guide.
TODO: provide a cross link to this????

4. Using the operating system's package manager
-----------------------------------------------

TODO: https://github.com/scikit-image/scikit-image/pull/4305

Optional Requirements
---------------------

Some functionality of ``scikit-image`` is only available with the following
additional python package installed:

* `PyQt5 <http://wiki.python.org/moin/PyQt>`__ or `PySide2 <https://wiki.qt.io/Qt_for_Python>`__ through `qtpy <https://github.com/spyder-ide/qtpy>`__
    A ``Qt`` plugin will provide ``imshow(x, fancy=True)`` and `skivi`.

* `PyAMG <http://pyamg.org/>`__
    The ``pyamg`` module is used for the fast `cg_mg` mode of random
    walker segmentation.

* `Astropy <http://www.astropy.org>`__
    Provides FITS I/O capability.

* `SimpleITK <http://www.simpleitk.org/>`__
    Optional I/O plugin providing a wide variety of `formats <http://www.itk.org/Wiki/ITK_File_Formats>`__.
    including specialized formats using in medical imaging.

* `imread <http://pythonhosted.org/imread/>`__
    Optional I/O plugin providing most standard `formats <http://pythonhosted.org//imread/formats.html>`__.

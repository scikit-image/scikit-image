Installing scikit-image
=======================

We are assuming that you have default Python environment already configured on
your computer and you intend to install ``scikit-image`` inside of it. If you
want to create and work with Python virtual environments, please follow the
instructions on `venv`_ and `virtual environments`_.

There are two ways you can install ``scikit-image`` on your preferred Python
environment.


1. Pre-installed scientific python distributions
2. Virtual environment based installation
3. Development Installation


1. Pre-installed scientific python distributions
------------------------------------------------

The fastest way to get going with scikit-image is to use one of several
scientific python distributions that come with scikit-image (and many other
useful packages) pre-installed.

  - `Anaconda <https://www.anaconda.com/download/>`_
  - `Enthought Canopy <https://www.enthought.com/product/canopy/>`_
  - `Python(x,y) <https://python-xy.github.io/>`_
  - `WinPython <https://winpython.github.io/>`_

2. Standard Installation:
-------------------------

While the pre-installed versions above give you an easy way to start, if you
wish to have more control over the installation procedure, you may want to use
a virtual environment based approach. Two common approaches use technologies
such as ``venv`` (``pip`` based) and ``conda`` (through Anaconda or Miniconda).

venv
====
When using ``venv``, you may find the following bash commands useful::

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

Now, whenver you want to use this specific version of scikit-image, make sure
you activate it by::

  # Change to the directory where you created the virtual environment
  cd YOUR_DIRECTORY
  # Activate it
  source skimage/bin/activate
  # Use python as you wish


The wheels can be downloaded manually from `PyPI <https://pypi.org/project/scikit-image/#files>`__.

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
explore `conda-forge <https://conda-forge.org/>`_ for a more updated version.

2. Development Installation:
----------------------------

You can install the ``scikit-image`` development version if either your
distribution ships an outdated version or you want to develop and work on new
features before the package is released officially.

First, uninstall any existing installations::

  pip uninstall scikit-image

or, on conda-based systems::

  conda uninstall scikit-image

Now, clone scikit-image on your local computer, and install::

  git clone https://github.com/scikit-image/scikit-image.git
  cd scikit-image
  pip install -e .

To update the installation::

  git pull  # Grab latest source
  pip install -e .  # Reinstall

Platform-specific notes follow below.

a. Windows
``````````
If you experience the error ``Error:unable to find vcvarsall.bat`` it means
that your computer does not have recommended compilers for Python. You can
either download and install Windows compilers from `here`_  or use
`MinGW compilers`_ . If using `MinGW`, make sure to correctly configure
``distutils`` by modifying (or create, if not existing) the configuration file
``distutils.cfg`` (located for example at
``C:\Python26\Lib\distutils\distutils.cfg``) to contain::

  [build]
   compiler=mingw32

A run-through of the compilation process for Windows is included in
our `setup of Azure Pipelines`_ (a continuous integration service).

.. _miniconda: http://conda.pydata.org/miniconda.html
.. _python.org: http://python.org/
.. _setup of Azure Pipelines: https://github.com/scikit-image/scikit-image/blob/master/azure-pipelines.yml
.. _here: https://wiki.python.org/moin/WindowsCompilers#Microsoft_Visual_C.2B-.2B-_14.0_standalone:_Visual_C.2B-.2B-_Build_Tools_2015_.28x86.2C_x64.2C_ARM.29
.. _venv: https://docs.python.org/3/library/venv.html
.. _virtual environments: https://docs.python-guide.org/en/latest/dev/virtualenvs/
.. _MinGW compilers: http://www.mingw.org/wiki/howto_install_the_mingw_gcc_compiler_suite

b. Debian and Ubuntu
````````````````````
Install all the required dependencies::

  sudo apt-get install python3-matplotlib python3-numpy python3-pil python3-scipy python3-tk

Install suitable compilers::

  sudo apt-get install build-essential cython3

Complete the general development installation instructions above.

Build Requirements
------------------
.. include:: ../../requirements/build.txt
   :literal:

Documentation Requirements
--------------------------

.. include:: ../../requirements/docs.txt
   :literal:

Runtime Requirements
--------------------

.. include:: ../../requirements/default.txt
   :literal:

Optional Requirements
---------------------

You can use ``scikit-image`` with the basic requirements listed above, but some
functionality is only available with the following installed:

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

Testing Requirements
--------------------

.. include:: ../../requirements/test.txt
   :literal:

Warnings during testing phase
-----------------------------

Scikit-image tries to catch all warnings in its development builds to ensure
that crucial warnings from dependencies are not missed.  This might cause
certain tests to fail if you are building scikit-image with versions of
dependencies that were not tested at the time of the release. To disable
failures on warnings, export the environment variable
``SKIMAGE_TEST_STRICT_WARNINGS`` with a value of `0` or `False` and run the
tests::

   export SKIMAGE_TEST_STRICT_WARNINGS=False
   pytest --pyargs skimage

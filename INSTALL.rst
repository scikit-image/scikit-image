Installing scikit-image
=======================

We are assuming that you have default Python environment already configured on
your computer and you intend to install ``scikit-image`` inside of it. If you
want to create and work with Python virtual environments, please follow the
instructions on `venv`_ and `virtual environments`_.

There are two ways you can install ``scikit-image`` on your preferred Python
environment.

1. Standard Installation
2. Development Installation

1. Standard Installation:
-------------------------

``scikit-image`` comes pre-installed with several Python distributions,
including `Anaconda <https://www.anaconda.com/distribution/>`_,
`Python(x,y) <https://python-xy.github.io/>`_ and
`WinPython <https://winpython.github.io/>`_.

On all major operating systems, install it via shell/command prompt::

  pip install scikit-image

If you are running Anaconda or miniconda, use::

  conda install -c conda-forge scikit-image

The wheels can be downloaded manually from `PyPI <https://pypi.org/project/scikit-image/#files>`__.

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

**a. Windows**

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
.. _python.org: https://www.python.org/
.. _setup of Azure Pipelines: https://github.com/scikit-image/scikit-image/blob/master/azure-pipelines.yml
.. _here: https://wiki.python.org/moin/WindowsCompilers#Microsoft_Visual_C.2B-.2B-_14.0_standalone:_Visual_C.2B-.2B-_Build_Tools_2015_.28x86.2C_x64.2C_ARM.29
.. _venv: https://docs.python.org/3/library/venv.html
.. _virtual environments: https://docs.python-guide.org/dev/virtualenvs/
.. _MinGW compilers: http://www.mingw.org/wiki/howto_install_the_mingw_gcc_compiler_suite

**b. Debian and Ubuntu**

Install all the required dependencies::

  sudo apt-get install python3-matplotlib python3-numpy python3-pil python3-scipy python3-tk

Install suitable compilers::

  sudo apt-get install build-essential cython3

Complete the general development installation instructions above.

Build Requirements
------------------
* `Python >= 3.5 <https://www.python.org/>`__
* `Numpy >= 1.11 <https://numpy.org/>`__
* `Cython >= 0.23.4 <https://cython.org/>`__

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

* `PyQt5 <https://wiki.python.org/moin/PyQt>`__ or `PySide2 <https://wiki.qt.io/Qt_for_Python>`__ through `qtpy <https://github.com/spyder-ide/qtpy>`__
    A ``Qt`` plugin will provide ``imshow(x, fancy=True)`` and `skivi`.

* `PyAMG <https://pyamg.org/>`__
    The ``pyamg`` module is used for the fast `cg_mg` mode of random
    walker segmentation.

* `Astropy <https://www.astropy.org>`__
    Provides FITS I/O capability.

* `SimpleITK <http://www.simpleitk.org/>`__
    Optional I/O plugin providing a wide variety of `formats <https://itk.org/Wiki/ITK_File_Formats>`__.
    including specialized formats using in medical imaging.

* `imread <https://pythonhosted.org/imread/>`__
    Optional I/O plugin providing most standard `formats <https://pythonhosted.org//imread/formats.html>`__.

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

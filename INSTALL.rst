Installing scikit-image
=======================

We outline 4 different ways that we recommend to get started with image
processing and scikit-image.

1. `Scientific Python Distributions`_ -- Recommended
2. `pip based installation`_
3. `conda based installation`_
4. `System-wide package manager`_

After your installation, you should be able to run the following code
from a Python console or Jupyter notebook:

.. code-block:: python

  import skimage
  print("scikit-image installed with version " + skimage.__version__)

Should you have any questions getting started using ``scikit-image`` after
reading this guide, please open an issue on
`github <https://github.com/scikit-image/scikit-image/issues>`_
to get additional help.

For those attempting to build scikit-image from source, or trying contribute
code back to scikit-image, we've provided development instructions below.

Scientific Python Distributions
-------------------------------

``scikit-image`` comes pre-installed with several Python distributions.
These have the advantage of comming pre-installed with many other useful
packages for scientific computing and image analysis. Follow the instructions
on the respective websites to get going. If you don't have python installed on
your computer, these are the fastest way to get started with your project.
As of June 2020, we recommend one of the following 3 distributions:

- `Anaconda <https://www.anaconda.com/distribution/>`_
- `Python(x,y) <https://python-xy.github.io/>`_
- `WinPython <https://winpython.github.io/>`_

Note that the version of ``scikit-image`` installed with these distributions
may different than the one you expect. This should not stop you from getting
started with your project. We simply suggest you check the installed version of
``scikit-image`` with the code snippet included at the top of this guide so
that you can refer to the appropriate documentation version.


pip based installation
----------------------

To install ``scikit-image`` with `pip <https://pip.pypa.io/en/stable/>`_, you
may use the following commands

.. code-block:: sh

  # Update pip to a more recent version
  python3 -m pip install -U pip
  # Install scikit-image
  python3 -m pip install -U scikit-image


This is the easiest way of obtaining the latest released version of
``scikit-image`` if python is already available on your system.

We also offer an easy way to install related packages. These are often included
in scientific python distributions. To install many of them using ``pip`` use
the command:

.. code-block:: sh

    python3 -m pip install scikit-image[optional]


.. tip::

    Scientific Python Distributions listed in the first section take care to
    only install compatibile packages together. Often conflicts can arise
    between specific combinations of packages. To avoid breaking your python
    environments, we recommend learning about virtual environments in python.
    You can learn more about them by reading Python's official documentation on
    `Virtual Environments and Packages
    <https://docs.python.org/3/tutorial/venv.html>`_.

conda based installation
------------------------

`conda <https://docs.conda.io>`_ can be a useful package manager for building
applications with dependencies not available on pypi in a cross-platform
setting. While detailed instructions on how to use conda are considered out of
scope in this tutorial, we encourage you to check these resources:

- `conda's official documnetation <https://docs.conda.io>`_
- `conda-forge <https://conda-forge.org>`_ a channel maintained with the latest scikit-image package.
- `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_

System-wide package manager
---------------------------

**Linux only:** Because installing ``scikit-image`` using your system's package
manager can have consequences on your system as a whole, we do not recommend
using this strategy. However, if you intend to use your system's package
manager to install scikit-image, we recommend reading the official
documentation from your operating system. Each operating system, and Linux
distribution, has its own package manager, and as such the exact installation
steps might differ for each system.

We warn that this may install an older version of scikit-image which we may not
support anymore.

For brevity, we provide instructions for getting started with Ubuntu.
To install scikit-image, you may use the command:

.. code-block:: sh

    sudo apt install python3-skimage

We refer users to Ubuntu's documention on `apt-get
<https://help.ubuntu.com/community/AptGet/Howto>`_ and their page listed the
information on their `python3-skimage package
<https://packages.ubuntu.com/focal/python/python3-skimage>`_.


.. warning::

    Please do not use the command ``sudo`` and ``pip`` together as ``pip`` may
    overwrite critical system libraries which may require you to reinstall your
    operating system.


Common challenges
-----------------

The following are a few common challenges encountered in install
``scikit-image``.

**Multiple Python installations**

Different python installations can cause confusion. The default command
``python3`` you type may not be the one you expect. To check which python
executable you are using, use the commands:


.. code-block:: sh

    # For Linux and MAC
    which python3
    # For Windows
    where python3

From your python console or Jupyter notebook, you may use the python commands:

.. code-block:: python

    import sys
    print(sys.executable)
    import skimage
    print("scikit-image installed with version " + skimage.__version__)

to make sure the executable used matches your expectations.

**sudo and pip**

Never use ``sudo`` when installing packages with ``pip``. This
may cause ``pip`` to overwrite important files for your operating system
which may require you to completely reinstall your system.


**User local packages**

Packages installed by your user account (using pip) may
take precedence over packages installed in your python environment.
If you have trouble getting the correct version of ``scikit-image``
take a look at the contents of the directories (if they exist):

- ``~/.local/lib/python3.6``
- ``~/.local/lib/python3.7``
- ``~/.local/lib/python3.8``

If you are using virtual environments, conda, or the system package manager,
then the installation files in those directories are not necessary, and can
safely be deleted.


Supported platforms
-------------------

The platforms officially supported by scikit-image are:

1. Windows 64 bit
2. Mac OSX
3. Linux 64 bit

Since Windows 32 bit remains a popular platform, we continue to support it for
the time being.  However, we strongly recommend all users of Windows 32 bit
begin to switch over to Windows 64 bit if they can.

We are very insterested in learning how ``scikit-image`` is
`used <https://github.com/scikit-image/scikit-image/issues/4375>`_.
Help us learn what other platforms you would like to install scikit-image!

Unsupported platforms include:

1. Linux on 32 bit Intel processors.
2. Linux on 32 bit arm (Raspberry Pi running Rapsbian): While we do not official support
   this distribution, we point users to `piwheels <https://wwww.piwheels.org>`_
   and
   `scikit-image's specific page <https://www.piwheels.org/project/scikit-image/>`_.

   - You may need to install additional system dependencies listed for
     `imagecodecs <https://www.piwheels.org/project/imagecodecs/>`_.
     See
     `issue 4721 <https://github.com/scikit-image/scikit-image/issues/4721>`_.

3. Linux on 64 bit arm (NVidia Jetson):
   - Follow the conversation on `Issue 4705 <https://github.com/scikit-image/scikit-image/issues/4705>`_.

While we do not directly support the platforms above, many of the core
developers have experience using the platforms listed above. Do not hesitate to
ask us questions pertaining to your specific use case.

.. note::

    If you would like to help package scikit-image for the platforms above, reach
    out on github so that we may help you get started.


.. tip::

    If you need to install ``scikit-image`` one these platforms today, we hope
    the developer instructions below can help get you started.


End of instructions
===================


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

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

For those attempting to build scikit-image from source, or trying contribute
code back to scikit-image, we've provided instructions under the heading
`Installing scikit-image for contributors`_.

Scientific Python Distributions
-------------------------------

``scikit-image`` comes pre-installed with several Python distributions.
These have the advantage of coming pre-installed with many other useful
packages for scientific computing and image analysis. Follow the instructions
on the respective websites to get going. If you don't have python installed on
your computer, these are the fastest way to get started with your project.
As of June 2020, we recommend one of the following 3 distributions:

- `Anaconda <https://www.anaconda.com/distribution/>`_
- `Python(x,y) <https://python-xy.github.io/>`_
- `WinPython <https://winpython.github.io/>`_

Note that the version of ``scikit-image`` installed with these distributions
may different than the one you expect. This should not stop you from getting
started with your project. We suggest you check the installed version of
``scikit-image`` with the code snippet included at the top of this guide so
that you can refer to the appropriate documentation version.


pip based installation
----------------------

This is the easiest way of obtaining the latest released version of
``scikit-image`` if Python is already available on your system.


**Finding your Python command**

To install ``scikit-image`` with `pip <https://pip.pypa.io/en/stable/>`_, you
may use the following commands from a command prompt. Note that this command
prompt is not the Python command prompt, but, depending on your operating
system, this is called:

- A linux terminal window
- A MacOSX terminal window
- A Windows Command Prompt

In that window, execute the command:

.. code-block:: sh

  # For linux or mac
  which python
  # Linux and Mac might not have the command python point to python3
  which python3
  # For windows
  where python
  # Windows might not have the command python
  where python3

Once you have found your Python command, use one of the commands to establish
the version of Python you are using:

.. code-block:: sh

  python --version

To ensure the version is at least Python 3.6. If not, you may not be able
to install the most updated version of scikit-image. However,
this should not stop you from installing an older version and get you going
with your image processing project.

We suggest you check the installed version of ``scikit-image`` with the
code snippet included at the top of this guide so that you can refer to the
appropriate documentation version.

**Installing scikit-image**

Now that you have found the Python executable on your machine, use the commands
below to install ``scikit-image``.
The instructions below use the ``python`` command for simplificy. Make sure you
use the command that you found calls the appropriate Python executable on your
particular machine.

.. code-block:: sh

  # Update pip to a more recent version
  python -m pip install -U pip
  # Install scikit-image
  python -m pip install -U scikit-image

We also offer an easy way to install related packages. These are often included
in scientific python distributions. To install many of them using ``pip`` use
the command:

.. code-block:: sh

    python -m pip install scikit-image[optional]


.. tip::

    Scientific Python Distributions listed in the first section take care to
    only install compatibile packages together. Often conflicts can arise
    between specific combinations of packages. To avoid breaking your python
    environments, we recommend learning about virtual environments in python.
    You can learn more about them by reading Python's official documentation on
    `Virtual Environments and Packages
    <https://docs.python.org/3/tutorial/venv.html>`_.

.. warning::

    Please do not use the command ``sudo`` and ``pip`` together as ``pip`` may
    overwrite critical system libraries which may require you to reinstall your
    operating system.

conda based installation
------------------------

`conda <https://docs.conda.io>`_ can be a useful package manager for building
applications with dependencies not available on `PyPi <https://pypi.org/>`_ in
a cross-platform setting. While detailed instructions on how to use conda are
considered out of scope in this tutorial, we encourage you to check these
resources:

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
``python`` you type may not be the one you expect. To check which python
executable you are using, use the commands:


.. code-block:: sh

    # For Linux and MAC
    which python
    # or
    which python3
    # For Windows
    where python
    # or
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

Packages installed by your user account (using pip) may take precedence over
packages installed in your python environment.  If you have trouble getting the
correct version of ``scikit-image`` up and running, take a look at the contents
of the directories (if they exist). ``X`` refers to the minor version of Python
you are using

- Linux: ``~/.local/lib/python3.X``,
- Windows: ``C:\Users\YOUR_USER_NAME_HERE\AppData\Roaming\Python\Python3X``

If you are using virtual environments, conda, or the system package manager,
then the installation files in those directories are not necessary, and can
safely be deleted.

**python3 vs python2 vs python**

For a long time, the executable `python` was a pointer to `python2` on Linux.
For Ubuntu 20.04 (focal) and onward, a package
`python-is-python3 <https://packages.ubuntu.com/focal/python-is-python3>`_
exists to help `python` become a pointer to `python3`.

For most Windows installations the `python3` executable does not exist.
This is currently being worked on with conda-forge and discussed in a
`GitHub issue <https://github.com/conda-forge/python-feedstock/issues/349>`_.

Versions of ``scikit-image`` after ``0.14.2`` no longer support Python2.

Additional Help and Support
---------------------------

Should you have any questions getting started using ``scikit-image`` after
reading this guide, feel free to ask us questions on:

- Our `forum on image.sc <https://forum.image.sc/tags/scikit-image>`_
- Our `mailing list <https://mail.python.org/mailman3/lists/scikit-image.python.org/>`_
- Our `chat channel <https://skimage.zulipchat.com/>`_
- `Stack Overflow <https://stackoverflow.com/questions/tagged/scikit-image>`_

to get additional help.

Finally, if you believe there is an issue with the instructions themselves,
please open a new issue on
`GitHub <https://github.com/scikit-image/scikit-image/issues>`_.

Supported platforms
-------------------

The platforms officially supported by scikit-image are:

1. Windows 64 bit on x86 processors
2. Mac OSX on x86 processors
3. Linux 64 bit on x86 processors

Since Windows 32 bit on x86 processors remains a popular platform, we continue
to support it for the time being. However, we strongly recommend all users of
Windows 32 bit begin to switch over to Windows 64 bit if they can.

We are very insterested in learning how ``scikit-image`` is
`used <https://github.com/scikit-image/scikit-image/issues/4375>`_.
Help us learn what other platforms you would like to install scikit-image!

Unsupported platforms include:

1. Linux on 32 bit x86 processors.
2. Linux on 32 bit on ARM processors (Raspberry Pi running Rapsbian):

   - While we do not official support this distribution, we point users to
     `piwheels <https://wwww.piwheels.org>`_
     and
     `scikit-image's specific page <https://www.piwheels.org/project/scikit-image/>`_.

   - You may need to install additional system dependencies listed for
     `imagecodecs <https://www.piwheels.org/project/imagecodecs/>`_.
     See
     `issue 4721 <https://github.com/scikit-image/scikit-image/issues/4721>`_.

3. Linux on 64 bit ARM processors (NVidia Jetson):

   - Follow the conversation on
     `Issue 4705 <https://github.com/scikit-image/scikit-image/issues/4705>`_.

While we do not directly support the platforms above, many of the core
developers have experience using the platforms listed above. Do not hesitate to
ask us questions pertaining to your specific use case.

.. note::

    If you would like to help package scikit-image for the platforms above,
    reach out on GitHub so that we may help you get started.


.. tip::

    If you need to install ``scikit-image`` one these platforms today, we hope
    the developer instructions below can help get you started.


Installing scikit-image for contributors
========================================

We are assuming that you have default Python environment already configured on
your computer and you intend to install ``scikit-image`` inside of it.

We also make a few more assumptions about your system:

- You have a C compiler setup.
- You have a C++ compiler setup.
- You are running a version of python compatible with our system as listed
  in our `setup.py file <https://github.com/scikit-image/scikit-image/blob/master/setup.py#L212>`_.
- You've cloned the git repository into a directory called ``scikit-image``.
  This directory contains the following files:

.. code-block::

    scikit-image
    ├── asv.conf.json
    ├── azure-pipelines.yml
    ├── benchmarks
    ├── CODE_OF_CONDUCT.md
    ├── CONTRIBUTING.txt
    ├── CONTRIBUTORS.txt
    ├── doc
    ├── INSTALL.rst
    ├── LICENSE.txt
    ├── Makefile
    ├── MANIFEST.in
    ├── README.md
    ├── RELEASE.txt
    ├── requirements
    ├── requirements.txt
    ├── setup.cfg
    ├── setup.py
    ├── skimage
    ├── TODO.txt
    ├── tools
    └── viewer_examples

All commands below are assumed to be running from the ``scikit-image``
directory containing the files above.


Build environment setup
-----------------------

Once you've cloned your fork of the scikit-image repository,
you should set up a Python development environment tailored for scikit-image.
You may choose the environment manager of your choice.
Here we provide instructions for two popular environment managers:
``venv`` (pip based) and ``conda`` (Anaconda or Miniconda).

venv
====
When using ``venv``, you may find the following bash commands useful:

.. code-block:: sh

  # Create a virtualenv named ``skimage-dev`` that lives in the directory of
  # the same name
  python -m venv skimage-dev
  # Activate it. On Linux and MacOS:
  source skimage-dev/bin/activate
  # Install all development and runtime dependencies of scikit-image
  pip install -r <(cat requirements/*.txt)
  # Build and install scikit-image from source
  pip install -e . -vv
  # Test your installation
  pytest skimage

On Windows, please use ``skimage-dev\Scripts\activate`` on the activation step.

conda
=====

When using conda for development, we
recommend adding the conda-forge channel for the most updated version
of many dependencies.
Some dependencies we use (for testing and documentation) are not available
from the default anaconda channel. Please follow the official
`conda-forge installation instructions <https://conda-forge.org/#about>`_
before you get started.

.. code-block:: sh

  # Create a conda environment named ``skimage-dev``
  conda create --name skimage-dev
  # Activate it
  conda activate skimage-dev
  # Install major development and runtime dependencies of scikit-image
  # (the rest can be installed from conda-forge or pip, if needed)
  conda install `for i in requirements/{default,build,test}.txt; do echo -n " --file $i "; done`
  # Install minimal testing dependencies
  conda install pytest
  # Install scikit-image from source
  pip install -e . -vv
  # Test your installation
  pytest skimage

Updating the installation
=========================

When updating your isntallation, it is often necessary to recompile submodules
that have changed. Do so with the following commands:

.. code-block:: sh

    # Grab the latest source
    git checkout master
    git pull
    # Update the installation
    pip install -e . -vv

Testing
-------

``scikit-image`` has an extensive test suite that ensures correct
execution on your system.  The test suite has to pass before a pull
request can be merged, and tests should be added to cover any
modifications to the code base.

We make use of the `pytest <https://docs.pytest.org/en/latest/>`__
testing framework, with tests located in the various
``skimage/submodule/tests`` folders.

Our testing requirements are listed below:

.. include:: ../../requirements/test.txt
   :literal:


Run all tests using:

.. code-block:: sh

    pytest skimage

Or the tests for a specific submodule:

.. code-block:: sh

    pytest skimage/morphology

Or tests from a specific file:

.. code-block:: sh

    pytest skimage/morphology/tests/test_grey.py

Or a single test within that file:

.. code-block:: sh

    pytest skimage/morphology/tests/test_grey.py::test_3d_fallback_black_tophat

Use ``--doctest-modules`` to run doctests. For example, run all tests and all
doctests using:

.. code-block:: sh

    pytest --doctest-modules skimage

Warnings during testing phase
-----------------------------

Scikit-image tries to catch all warnings in its development builds to ensure
that crucial warnings from dependencies are not missed.  This might cause
certain tests to fail if you are building scikit-image with versions of
dependencies that were not tested at the time of the release. To disable
failures on warnings, export the environment variable
``SKIMAGE_TEST_STRICT_WARNINGS`` with a value of `0` or `False` and run the
tests:

.. code-block:: sh

   export SKIMAGE_TEST_STRICT_WARNINGS=False
   pytest --pyargs skimage

Platform-specific notes
-----------------------

**Windows**

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

.. _setup of Azure Pipelines: https://github.com/scikit-image/scikit-image/blob/master/azure-pipelines.yml
.. _here: https://wiki.python.org/moin/WindowsCompilers#Microsoft_Visual_C.2B-.2B-_14.0_standalone:_Visual_C.2B-.2B-_Build_Tools_2015_.28x86.2C_x64.2C_ARM.29
.. _MinGW compilers: http://www.mingw.org/wiki/howto_install_the_mingw_gcc_compiler_suite

**Debian and Ubuntu**

Install suitable compilers:

.. code-block:: sh

  sudo apt-get install build-essential


Full reuqirements list
----------------------
**Build Requirements**

.. include:: ../../requirements/build.txt
   :literal:

**Runtime Requirements**

.. include:: ../../requirements/default.txt
   :literal:

**Test Requirements**

.. include:: ../../requirements/test.txt
   :literal:

**Documentation Requirements**

.. include:: ../../requirements/docs.txt
   :literal:

**Optional Requirements**

You can use ``scikit-image`` with the basic requirements listed above, but some
functionality is only available with the following installed:

* `SimpleITK <http://www.simpleitk.org/>`__
    Optional I/O plugin providing a wide variety of `formats <https://itk.org/Wiki/ITK_File_Formats>`__.
    including specialized formats using in medical imaging.

* `Astropy <https://www.astropy.org>`__
    Provides FITS I/O capability.

* `PyQt5 <https://wiki.python.org/moin/PyQt>`__ or `PySide2 <https://wiki.qt.io/Qt_for_Python>`__ through `qtpy <https://github.com/spyder-ide/qtpy>`__
    A ``Qt`` plugin will provide ``imshow(x, fancy=True)`` and `skivi`.

* `PyAMG <https://pyamg.org/>`__
    The ``pyamg`` module is used for the fast `cg_mg` mode of random
    walker segmentation.

* `Dask <https://dask.org/>`__
    The ``dask`` module is used as to scale up certain functions.


.. include:: ../../requirements/optional.txt
  :literal:


**Extra Requirements**

These requirements have been included as a conveinence, but are not widely
installable through pypi on our supported platforms.  As such, we keep them in
a seperate list for more advanced members of our community to install.

* `imread <https://pythonhosted.org/imread/>`__
    Optional I/O plugin providing most standard `formats <https://pythonhosted.org//imread/formats.html>`__.

.. include:: ../../requirements/extras.txt
  :literal:


.. _installing-scikit-image:

Installing scikit-image
==============================================================================

How you should install ``scikit-image`` depends on your needs and skills:

- Simplest solution:
  `scientific Python distribution <#scientific-python-distributions>`_.

- If you can install Python packages and work in virtual environments:

  - `pip <#install-via-pip>`_

  - `conda <#install-via-conda>`_

- Easy solution but with pitfalls: `system package manager <#system-package-managers>`_ (yum, apt, ...).

- `You're looking to contribute to scikit-image <#installing-scikit-image-for-contributors>`_.

Supported platforms
------------------------------------------------------------------------------

- Windows 64-bit on x86 processors
- Mac OS X on x86 processors
- Linux 64-bit on x86 processors

For information on other platforms, see `other platforms <#other-platforms>`_.

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

Scientific Python distributions
------------------------------------------------------------------------------

In a single install these give you Python,
``scikit-image`` and libraries it depends on, and other useful scientific
packages. They install into an isolated environment, so they won't conflict
with any existing installed programs.

Drawbacks are that the install can be large and you may not get
the most recent ``scikit-image``.

We recommend one of these distributions:

- `Anaconda <https://www.anaconda.com/distribution/>`_
- `Python(x,y) <https://python-xy.github.io/>`_
- `WinPython <https://winpython.github.io/>`_

When using the ``scikit-image``
documentation, make sure it's for the version you've installed (see
`Version check <#version-check>`_ above).


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
all the demo datasets to be able to work offline, you can run this command:

.. code-block:: sh

    python -c 'from skimage.data import download_all; download_all()'

or call ``download_all()`` in your favourite interactive Python environment
(IPython, Jupyter notebook, ...).

Other platforms
------------------------------------------------------------------------------

We still support Windows 32-bit on x86 processors but urge switching
to Windows 64-bit.

Unsupported platforms include:

1. Linux on 32-bit x86 processors.
2. Linux on 32-bit on ARM processors (Raspberry Pi running Raspbian):

   - While we do not officially support this distribution, we point users to
     `piwheels <https://wwww.piwheels.org>`_
     and their
     `scikit-image's specific page <https://www.piwheels.org/project/scikit-image/>`_.

   - You may need to install additional system dependencies listed for
     `imagecodecs <https://www.piwheels.org/project/imagecodecs/>`_.
     See
     `issue 4721 <https://github.com/scikit-image/scikit-image/issues/4721>`_.

3. Linux on 64-bit ARM processors (Nvidia Jetson):

   - Follow the conversation on
     `issue 4705 <https://github.com/scikit-image/scikit-image/issues/4705>`_.

Although these platforms lack official support, many of the core
developers have experience with them and can help with questions.

If you want to install on an unsupported platform, try
`building from source <#building-from-source>`_.

Tell us which other platforms you'd like to see ``scikit-image`` on!
We are very interested in how ``scikit-image`` gets
`used <https://github.com/scikit-image/scikit-image/issues/4375>`_.

If you'd like to package ``scikit-image`` for an as-yet-unsupported platform,
`reach out on GitHub <https://github.com/scikit-image/scikit-image/issues>`_.


Additional help
------------------------------------------------------------------------------

If you still have questions, reach out through

- our `forum on image.sc <https://forum.image.sc/tags/scikit-image>`_
- our `mailing list <https://mail.python.org/mailman3/lists/scikit-image.python.org/>`_
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
  in our `setup.py file <https://github.com/scikit-image/scikit-image/blob/master/setup.py#L212>`_.
- You've cloned the git repository into a directory called ``scikit-image``.
  You have set up the `upstream` remote to point to our repository and `origin`
  to point to your fork.


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
------------------------------------------------------------------------------

Once you've cloned your fork of the scikit-image repository,
you should set up a Python development environment tailored for scikit-image.
You may choose the environment manager of your choice.
Here we provide instructions for two popular environment managers:
``venv`` (pip based) and ``conda`` (Anaconda or Miniconda).

venv
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When using ``venv``, you may find the following bash commands useful:

.. code-block:: sh

  # Create a virtualenv named ``skimage-dev``
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
  # Install major development and runtime dependencies of scikit-image
  conda install `for i in requirements/{default,build,test}.txt; do echo -n " --file $i "; done`
  # Install scikit-image from source
  pip install -e . -vv
  # Test your installation
  pytest skimage

Updating the installation
------------------------------------------------------------------------------

When updating your installation, it is often necessary to recompile submodules
that have changed. Do so with the following commands:

.. code-block:: sh

    # Grab the latest source
    git checkout master
    git pull upstream master
    # Update the installation
    pip install -e . -vv

Testing
-------

``scikit-image`` has an extensive test suite that ensures correct
execution on your system.  The test suite must pass before a pull
request can be merged, and tests should be added to cover any
modifications to the code base.

We use the `pytest <https://docs.pytest.org/en/latest/>`__
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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Scikit-image tries to catch all warnings in its development builds to ensure
that crucial warnings from dependencies are not missed. This might cause
certain tests to fail if you are building scikit-image with versions of
dependencies that were not tested at the time of the release. To disable
failures on warnings, export the environment variable
``SKIMAGE_TEST_STRICT_WARNINGS`` with a value of `0` or `False` and run the
tests:

.. code-block:: sh

   export SKIMAGE_TEST_STRICT_WARNINGS=False
   pytest --pyargs skimage

Platform-specific notes
------------------------------------------------------------------------------

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


Full requirements list
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
    The ``pyamg`` module is used for the fast ``cg_mg`` mode of random
    walker segmentation.

* `Dask <https://dask.org/>`__
    The ``dask`` module is used to speed up certain functions.


.. include:: ../../requirements/optional.txt
  :literal:


**Extra Requirements**

These requirements have been included as a convenience, but are not widely
installable through PyPI on our supported platforms. As such, we keep them in
a separate list for more advanced members of our community to install.

* `imread <https://pythonhosted.org/imread/>`__
    Optional I/O plugin providing most standard `formats <https://pythonhosted.org//imread/formats.html>`__.

.. include:: ../../requirements/extras.txt
  :literal:

Help with contributor installation
------------------------------------------------------------------------------

See `Additional help <#additional-help>`_ above.

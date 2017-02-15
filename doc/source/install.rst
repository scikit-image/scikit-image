Installing scikit-image
=======================

Standard Python installation
----------------------------

From the shell, execute::

  pip install scikit-image

Conda-based distributions (Anaconda, Miniconda)
-----------------------------------------------

From the shell/command prompt, execute::

  conda install scikit-image

Windows
-------
``scikit-image`` comes pre-installed with several Python
distributions, including Anaconda_, `Enthought Canopy`_,
`Python(x,y)`_ and `WinPython`_.

.. _Anaconda: https://store.continuum.io/cshop/anaconda/
.. _Enthought Canopy: https://www.enthought.com/products/canopy/
.. _Python(x,y): http://code.google.com/p/pythonxy/wiki/Welcome
.. _WinPython: https://winpython.github.io/

If you are using the distribution from python.org_, you'll need to
manually download a few packages: numpy, scipy and scikit-image from
`Christoph Gohlke's`_ website. Python wheels are installed using::

  pip install SomePackage-1.0-py2.py3-none-any.whl

Debian and Ubuntu
-----------------
On Debian and Ubuntu install scikit-image with::

  sudo apt-get install python-skimage

Building from source
--------------------

Execute the following command from the shell::

  pip install scikit-image

Windows
```````
We are assuming that you have default Python environment already configured and you want to install
the scikit-image development version inside of it. For creating and working with Python virtual environments,
you can follow instructions on `venv`_ and `virtual environments`_.

Before installing scikit-image development version, uninstall the standard version of scikit-image using pip as::

  pip uninstall scikit-image

or using conda (for Anaconda users) as::

  conda uninstall scikit-image

Now clone scikit-image on your local computer::

  git clone https://github.com/scikit-image/scikit-image.git

Change the directory and build from source code::

  cd scikit-image
  python setup.py develop

If you experience the error ``Error:unable to find vcvarsall.bat`` it means that
your computer does not have recommended compilers for python. You can download and
install Windows compilers from `here`_  or correctly configure distutils
to use the C compiler by modifying (or create, if not existing) the configuration file
``distutils.cfg`` (located for example at ``C:\Python26\Lib\distutils\distutils.cfg``)
to contain::

  [build]
   compiler=mingw32

Once the build process is complete, run::

   pip install -U -e .

Make sure to give space after ``-e`` and add dot at the end. This will install scikit-image development version
and upgrade (or install) all the required dependencies. Otherwise, you can run the following command
to skip installation of dependencies::

   pip install -U[--no-deps] -e .

You can install or upgrade dependencies required for scikit-image anytime after installation using::

   pip install -r requirements.txt --upgrade

For more details on compiling in Windows, there is a lot of knowledge iterated
into the `setup of appveyor`_ (a continuous integration service).

.. _miniconda: http://conda.pydata.org/miniconda.html
.. _python.org: http://python.org/
.. _Christoph Gohlke's: http://www.lfd.uci.edu/~gohlke/pythonlibs/
.. _setup of appveyor: https://github.com/scikit-image/scikit-image/blob/master/appveyor.yml
.. _here: https://wiki.python.org/moin/WindowsCompilers#Microsoft_Visual_C.2B-.2B-_14.0_standalone:_Visual_C.2B-.2B-_Build_Tools_2015_.28x86.2C_x64.2C_ARM.29
.. _venv: https://docs.python.org/3/library/venv.html
.. _virtual environments: http://docs.python-guide.org/en/latest/dev/virtualenvs/

Debian and Ubuntu
`````````````````
If your distribution ships an outdated version, you may recompile from
source.  First install the dependencies::

  sudo apt-get install python-matplotlib python-numpy python-pil python-scipy

Get compilers::

  sudo apt-get install build-essential cython

Then run the pip installation command.


Running the development version
-------------------------------

Obtain the source from the git repository at
``http://github.com/scikit-image/scikit-image`` by running::

  git clone https://github.com/scikit-image/scikit-image.git

After unpacking, change into the source directory and execute:

  pip install -e .

To update::

  git pull  # Grab latest source
  python setup.py build_ext -i  # Compile any modified extensions


Building with bento
```````````````````

Alternatively, ``scikit-image`` can also be built using `bento
<http://cournape.github.io/Bento/>`__.  Bento depends on `WAF
<https://code.google.com/p/waf/>`__ for compilation.

Follow the `Bento installation instructions
<http://cournape.github.io/Bento/html/install.html>`__ and `download the WAF
source <http://code.google.com/p/waf/downloads/list>`__.

Tell Bento where to find WAF by setting the ``WAFDIR`` environment variable::

    export WAFDIR=<path/to/waf>

From the ``scikit-image`` source directory::

    bentomaker configure
    bentomaker build -j     # (add -i for in-place build)
    bentomaker install      # (when not building in-place)

Depending on file permissions, the install commands may need to be run as
sudo.

.. include:: ../../DEPENDS.txt

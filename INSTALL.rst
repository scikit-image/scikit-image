.. _installing-scikit-image:

Installing scikit-image
==============================================================================

How you install ``scikit-image`` depends on your needs and skill:

- Simplest solution:
  `a scientific Python distribution <#scientific-python-distributions>`_.\

- If you can install Python packages and work in virtual environments:

  - `pip <#install-via-pip>`_

  - `conda <#install-via-conda>`_

- Easy but has pitfalls: `system package manager (yum, apt-get,...) <#system-package-manager>`_

- `You're looking to contribute to scikit-image <#building-from-source>`_

Supported platforms
------------------------------------------------------------------------------

- Windows 64-bit on x86 processors
- Mac OS X on x86 processors
- Linux 64-bit on x86 processors

For information on other platforms, see `Other platforms <#other-platforms>`_.

Version check
------------------------------------------------------------------------------

To see whether ``scikit-image`` is already installed or to check if an install has
worked, run the following in a Python shell or Jupyter notebook:

.. code-block:: python

  import skimage
  print(skimage.__version__)

or, from the command line

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
`virtual environment \
<https://towardsdatascience.com/virtual-environments-104c62d48c54?gi=2532aa12906#ee81>`_
(any of
`several \
<https://stackoverflow.com/questions/41573587/what-is-the-difference-between-venv-pyvenv-pyenv-virtualenv-virtualenvwrappe>`_\
).

Nothing will stop you from using pip without a virtual environment,
but `it's a bad idea that will bite you later \
<https://en.wikipedia.org/wiki/Dependency_hell>`_.

To install the current ``scikit-image`` you'll need at least Python 3.6. If
your Python is older, pip will find the most recent compatible version.

.. code-block:: sh

  # Update pip
  python -m pip install -U pip
  # Install scikit-image
  python -m pip install -U scikit-image

To include a selection of other scientific python packages as well,
replace the last line with

.. code-block:: sh

    python -m pip install -U scikit-image[optional]


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


.. _system-package-manager:

Installing via the system package manager
------------------------------------------------------------------------------

Using a package manager (``yum``, ``apt-get``, etc.) to install ``scikit-image``
or other Python packages is not your best option:

- You're likely to get an older version.

- You'll probably want to make updates and add new packages outside
  the package manager, leaving you open to the same kind of
  dependency conflicts you see when using pip without a virtual environment.

- There's an added risk because operating systems use Python, so if you
  make system-wide Python changes (installing as root or using sudo),
  you can break the OS.

Building from source
------------------------------------------------------------------------------
Prerequisite: A local copy of the ``scikit-image`` git repo.

In the top directory, run

.. code-block:: sh

   pip install -e .

Note the final dot. ``-e`` installs ``scikit-image`` in editable
mode, meaning that ``import`` will pick up changes immediately.

If you change Cython files (or have never built them before), you first
will need to run:

.. code-block:: sh

   python setup.py build_ext -i

You'll need to install Cython if this returns a message like

.. code-block:: sh

   ModuleNotFoundError: No module named 'Cython'

.. _other-platforms:

Other platforms
------------------------------------------------------------------------------

We still support Windows 32-bit on x86 processors but urge switching
to Windows 64-bit.

Unsupported platforms include:

1. Linux on 32-bit x86 processors.
2. Linux on 32-bit on ARM processors (Raspberry Pi running Rapsbian):

   - While we do not officially support this distribution, we point users to
     `piwheels <https://wwww.piwheels.org>`_
     and their
     `scikit-image's specific page <https://www.piwheels.org/project/scikit-image/>`_.

   - You may need to install additional system dependencies listed for
     `imagecodecs <https://www.piwheels.org/project/imagecodecs/>`_.
     See
     `issue 4721 <https://github.com/scikit-image/scikit-image/issues/4721>`_.

3. Linux on 64-bit ARM processors (NVidia Jetson):

   - Follow the conversation on
     `Issue 4705 <https://github.com/scikit-image/scikit-image/issues/4705>`_.

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
`please open an issue on GitHub <https://github.com/scikit-image/scikit-image/issues>`_.


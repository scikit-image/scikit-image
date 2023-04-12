# Building with Meson

We are assuming that you have a default Python environment already configured on
your computer and that you intend to install `scikit-image` inside of it.

### Developer build

Install the required dependencies:

```
pip install -r requirements.txt
pip install -r requirements/build.txt
```

Then build `skimage`:

```
spin build
spin test
```

To run a specific test:

```
spin test -- skimage/io/tests/test_imageio.py
```

Or to try the new version in IPython:

```
pip install ipython
spin ipython
```

Run `spin --help` for more commands.

### Developer build (explicit)

**Install build tools:** `pip install -r requirements/build.txt`

**Generate ninja make files:** `meson build --prefix=$PWD/build`

**Compile:** `ninja -C build`

**Install:** `meson install -C build`

The installation step copies the necessary Python files into the build dir to form a
complete package.
Do not skip this step, or the package won't work.

To use the package, add it to your PYTHONPATH:

```
export PYTHONPATH=${PWD}/build/lib64/python3.10/site-packages
pytest --pyargs skimage
```

### pip install

The standard installation procedure via pip still works:

```
pip install --no-build-isolation .
```

Note, however, that `pip install -e .` (in-place developer install) does not!
See "Developer build" above.

### sdist and wheel

The Python `build` module calls Meson and ninja as necessary to
produce an sdist and a wheel:

```
python -m build --no-isolation
```

### Conda (Experimental)

.. warning::
   Combining `conda` and `pip` is not recommanded. Use with caution!

The recommanded versions of `scikit-image` dependencies are
unfortunatly not available in the main Conda channel. But you still
can get `scikit-image` installed in editable mode for development
using a combination of `conda`, `pip` and `meson` command lines:

- First, create a conda environment with available build and run dependencies
  ```
  conda create -n skimage-dev python=3.10 scipy networkx pywavelets pillow imageiomeson-python cython pythran
  conda activate skimage-dev
  ```
- then install skimage in editable mode
  ```
  cd SKIMAGE_SRC_PATH
  pip install -e . --config-settings editable-verbose=true
  ```
- reconfigure meson
  ```
  meson setup .mesonpy/editable/build --wipe
  ```

and you are done!

#### Testing with pytest

Testing in the above settings can be achieved using

- Activate the previously created development environment and install `pytest`
  ```
  conda activate skimage-dev
  conda install pytest
  ```
- use the `importlib` import mode when you run the tests:
  ```
  pytest --import-mode=importlib skimage/
  ```

## Notes

### Templated Cython files

The `skimage/morphology/skeletonize_3d.pyx.in` is converted into a pyx
file using Tempita. That pyx file appears in the _build_
directory, and can be compiled from there.

If that file had to import local `*.pyx` files (it does not) then the
build dependencies would need be set to ensure that the relevant pyx
files are copied into the build directory prior to compilation (see
`_cython_tree` in the SciPy Meson build files).

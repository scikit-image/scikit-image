# pip requirements files

## Index

- [default.txt](default.txt)
  Default requirements
- [docs.txt](docs.txt)
  Documentation requirements
- [optional.txt](optional.txt)
  Optional requirements. All of these are installable without a compiler through pypi.
- [extras.txt](extras.txt)
  Optional requirements that require a compiler to install.
- [test.txt](test.txt)
  Requirements for running test suite
- [build.txt](build.txt)
  Requirements for building from the source repository

## Examples

### Installing requirements

```bash
$ pip install -U -r requirements/default.txt
```

### Running the tests

```bash
$ pip install -U -r requirements/default.txt
$ pip install -U -r requirements/test.txt
```

## Justification for blocked versions

* Cython 0.28.2 was empircally found to fail tests while other patch-releases 0.28.x do not
* Cython 0.29.0 erroneously sets the `__path__` to `None`. See https://github.com/cython/cython/issues/2662
* Cython 0.29.18 fails due to a bad definition of M_PI. See https://github.com/cython/cython/issues/3622
* matplotlib 3.0.0 is not used because of a bug that collapses 3D axes (see https://github.com/scikit-image/scikit-image/pull/3474 and https://github.com/matplotlib/matplotlib/issues/12239).
* pillow 7.1.0 fails on png files, See https://github.com/scikit-image/scikit-image/issues/4548
* pillow 7.1.1 fails due to https://github.com/python-pillow/Pillow/issues/4518
* imread 0.7.2 fails due to bluid failure https://github.com/luispedro/imread/issues/36

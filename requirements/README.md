# pip requirements files

## Index

- [default.txt](default.txt)
  Default requirements
- [docs.txt](docs.txt)
  Documentation requirements
- [optional.txt](optional.txt)
  Optional requirements. All of these are installable without a compiler through pypi.
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

- Cython 0.28.2 was empircally found to fail tests while other patch-releases 0.28.x do not
- Cython 0.29.0 erroneously sets the `__path__` to `None`. See https://github.com/cython/cython/issues/2662
- Cython 0.29.18 fails due to a bad definition of M_PI. See https://github.com/cython/cython/issues/3622
- sphinx-gallery 0.8.0 is banned due to bug introduced on binder: https://github.com/scikit-image/scikit-image/pull/4959#issuecomment-687653537

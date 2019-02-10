# pip requirements files

## Index

- [default.txt](default.txt)
  Default requirements
- [docs.txt](docs.txt)
  Documentation requirements
- [optional.txt](optional.txt)
  Optional requirements
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

## Blacklist justification

* Cython 0.28.2 was empircally found to fail tests while other patch-releases 0.28.x do not
* Cython 0.29.0 erroneously sets the `__path__` to `None`. See https://github.com/cython/cython/issues/2662
* matplotlib 3.0.0 is not used because of a bug that collapses 3D axes (see https://github.com/scikit-image/scikit-image/pull/3474 and https://github.com/matplotlib/matplotlib/issues/12239). We expect this issue to be resolved in the next matplotlib release.

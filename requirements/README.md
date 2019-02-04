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

  * cython 0.28.2 was empircally found to fail tests.
  * cython 0.29.0 eroneously sets the the `__path__` to `none`. See https://github.com/cython/cython/issues/2662

# pip requirements files

## Index

- [default.txt](default.txt)(#default.txt-reasoning)
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

## [default.txt-reasoning]default.txt

matplotlib 3.0.0 is decided against due to this PR https://github.com/scikit-image/scikit-image/pull/3474. The marching cube example on the dev version is not displayed correctly because of a bug from matplotlib version 3.0.0 https://github.com/matplotlib/matplotlib/issues/12239. 

#!/usr/bin/env bash
# Fail on non-zero exit and echo the commands
set -ev
export PY=${TRAVIS_PYTHON_VERSION}

# Matplotlib settings - do not show figures during doc examples
if [[ $MINIMUM_REQUIREMENTS == 1 || $TRAVIS_OS_NAME == "osx" ]]; then
    MPL_DIR=$HOME/.matplotlib
else
    MPL_DIR=$HOME/.config/matplotlib
fi

mkdir -p $MPL_DIR
touch $MPL_DIR/matplotlibrc

if [[ $TRAVIS_OS_NAME == "osx" ]]; then
    echo 'backend : Template' > $MPL_DIR/matplotlibrc
fi

section "Test.with.min.requirements"
# Python 3.5 doesn't do ordered dictionaries.
# we need to set the hashseed to be common between processes
# so that pytest-xdist works correctly
# see: https://github.com/pytest-dev/pytest-xdist/issues/63
PYTHONHASHSEED=0 pytest -n 4 $TEST_ARGS skimage
section_end "Test.with.min.requirements"

section "Flake8.test"
flake8 --exit-zero --exclude=test_* skimage doc/examples viewer_examples
section_end "Flake8.test"

section "Tests.pytest"
# run tests. If running with optional dependencies, report coverage
if [[ "$OPTIONAL_DEPS" == "1" ]]; then
  export TEST_ARGS="${TEST_ARGS} --cov=skimage"
fi
# Show what's installed
pip list
PYTHONHASHSEED=0 pytest -n 4 ${TEST_ARGS} skimage
section_end "Tests.pytest"


section "Tests.examples"
# Run example applications
echo Build or run examples
if [[ "${BUILD_DOCS}" == "1" ]]; then
  # requirements/docs.txt fails on Travis OSX
  pip install --retries 3 -q -r ./requirements/docs.txt
  export SPHINXCACHE=${HOME}/.cache/sphinx; make html
elif [[ "${TEST_EXAMPLES}" != "0" ]]; then
  # OSX Can't install sphinx-gallery.
  # I think all it needs is scikit-learn from that requirements doc
  # to run the tests. See Issue #3084
  if [[ "${TRAVIS_OS_NAME}" == "osx" ]]; then
    pip install --retries 3 -q scikit-learn
  else
    pip install --retries 3 -q -r ./requirements/docs.txt
  fi
  cp $MPL_DIR/matplotlibrc $MPL_DIR/matplotlibrc_backup
  echo 'backend : Template' > $MPL_DIR/matplotlibrc
  # run these jobs in parallel,
  # xargs will return failed if one job fails
  ls -1 doc/examples/*/*.py | xargs -n 1 -P 4 python
  mv $MPL_DIR/matplotlibrc_backup $MPL_DIR/matplotlibrc
fi
section_end "Tests.examples"

set +ev

#!/usr/bin/env bash
# Fail on non-zero exit and echo the commands
set -ev
export PY=${TRAVIS_PYTHON_VERSION}

section "Flake8.test"
flake8 --exit-zero --exclude=test_* skimage doc/examples viewer_examples
section_end "Flake8.test"

section "Tests.pytest"

# Show what's installed
pip list
pytest ${TEST_ARGS} --pyargs skimage
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
  for f in doc/examples/*/*.py; do
    python "${f}"
    if [ $? -ne 0 ]; then
      exit 1
    fi
  done
  mv $MPL_DIR/matplotlibrc_backup $MPL_DIR/matplotlibrc
fi
section_end "Tests.examples"

set +ev

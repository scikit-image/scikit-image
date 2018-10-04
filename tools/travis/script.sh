#!/usr/bin/env bash
# Fail on non-zero exit and echo the commands
set -ev
export PY=${TRAVIS_PYTHON_VERSION}

mkdir -p $MPL_DIR
touch $MPL_DIR/matplotlibrc

if [[ $TRAVIS_OS_NAME == "osx" ]]; then
    echo 'backend : Template' > $MPL_DIR/matplotlibrc
fi

section "List.installed.dependencies"
pip list
tools/build_versions.py
section_end "List.installed.dependencies"

section "Test"
pytest $TEST_ARGS skimage
section_end "Test"

section "Flake8.test"
flake8 --exit-zero --exclude=test_* skimage doc/examples viewer_examples
section_end "Flake8.test"

section "Tests.examples"
# Run example applications
echo Build or run examples
pip install --retries 3 -q -r ./requirements/docs.txt
pip list
tools/build_versions.py
echo 'backend : Template' > $MPL_DIR/matplotlibrc
if [[ "${BUILD_DOCS}" == "1" ]]; then
  export SPHINXCACHE=${HOME}/.cache/sphinx; make html
elif [[ "${TEST_EXAMPLES}" != "0" ]]; then
  for f in doc/examples/*/*.py; do
    python "${f}"
    if [ $? -ne 0 ]; then
      exit 1
    fi
  done
fi
section_end "Tests.examples"

set +ev

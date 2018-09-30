#!/usr/bin/env bash
# Fail on non-zero exit and print commands
set -ex
export PY=${TRAVIS_PYTHON_VERSION}

section "Tests.flake8"
flake8 --exit-zero --exclude=test_*,six.py skimage doc/examples viewer_examples
section_end "Tests.flake8"


section "Tests.pytest"
# run tests. If running with optional dependencies, report coverage
if [[ "$OPTIONAL_DEPS" == "1" ]]; then
  export TEST_ARGS="${TEST_ARGS} --cov=skimage"
fi
# Show what's installed
pip list
pytest ${TEST_ARGS} skimage
section_end "Tests.pytest"


section "Tests.examples"
# Run example applications
echo Build or run examples
pip install --retries 3 -q -r ./requirements/docs.txt
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

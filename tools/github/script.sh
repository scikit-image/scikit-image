#!/usr/bin/env bash

# Options:
#
# BUILD_DOCS:
#   Whether or not to build the docs.
# TEST_EXAMPLES:
#   Whether or not to run the examples.
#   This is mutually exclusive with BUILD_DOCS=1, which
#   already runs the examples.

# Fail on non-zero exit and echo the commands
set -evx

TEST_ARGS="--doctest-plus --showlocals"

# Run the tests
(cd .. && pytest $TEST_ARGS --pyargs skimage)

# Optionally, build the docs or test examples
if [[ "${BUILD_DOCS}" == "1" ]]; then
  echo Build docs
  export SPHINXCACHE=${HOME}/.cache/sphinx; make -C doc html
elif [[ "${TEST_EXAMPLES}" == "1" ]]; then
  echo Test examples
  for f in doc/examples/*/*.py; do
    python "${f}"
    if [ $? -ne 0 ]; then
      exit 1
    fi
  done
fi

set +ev

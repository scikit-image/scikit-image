#!/usr/bin/env bash
# Fail on non-zero exit and echo the commands
set -ev

mkdir -p $MPL_DIR
touch $MPL_DIR/matplotlibrc


python -m pip list
tools/build_versions.py

# When installing from sdist
# We can't run it in the git directory since there is a folder called `skimage`
# in there. pytest will crawl that instead of the module we installed and want to test
(cd .. && pytest $TEST_ARGS --pyargs skimage)

flake8 --exit-zero --exclude=test_* skimage doc/examples viewer_examples

set +ev

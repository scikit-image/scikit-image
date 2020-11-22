#!/usr/bin/env bash
# Fail on non-zero exit and echo the commands
set -ev

mkdir -p $MPL_DIR
touch $MPL_DIR/matplotlibrc


section "List.installed.dependencies"
python -m pip list
tools/build_versions.py
section_end "List.installed.dependencies"

section "Test"
# When installing from sdist
# We can't run it in the git directory since there is a folder called `skimage`
# in there. pytest will crawl that instead of the module we installed and want to test
(cd .. && pytest $TEST_ARGS --pyargs skimage)
section_end "Test"

section "Flake8.test"
flake8 --exit-zero --exclude=test_* skimage doc/examples viewer_examples
section_end "Flake8.test"

set +ev

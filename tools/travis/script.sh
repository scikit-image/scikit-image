#!/usr/bin/env bash
# Fail on non-zero exit and print commands
set -ex
export PY=${TRAVIS_PYTHON_VERSION}

section "Tests.pytest"
pip list
pytest ${TEST_ARGS} --verbose skimage
section_end "Tests.pytest"


#!/usr/bin/env bash
set -ex

echo -en "travis_fold:start:test.min\r"
nosetests $TEST_ARGS skimage
echo -en "travis_fold:end:test.min\r"

echo -en "travis_fold:start:test.flake8\r"
flake8 --exit-zero --exclude=test_*,six.py skimage doc/examples viewer_examples
echo -en "travis_fold:end:test.flake8\r"

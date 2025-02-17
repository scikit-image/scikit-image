#!/usr/bin/env bash

# Options:
#
# PIP_FLAGS:
#   These options are passed to pip.

# Why does this script not have more options? Because we will be
# building our *wheels* with one set of dependencies, so that's what
# we'll be using in the CI as well.

# TODO: Remove special handling of free-threaded dependencies below.

set -ex

export PIP_DEFAULT_TIMEOUT=60
python -m pip install --upgrade pip

# Install build time requirements
python -m pip install $PIP_FLAGS -r requirements/build.txt

# TODO: delete when cython free-threaded wheels are available on PyPi
FREE_THREADED_BUILD="$(python -c"import sysconfig; print(bool(sysconfig.get_config_var('Py_GIL_DISABLED')))")"
if [[ $FREE_THREADED_BUILD == "True" ]]; then
    pip install --pre -i https://pypi.anaconda.org/scientific-python-nightly-wheels/simple cython numpy
fi

# Show what's installed
python -m pip list

set +ex

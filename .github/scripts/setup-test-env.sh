#!/usr/bin/env bash

# Options:
#
# MINIMUM_REQUIREMENTS:
#   Install the minimum versions of all requirements, instead of the latest.
#
# OPTIONAL_DEPS:
#   Install optional requirements.
#
# WITHOUT_POOCH:
#   Remove pooch from environment.
#
# PIP_FLAGS:
#   These options are passed to pip.

# TODO: Remove special handling of free-threaded dependencies below.

set -ex

export PIP_DEFAULT_TIMEOUT=60

if [[ $MINIMUM_REQUIREMENTS == 1 ]]; then
    for filename in requirements/*.txt; do
        sed -i 's/>=/==/g' "$filename"
    done
fi

# Combine requirement files for a more robust pip solve
# installing successively may update previously constrained dependencies
REQUIREMENT_FILES="-r requirements/default.txt -r requirements/test.txt -r requirements/asv.txt"
if [[ "${OPTIONAL_DEPS}" == "1" ]]; then
    REQUIREMENT_FILES="${REQUIREMENT_FILES} -r requirements/optional.txt"
fi

python -m pip install $PIP_FLAGS $REQUIREMENT_FILES

# TODO: delete when cython free-threaded wheels are available on PyPi
FREE_THREADED_BUILD="$(python -c"import sysconfig; print(bool(sysconfig.get_config_var('Py_GIL_DISABLED')))")"
if [[ $FREE_THREADED_BUILD == "True" ]]; then
    pip install --pre -i https://pypi.anaconda.org/scientific-python-nightly-wheels/simple cython
fi

if [[ ${WITHOUT_POOCH} == "1" ]]; then
  python -m pip uninstall pooch -y
fi

# Show what's installed
python -m pip list

set +ex

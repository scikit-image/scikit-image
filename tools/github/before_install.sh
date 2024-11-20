#!/usr/bin/env bash
set -ex

export PIP_DEFAULT_TIMEOUT=60

if [[ $MINIMUM_REQUIREMENTS == 1 ]]; then
    for filename in requirements/*.txt; do
        sed -i 's/>=/==/g' "$filename"
    done
fi

python -m pip install --upgrade pip

# TODO: delete when scipy, numpy, cython and pywavelets free-threaded wheels are available on PyPi
FREE_THREADED_BUILD="$(python -c"import sysconfig; print(bool(sysconfig.get_config_var('Py_GIL_DISABLED')))")"
if [[ $FREE_THREADED_BUILD == "True" ]]; then
    pip install --pre -i https://pypi.anaconda.org/scientific-python-nightly-wheels/simple cython numpy scipy pywavelets
fi

# Install build time requirements
python -m pip install $PIP_FLAGS -r requirements/build.txt

# Show what's installed
python -m pip list

set +ex

#!/usr/bin/env bash

# Options:
#
# MINIMUM_REQUIREMENTS:
#   Install the minimum versions of all requirements, instead of the latest.
#
# PIP_FLAGS:
#   These options are passed to pip.
#
# BUILD_DOCS / TEST_EXAMPLES:
#   Install documentation dependencies, and set up headless Matplotlib backend.
#
# OPTIONAL_DEPS:
#   Install optional requirements.
#
# WITHOUT_POOCH:
#   Remove pooch from environment.

# TODO: Remove special handling of free-threaded dependencies below.

set -ex

export PIP_DEFAULT_TIMEOUT=60

if [[ $MINIMUM_REQUIREMENTS == 1 ]]; then
    for filename in requirements/*.txt; do
        sed -i 's/>=/==/g' "$filename"
    done
fi

python -m pip install --upgrade pip

# Combine requirement files for a more robust pip solve
# installing successively may update previously constrained dependencies
REQUIREMENT_FILES="-r requirements/default.txt -r requirements/test.txt"
if [[ "${OPTIONAL_DEPS}" == "1" ]]; then
    REQUIREMENT_FILES="${REQUIREMENT_FILES} -r requirements/optional.txt"
fi

python -m pip install $PIP_FLAGS $REQUIREMENT_FILES


# TODO: delete when scipy, numpy, cython and pywavelets free-threaded wheels are available on PyPi
FREE_THREADED_BUILD="$(python -c"import sysconfig; print(bool(sysconfig.get_config_var('Py_GIL_DISABLED')))")"
if [[ $FREE_THREADED_BUILD == "True" ]]; then
    pip install --pre -i https://pypi.anaconda.org/scientific-python-nightly-wheels/simple cython numpy scipy pywavelets
fi

# Install build time requirements
python -m pip install $PIP_FLAGS -r requirements/build.txt

# Prepare for building the docs (and test examples)
if [[ "${BUILD_DOCS}" == "1" ]] || [[ "${TEST_EXAMPLES}" == "1" ]]; then
  echo "Build or run examples"
  # Use previous installed requirements as well, otherwise installing
  # successively may update previously constraint dependencies
  python -m pip install $PIP_FLAGS $REQUIREMENT_FILES -r ./requirements/docs.txt
  python -m pip list

  export MPL_DIR
  MPL_DIR=$(python -c 'import matplotlib; print(matplotlib.get_configdir())')
  if [[ -n "${MPL_DIR}" ]]; then
    mkdir -p "${MPL_DIR}"
    touch "${MPL_DIR}/matplotlibrc"
    echo 'backend : Template' > "${MPL_DIR}/matplotlibrc"
  fi
fi

if [[ ${WITHOUT_POOCH} == "1" ]]; then
  # remove pooch (previously installed via requirements/test.txt)
  python -m pip uninstall pooch -y
fi

# Show what's installed
python -m pip list

set +ex

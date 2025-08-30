#!/usr/bin/env bash

# Options:
#
# PIP_FLAGS:
#   These options are passed to pip.

set -ex

export PIP_DEFAULT_TIMEOUT=60

echo "Installing docs / examples dependencies..."
# We are fine with updating previous requirements
# for the doc build (for now)
python -m pip install $PIP_FLAGS --group docs

echo "Set matplotlib backend to Template..."
export MPL_DIR
MPL_DIR=$(python -c 'import matplotlib; print(matplotlib.get_configdir())')
if [[ -n "${MPL_DIR}" ]]; then
  mkdir -p "${MPL_DIR}"
  echo 'backend : Template' > "${MPL_DIR}/matplotlibrc"
fi

# Show what's installed
python -m pip list

set +ex

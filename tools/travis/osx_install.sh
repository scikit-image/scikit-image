#!/bin/bash
set -ex

# Set up virtualenv on OSX
git clone --depth 1 --branch devel https://github.com/matthew-brett/multibuild ~/multibuild
source ~/multibuild/osx_utils.sh
get_macpython_environment $MB_PYTHON_VERSION ~/venv

export PATH="$PATH:/Library/TeX/texbin"

set +ex

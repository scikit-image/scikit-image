#!/bin/bash
set -ex

brew update
brew install ccache
brew tap homebrew/homebrew-cask
brew cask install basictex

export PATH="$PATH:/Library/TeX/texbin"
sudo tlmgr --verify-repo=none update --self
sudo tlmgr --verify-repo=none install ucs dvipng anyfontsize

# Set up virtualenv on OSX
git clone --depth 1 --branch devel https://github.com/matthew-brett/multibuild ~/multibuild
source ~/multibuild/osx_utils.sh
get_macpython_environment $MB_PYTHON_VERSION ~/venv

# libpng 1.6.32 has a bug in reading PNG
# that seems to be the default library that is installed
# https://github.com/ImageMagick/ImageMagick/issues/747#issuecomment-328521685
# It seems libpng causes gdal and postgis to get updated.
# These cause conflicts in the update process and don't seem necessary
brew uninstall postgis gdal
brew upgrade libpng

set +ex

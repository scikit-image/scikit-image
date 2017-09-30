#!/bin/bash
brew update
brew install ccache
brew tap caskroom/cask
brew cask install basictex

export PATH="$PATH:/Library/TeX/texbin"
sudo tlmgr update --self
sudo tlmgr install ucs dvipng anyfontsize

# Set up virtualenv on OSX
git clone https://github.com/matthew-brett/multibuild ~/multibuild
source ~/multibuild/osx_utils.sh
get_macpython_environment $TRAVIS_PYTHON_VERSION ~/venv

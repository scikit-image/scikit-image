#!/bin/bash
brew update
brew install ccache
brew tap caskroom/cask
brew cask install basictex

export PATH="$PATH:/Library/TeX/texbin"
sudo tlmgr update --self
sudo tlmgr install ucs dvipng anyfontsize

git clone https://github.com/MacPython/terryfy.git ~/terryfy
source ~/terryfy/travis_tools.sh
get_python_environment macpython $TRAVIS_PYTHON_VERSION ~/macpython_venv
source ~/macpython_venv/bin/activate
pip install virtualenv


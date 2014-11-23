#!/usr/bin/env bash
set -ex

sh -e /etc/init.d/xvfb start

pip install wheel flake8 coveralls nose

# install system tk
sudo apt-get install python-tk

# on Python 2.7, use the system versions of numpy, scipy, and matplotlib
# and the minimum version of cython and networkx
if [[ $TRAVIS_PYTHON_VERSION == 2.7* ]]; then
    sudo apt-get install python-scipy python-matplotlib
    sed -i 's/cython>=/cython==/g' requirements.txt
    sed -i 's/networkx>=/networkx==/g' requirements.txt
fi

# on Python 3.2, use matplotlib 1.3.1
if [[ $TRAVIS_PYTHON_VERSION == 2.7* ]]; then
    sed -i 's/matplotlib>=*.*.*/matplotlib==1.3.1/g' requirements.txt
fi

pip install $WHEELHOUSE -r requirements.txt

# clean up disk space
sudo apt-get clean
sudo rm -r /tmp/*

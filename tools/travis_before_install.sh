#!/usr/bin/env bash

export WHEELHOUSE="--no-index --find-links=http://travis-wheels.scikit-image.org/"


repip () {
    travis_retry pip $@
}


# on Python 2.7, use the system versions of numpy, scipy, and matplotlib
# and the minimum version of cython and networkx
if [[ $TRAVIS_PYTHON_VERSION == 2.7* ]]; then
    virtualenv -p python --system-site-packages ~/venv
    sudo apt-get install python-scipy python-matplotlib
    sed -i 's/cython>=/cython==/g' requirements.txt
    sed -i 's/networkx>=/networkx==/g' requirements.txt
else
    virtualenv -p python --system-site-packages ~/venv
fi

source ~/venv/bin/activate
repip install wheel flake8 coveralls nose

# install system tk for matplotlib
sudo apt-get install python-tk


# on Python 3.2, use matplotlib 1.3.1
if [[ $TRAVIS_PYTHON_VERSION == 2.7* ]]; then
    sed -i 's/matplotlib>=*.*.*/matplotlib==1.3.1/g' requirements.txt
fi

repip install $WHEELHOUSE -r requirements.txt

# clean up disk space
sudo apt-get clean
sudo rm -rf /tmp/*


fold_start () {
    echo -en "travis_fold:start:$1\r"
}

fold_end () {
    echo -en "travis_fold:end:$1\r"
}

export -f fold_start
export -f fold_end
export -f repip

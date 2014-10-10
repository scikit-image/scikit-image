#!/usr/bin/env bash
set -ex

export DISPLAY=:99.0
export PYTHONWARNINGS="all"
WHEELHOUSE="--no-index --find-links=http://wheels.scikit-image.org/"

tools/header.py "Run all tests with minimum dependencies"
nosetests --exe -v skimage

tools/header.py "Pep8 and Flake tests"
flake8 --exit-zero --exclude=test_*,six.py skimage doc/examples viewer_examples

tools/header.py "Install optional dependencies"

# Install Qt and then update the Matplotlib settings
if [[ $TRAVIS_PYTHON_VERSION == 2.7* ]]; then
    sudo apt-get install -q python-qt4
    MPL_QT_API=PyQt4
    export QT_API=pyqt

else
    sudo apt-get install -q libqt4-dev
    pip install PySide $WHEELHOUSE
    python ~/virtualenv/python${TRAVIS_PYTHON_VERSION}/bin/pyside_postinstall.py -install
    MPL_QT_API=PySide
    export QT_API=pyside
fi

# Matplotlib settings - must be after we install Pyside
MPL_DIR=$HOME/.config/matplotlib
mkdir -p $MPL_DIR
touch $MPL_DIR/matplotlibrc
echo 'backend : Agg' > $MPL_DIR/matplotlibrc
echo 'backend.qt4 : '$MPL_QT_API >> $MPL_DIR/matplotlibrc

# imread does NOT support py3.2
if [[ $TRAVIS_PYTHON_VERSION != 3.2 ]]; then
    sudo apt-get install -q libtiff4-dev libwebp-dev libpng12-dev xcftools
    pip install imread
fi

# TODO: update when SimpleITK become available on py34 or hopefully pip
if [[ $TRAVIS_PYTHON_VERSION != 3.4 ]]; then
    easy_install SimpleITK
fi

sudo apt-get install libfreeimage3
pip install astropy

if [[ $TRAVIS_PYTHON_VERSION == 2.* ]]; then
    pip install pyamg
fi

tools/header.py "Run doc examples"
for f in doc/examples/*.py; do
    python "$f";
    if [ $? -ne 0 ]; then
        exit 1;
    fi
done

for f in doc/examples/applications/*.py; do
    python "$f";
    if [ $? -ne 0 ]; then
        exit 1;
    fi
done

tools/header.py "Run tests with all dependencies"
# run tests again with optional dependencies to get more coverage
if [[ $TRAVIS_PYTHON_VERSION == 3.3 ]]; then
    export TEST_ARGS="--with-cov --cover-package skimage"
else
    export TEST_ARGS=""
fi
nosetests --exe -v --with-doctest $TEST_ARGS


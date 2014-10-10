#!/usr/bin/sh
./header.py "Run all tests with minimum dependencies"
nosetests --exe -v skimage

./header.py "Pep8 and Flake tests"
flake8 --exit-zero --exclude=test_*,six.py skimage doc/examples viewer_examples

./header.py "Install optional dependencies"

# Install Qt and then update the Matplotlib settings
if [[ $TRAVIS_PYTHON_VERSION == 2.7* ]]; then
    sudo apt-get install -q python-qt4
    SCI_QT_API=PyQt4

else
    sudo apt-get install -q libqt4-dev
     travis_retry pip install PySide $WHEELHOUSE
     python ~/virtualenv/python${TRAVIS_PYTHON_VERSION}/bin/pyside_postinstall.py -install
     SCI_QT_API=PySide
fi

# Matplotlib settings - must be after we install Pyside
MPL_DIR=$HOME/.config/matplotlib
mkdir -p $MPL_DIR
touch $MPL_DIR/matplotlibrc
echo 'backend : Agg' > $MPL_DIR/matplotlibrc
echo 'backend.qt4 : '$SCI_QT_API >> $MPL_DIR/matplotlibrc

# imread does NOT support py3.2
if [[  $TRAVIS_PYTHON_VERSION != 3.2 ]]; then
    sudo apt-get install -q libtiff4-dev libwebp-dev libpng12-dev xcftools
    travis_retry pip install imread
fi

# TODO: update when SimpleITK become available on py34 or hopefully pip
if [[  $TRAVIS_PYTHON_VERSION != 3.4 ]]; then
    travis_retry easy_install SimpleITK
fi

travis_retry sudo apt-get install libfreeimage3
travis_retry pip install astropy

if [[ $TRAVIS_PYTHON_VERSION == 2.* ]]; then
    travis_retry pip install pyamg
fi

./header.py "Run doc examples"
PYTHONPATH=$(pwd):$PYTHONPATH
for f in doc/examples/*.py; do python "$f"; if [ $? -ne 0 ]; then exit 1; fi done
for f in doc/examples/applications/*.py; do python "$f"; if [ $? -ne 0 ]; then exit 1; fi done

./header.py "Run tests with all dependencies"
# run tests again with optional dependencies to get more coverage
# measure coverage on py3.3
if [[ $TRAVIS_PYTHON_VERSION == 3.3 ]]; then
    nosetests --exe -v --with-doctest --with-cov --cover-package skimage
else
    nosetests --exe -v --with-doctest skimage
fi


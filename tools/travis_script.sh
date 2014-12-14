#!/usr/bin/env bash
set -ex

fold_start "script.setup"
sh -e /etc/init.d/xvfb start
export DISPLAY=:99.0
PYTHONWARNINGS="all"
TEST_ARGS="--exe --ignore-files=^_test -v --with-doctest --ignore-files=^setup.py$"
fold_end "script.setup"


fold_start "test.min"
tools/header.py "Test with min requirements"

nosetests $TEST_ARGS skimage
fold_end "test.min"


fold_start "test.flake8"
tools/header.py "Flake8 test"

flake8 --exit-zero --exclude=test_*,six.py skimage doc/examples viewer_examples
fold_end "test.flake8"



fold_start "install.all"
tools/header.py "Install optional dependencies"

# Install Qt and then update the Matplotlib settings
if [[ $TRAVIS_PYTHON_VERSION == 2.7* ]]; then
    sudo apt-get install -q python-qt4

    # http://stackoverflow.com/a/9716100
    LIBS=( PyQt4 sip.so )

    VAR=( $(which -a python$TRAVIS_PYTHON_VERSION) )

    GET_PYTHON_LIB_CMD="from distutils.sysconfig import get_python_lib; print (get_python_lib())"
    LIB_VIRTUALENV_PATH=$(python -c "$GET_PYTHON_LIB_CMD")
    LIB_SYSTEM_PATH=$(${VAR[-1]} -c "$GET_PYTHON_LIB_CMD")

    for LIB in ${LIBS[@]}
    do
        sudo ln -sf $LIB_SYSTEM_PATH/$LIB $LIB_VIRTUALENV_PATH/$LIB
    done

else
    sudo apt-get install -q libqt4-dev
    repip install -q PySide $WHEELHOUSE
    python ~/venv/bin/pyside_postinstall.py -install
fi

# imread does NOT support py3.2
if [[ $TRAVIS_PYTHON_VERSION != 3.2 ]]; then
    sudo apt-get install -q libtiff4-dev libwebp-dev libpng12-dev xcftools
    repip install -q imread
fi

# Install SimpleITK from wheelhouse if available (not 3.2 or 3.4)
if [[ $TRAVIS_PYTHON_VERSION =~ 3\.[24] ]]; then
    echo "SimpleITK unavailable on $TRAVIS_PYTHON_VERSION"
else
    repip install -q SimpleITK $WHEELHOUSE
fi

sudo apt-get install -q libfreeimage3
repip install -q astropy $WHEELHOUSE

if [[ $TRAVIS_PYTHON_VERSION == 2.* ]]; then
    repip install -q pyamg
fi

repip install -q tifffile

fold_end "install.all"


fold_start "doc.examples"
tools/header.py "Run doc examples"

for f in doc/examples/*.py; do
    python "$f"
    if [ $? -ne 0 ]; then
        exit 1
    fi
done

fold_end "doc.examples"


fold_start "doc.applications"
tools/header.py "Run doc applications"

for f in doc/examples/applications/*.py; do
    python "$f"
    if [ $? -ne 0 ]; then
        exit 1
    fi
done

# Now configure Matplotlib to use Qt4
echo 'backend: Agg' > $MPL_DIR/matplotlibrc
echo 'backend.qt4 : '$MPL_QT_API >> $MPL_DIR/matplotlibrc

fold_end "doc.applications"


fold_start "test.all"
tools/header.py "Test with optional dependencies"

# run tests again with optional dependencies to get more coverage
if [[ $TRAVIS_PYTHON_VERSION == 3.3 ]]; then
    TEST_ARGS="$TEST_ARGS --with-cov --cover-package skimage"
fi
nosetests $TEST_ARGS

fold_end "test.all"

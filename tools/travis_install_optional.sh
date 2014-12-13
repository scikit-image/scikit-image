#!/usr/bin/env bash
set -ex

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
        sudo ln -s $LIB_SYSTEM_PATH/$LIB $LIB_VIRTUALENV_PATH/$LIB
    done

else
    sudo apt-get install -q libqt4-dev
    pip install -q PySide $WHEELHOUSE
    python ~/venv/bin/pyside_postinstall.py -install
fi

# imread does NOT support py3.2
if [[ $TRAVIS_PYTHON_VERSION != 3.2 ]]; then
    sudo apt-get install -q libtiff4-dev libwebp-dev libpng12-dev xcftools
    pip install -q imread
fi

# Install SimpleITK from wheelhouse if available
pip install -q SimpleITK $WHEELHOUSE; true

sudo apt-get install -q libfreeimage3
pip install -q astropy $WHEELHOUSE

if [[ $TRAVIS_PYTHON_VERSION == 2.* ]]; then
    pip install -q pyamg
fi

pip install -q tifffile

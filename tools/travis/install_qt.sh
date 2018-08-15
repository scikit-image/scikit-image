#!/usr/bin/env bash
set -ex

if [[ "${TRAVIS_OS_NAME}" == "osx" ]]; then
    echo "backend : Template" > $MPL_DIR/matplotlibrc
fi
# Now configure Matplotlib to use Qt4
if [[ "${QT}" == "PyQt4" ]]; then
    # only do this for python 2.7
    # http://stackoverflow.com/a/9716100
    LIBS=( PyQt4 sip.so )

    VAR=( $(which -a python$PY) )

    GET_PYTHON_LIB_CMD="from distutils.sysconfig import get_python_lib; print (get_python_lib())"
    LIB_VIRTUALENV_PATH=$(python -c "$GET_PYTHON_LIB_CMD")
    LIB_SYSTEM_PATH=$(${VAR[-1]} -c "$GET_PYTHON_LIB_CMD")

    for LIB in ${LIBS[@]}
    do
        ln -sf $LIB_SYSTEM_PATH/$LIB $LIB_VIRTUALENV_PATH/$LIB
    done

    MPL_QT_API=PyQt4
    export QT_API=pyqt
elif [[ "${QT}" == "PySide" ]]; then
    python ~/venv/bin/pyside_postinstall.py -install
    MPL_QT_API=PySide
    export QT_API=pyside
# Now configure Matplotlib to use Qt5
elif [[ "${QT}" == "PyQt5" ]]; then
    pip install --retries 3 -q $PIP_FLAGS pyqt5
    MPL_QT_API=PyQt5
    export QT_API=pyqt5
elif [[ "${QT}" == "PySide2" ]]; then
    pip install--retries 3 -q $PIP_FLAGS pyside2
    MPL_QT_API=PySide2
    export QT_API=pyside2
else
    echo 'backend: Template' > $MPL_DIR/matplotlibrc
fi
if [[ "${QT}" == "PyQt4" || "${QT}" == "PySide" ]]; then
    echo 'backend: Qt4Agg' > $MPL_DIR/matplotlibrc
    echo 'backend.qt4 : '$MPL_QT_API >> $MPL_DIR/matplotlibrc
elif [[ "${QT}" == "PyQt5" || "${QT}" == "PySide2" ]]; then
    # Is this correct for PySide2?
    echo 'backend: Qt5Agg' > $MPL_DIR/matplotlibrc
    echo 'backend.qt5 : '$MPL_QT_API >> $MPL_DIR/matplotlibrc
fi

set -ex

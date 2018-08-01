#!/usr/bin/env bash
set -ev

if [[ "${TRAVIS_OS_NAME}" == "osx" ]]; then
    echo "backend : Template" > $MPL_DIR/matplotlibrc
fi
# Now configure Matplotlib to use Qt5
if [[ "${QT}" == "PyQt5" ]]; then
    # Matplotlib 2.2.2 doesn't support pyqt5 5.11 until this commit
    # https://github.com/matplotlib/matplotlib/commit/260e6d611afae9adaac528c28a4879097c357b74#diff-025508b6963efcfa97ba27389f2d2b86
    # becomes released
    pip install --retries 3 -q $PIP_FLAGS 'pyqt5<5.11'
    MPL_QT_API=PyQt5
    export QT_API=pyqt5
elif [[ "${QT}" == "PySide2" ]]; then
    pip install--retries 3 -q $PIP_FLAGS pyside2
    MPL_QT_API=PySide2
    export QT_API=pyside2
else
    echo 'backend: Template' > $MPL_DIR/matplotlibrc
fi
if [[ "${QT}" == "PyQt5" || "${QT}" == "PySide2" ]]; then
    # Is this correct for PySide2?
    echo 'backend: Qt5Agg' > $MPL_DIR/matplotlibrc
    echo 'backend.qt5 : '$MPL_QT_API >> $MPL_DIR/matplotlibrc
fi

set +ev

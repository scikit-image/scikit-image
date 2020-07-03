#!/usr/bin/env bash
set -ev

if [[ "${TRAVIS_OS_NAME}" == "osx" ]]; then
    echo "backend : Template" > $MPL_DIR/matplotlibrc
fi
# Now configure Matplotlib to use Qt5
if [[ "${QT}" == "PyQt5" ]]; then
    if [[ $MINIMUM_REQUIREMENTS == "1" ]]; then
        # PyQt 5.11 changed how they ship SIP
        # Which conflicts with how matplotlib detects the
        # presence of PyQt before MPL 2.2.3
        pip install --retries 3 -q $PIP_FLAGS "pyqt5<5.11"
    elif [[ `lsb_release  -r -s` == "14.04" ]]; then
        # Apparently Qt 5.12 is only supported by ubuntu 16.04
        # https://github.com/scikit-image/scikit-image/pull/3744#issuecomment-463450663
        pip install --retries 3 -q $PIP_FLAGS "pyqt5<5.12"
    else
        pip install --retries 3 -q $PIP_FLAGS "pyqt5!=5.15.0"
    fi
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

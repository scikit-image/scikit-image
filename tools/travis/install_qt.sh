#!/usr/bin/env bash
# Now configure Matplotlib to use Qt4
if [[ $QT == "PyQt4" ]]; then
    pip install pyqt4
    MPL_QT_API=PyQt4
    export QT_API=pyqt
elif [[ $QT == "PyQt4" ]]; then
    pip install pyqt5
    MPL_QT_API=PyQt5
    export QT_API=pyqt5
elif [[ $QT == "PySide" ]]; then
    pip install pyside
    MPL_QT_API=PySide
    export QT_API=pyside
elif [[ $QT == "PySide2" ]]; then
    pip install pyside2
    MPL_QT_API=PySide2
    export QT_API=pyside2
fi
if [[ $QT == "PyQt4" || $QT == "PySide" ]]; then
    echo 'backend: Qt4Agg' > $MPL_DIR/matplotlibrc
    echo 'backend.qt4 : '$MPL_QT_API >> $MPL_DIR/matplotlibrc
elif [[ $QT == "PyQt5" || $QT == "PySide2" ]]; then
    # Is this correct for PySide2?
    echo 'backend: Qt5Agg' > $MPL_DIR/matplotlibrc
    echo 'backend.qt5 : '$MPL_QT_API >> $MPL_DIR/matplotlibrc
fi

try:
    from matplotlib.backends.qt_compat import QtGui, QtCore, QtWidgets
except ImportError:
    from matplotlib.backends.qt4_compat import QtGui, QtCore
    QtWidgets = QtGui

Qt = QtCore.Qt
Signal = QtCore.Signal

has_qt = True

try:
    from matplotlib.backends.qt_compat import QtGui, QtCore, QtWidgets
except ImportError:
    try:
        from matplotlib.backends.qt4_compat import QtGui, QtCore
        QtWidgets = QtGui
    except ImportError:
        # Mock objects
        class QtGui(object):
            QMainWindow = object
            QDialog = object
            QWidget = object

        class QtCore(object):
            Qt = object

            def Signal(self, *args, **kwargs): 
                pass

        QtWidgets = QtGui

        has_qt = False

Qt = QtCore.Qt
Signal = QtCore.Signal

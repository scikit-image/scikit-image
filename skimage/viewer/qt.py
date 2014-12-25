has_qt = True

try:
    from matplotlib.backends.qt_compat import QtGui, QtCore, QtWidgets
except ImportError:
    try:
        from matplotlib.backends.qt4_compat import QtGui, QtCore
        QtWidgets = QtGui
    except ImportError:
        # Mock objects
        class QtGui_cls(object):
            QMainWindow = object
            QDialog = object
            QWidget = object

        class QtCore_cls(object):
            class Qt(object):
                 TopDockWidgetArea = None
                 BottomDockWidgetArea = None
                 LeftDockWidgetArea = None
                 RightDockWidgetArea = None

            def Signal(self, *args, **kwargs): 
                pass

        QtGui = QtWidgets = QtGui_cls()
        QtCore = QtCore_cls()

        has_qt = False

Qt = QtCore.Qt
Signal = QtCore.Signal

try:
    from matplotlib.backends.qt_compat import QtGui, QtCore, QtWidgets, QT_RC_MAJOR_VERSION as has_qt
except ImportError:
    try:
        from matplotlib.backends.qt4_compat import QtGui, QtCore
        QtWidgets = QtGui
        has_qt = 4
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

if has_qt == 5:
    from matplotlib.backends.backend_qt5 import FigureManagerQT
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
elif has_qt == 4:
    from matplotlib.backends.backend_qt4 import FigureManagerQT
    from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg
else:
    assert not has_qt, 'Unsupported Qt version {0}'.format(has_qt)
    FigureManagerQT = object
    FigureCanvasQTAgg = object

Qt = QtCore.Qt
Signal = QtCore.Signal

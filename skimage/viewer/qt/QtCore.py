from . import qt_api

if qt_api == 'pyside':
    from PySide.QtCore import *
elif qt_api == 'pyqt':
    from PyQt4.QtCore import *
    # Use pyside names for signals and slots
    Signal = pyqtSignal
    Slot = pyqtSlot
else:
    # Mock objects for buildbot (which doesn't have Qt, but imports viewer).
    class Qt(object):
        TopDockWidgetArea = None
        BottomDockWidgetArea = None
        LeftDockWidgetArea = None
        RightDockWidgetArea = None

    def Signal(*args, **kwargs):
        pass

    def Slot(*args, **kwargs):
        pass

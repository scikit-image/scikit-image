from . import qt_api

if qt_api == 'pyside':
    from PySide.QtCore import *
elif qt_api == 'pyqt':
    from PyQt4.QtCore import *
else:
    # Mock objects for buildbot (which doesn't have Qt, but imports viewer).
    class Qt(object):
        TopDockWidgetArea = None
        BottomDockWidgetArea = None
        LeftDockWidgetArea = None
        RightDockWidgetArea = None

    def pyqtSignal(*args, **kwargs):
        pass

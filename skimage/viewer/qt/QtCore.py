from . import qt_api

if qt_api == 'pyside':
    from PySide.QtCore import *
elif qt_api == 'pyqt':
    from PyQt4.QtCore import *
else:
    # Mock objects
    Qt = None
    pyqtSignal = None

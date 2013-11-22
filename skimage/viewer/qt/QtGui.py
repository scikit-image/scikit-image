from . import qt_api

if qt_api == 'pyside':
    from PySide.QtGui import *
elif qt_api == 'pyqt':
    from PyQt4.QtGui import *
else:
    # Mock objects
    QMainWindow = object
    QDialog = object
    QWidget = object

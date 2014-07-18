# -*- coding: utf-8 -*-
from numpy.testing.decorators import skipif

try:
    from skimage.viewer.qt import qt_api, QtCore, QtGui
    from skimage.viewer import utils
    from skimage.viewer.utils import dialogs
except ImportError:
    qt_api = None


@skipif(qt_api is None)
def test_event_loop():
    utils.init_qtapp()
    timer = QtCore.QTimer()
    timer.singleShot(10, QtGui.QApplication.quit)
    utils.start_qtapp()


@skipif(qt_api is None)
def test_format_filename():
    fname = dialogs._format_filename(('apple', 2))
    assert fname == 'apple'
    fname = dialogs._format_filename('')
    assert fname is None


@skipif(qt_api is None)
def test_open_file_dialog():
    utils.init_qtapp()
    timer = QtCore.QTimer()
    timer.singleShot(10, QtGui.QApplication.quit)
    filename = dialogs.open_file_dialog()
    assert filename is None


@skipif(qt_api is None)
def test_save_file_dialog():
    utils.init_qtapp()
    timer = QtCore.QTimer()
    timer.singleShot(10, QtGui.QApplication.quit)
    filename = dialogs.save_file_dialog()
    assert filename is None

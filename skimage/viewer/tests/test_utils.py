# -*- coding: utf-8 -*-
try:
    from skimage.viewer import utils
    from skimage.viewer.utils import dialogs
    from skimage.viewer.qt import QtCore, QtGui
except ImportError:
    QtCore = None
from numpy.testing.decorators import skipif


@skipif(QtCore is None)
def test_event_loop():
    utils.init_qtapp()
    timer = QtCore.QTimer()
    timer.singleShot(10, QtGui.QApplication.quit)
    utils.start_qtapp()


@skipif(QtCore is None)
def test_format_filename():
    fname = dialogs._format_filename(('apple', 2))
    assert fname == 'apple'
    fname = dialogs._format_filename('')
    assert fname is None


@skipif(QtCore is None)
def test_open_file_dialog():
    utils.init_qtapp()
    timer = QtCore.QTimer()
    timer.singleShot(100, lambda: QtGui.QApplication.quit())
    filename = dialogs.open_file_dialog()
    assert filename is None


@skipif(QtCore is None)
def test_save_file_dialog():
    utils.init_qtapp()
    timer = QtCore.QTimer()
    timer.singleShot(100, lambda: QtGui.QApplication.quit())
    filename = dialogs.save_file_dialog()
    assert filename is None

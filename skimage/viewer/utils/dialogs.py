import os

from ..qt import QtGui


def open_file_dialog(default_format='png'):
    """Return user-selected file path."""
    filename = str(QtGui.QFileDialog.getOpenFileName())
    if len(filename) == 0:
        return None
    return filename


def save_file_dialog(default_format='png'):
    """Return user-selected file path."""
    filename = QtGui.QFileDialog.getSaveFileName()
    # Handle discrepancy between PyQt4 and PySide APIs.
    if isinstance(filename, tuple):
        filename = filename[0]
    filename = str(filename)

    if len(filename) == 0:
        return None
    #TODO: io plugins should assign default image formats
    basename, ext = os.path.splitext(filename)
    if not ext:
        filename = '%s.%s' % (filename, default_format)
    return filename

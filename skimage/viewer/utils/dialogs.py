import os

try:
    from PyQt4 import QtGui
except ImportError:
    print("Could not import PyQt4 -- skimage.viewer not available.")


def open_file_dialog(default_format='png'):
    """Return user-selected file path."""
    filename = str(QtGui.QFileDialog.getOpenFileName())
    if len(filename) == 0:
        return None
    return filename


def save_file_dialog(default_format='png'):
    """Return user-selected file path."""
    filename = str(QtGui.QFileDialog.getSaveFileName())
    if len(filename) == 0:
        return None
    #TODO: io plugins should assign default image formats
    basename, ext = os.path.splitext(filename)
    if not ext:
        filename = '%s.%s' % (filename, default_format)
    return filename

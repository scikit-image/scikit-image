from textwrap import dedent

from ..qt import QtGui
from ..qt import QtCore

import numpy as np

import skimage
from skimage import io
from .core import BaseWidget
from ..utils import dialogs


__all__ = ['OKCancelButtons', 'SaveButtons']


class OKCancelButtons(BaseWidget):
    """Buttons that close the parent plugin.

    OK will replace the original image with the current (filtered) image.
    Cancel will just close the plugin.
    """
    def __init__(self, button_width=80):
        name = 'OK/Cancel'
        super(OKCancelButtons, self).__init__(name)

        self.ok = QtGui.QPushButton('OK')
        self.ok.clicked.connect(self.update_original_image)
        self.ok.setMaximumWidth(button_width)
        self.ok.setFocusPolicy(QtCore.Qt.NoFocus)
        self.cancel = QtGui.QPushButton('Cancel')
        self.cancel.clicked.connect(self.close_plugin)
        self.cancel.setMaximumWidth(button_width)
        self.cancel.setFocusPolicy(QtCore.Qt.NoFocus)

        self.layout = QtGui.QHBoxLayout(self)
        self.layout.addStretch()
        self.layout.addWidget(self.cancel)
        self.layout.addWidget(self.ok)

    def update_original_image(self):
        image = self.plugin.image_viewer.image
        self.plugin.image_viewer.original_image = image
        self.plugin.close()

    def close_plugin(self):
        # Image viewer will restore original image on close.
        self.plugin.close()


class SaveButtons(BaseWidget):
    """Buttons to save image to io.stack or to a file."""

    def __init__(self, name='Save to:', default_format='png'):
        super(SaveButtons, self).__init__(name)

        self.default_format = default_format

        self.name_label = QtGui.QLabel()
        self.name_label.setText(name)

        self.save_file = QtGui.QPushButton('File')
        self.save_file.clicked.connect(self.save_to_file)
        self.save_file.setFocusPolicy(QtCore.Qt.NoFocus)
        self.save_stack = QtGui.QPushButton('Stack')
        self.save_stack.clicked.connect(self.save_to_stack)
        self.save_stack.setFocusPolicy(QtCore.Qt.NoFocus)

        self.layout = QtGui.QHBoxLayout(self)
        self.layout.addWidget(self.name_label)
        self.layout.addWidget(self.save_stack)
        self.layout.addWidget(self.save_file)

    def save_to_stack(self):
        image = self.plugin.filtered_image.copy()
        io.push(image)

        msg = dedent('''\
            The image has been pushed to the io stack.
            Use io.pop() to retrieve the most recently pushed image.
            NOTE: The io stack only works in interactive sessions.''')
        notify(msg)

    def save_to_file(self, filename=None):
        if filename is None:
            filename = dialogs.save_file_dialog()
        if filename is None:
            return
        image = self.plugin.filtered_image
        if image.dtype == np.bool:
            #TODO: This check/conversion should probably be in `imsave`.
            image = skimage.img_as_ubyte(image)
        io.imsave(filename, image)


def notify(msg):
    msglabel = QtGui.QLabel(msg)
    dialog = QtGui.QDialog()
    ok = QtGui.QPushButton('OK', dialog)
    ok.clicked.connect(dialog.accept)
    ok.setDefault(True)
    dialog.layout = QtGui.QGridLayout(dialog)
    dialog.layout.addWidget(msglabel, 0, 0, 1, 3)
    dialog.layout.addWidget(ok, 1, 1)
    dialog.exec_()

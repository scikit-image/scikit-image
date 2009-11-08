from util import prepare_for_display, window_manager, GuiLockError
from textwrap import dedent
import numpy as np
import sys

try:
    # We try to aquire the gui lock first or else the gui import might
    # trample another GUI's PyOS_InputHook.
    window_manager.acquire('qt')

except GuiLockError, gle:
    print gle

else:
    try:
        from PyQt4.QtGui import (QApplication, QMainWindow, QImage, QPixmap,
                                 QLabel, QWidget)
        from PyQt4 import QtCore, QtGui

    except ImportError:
        print 'PyQT4 libraries not installed.  Plugin not loaded.'
        window_manager._release('qt')

    else:

        app = None

        class ImageLabel(QLabel):
            def __init__(self, parent, arr):
                QLabel.__init__(self)

                # we need to hold a reference to
                # arr because QImage doesn't copy the data
                # and the buffer must be alive as long
                # as the image is alive.
                self.arr = arr

                # we also need to pass in the row-stride to
                # the constructor, because we can't guarantee
                # that every row of the numpy data is
                # 4-byte aligned. Which Qt would require
                # if we didnt pass the stride.
                self.img = QImage(arr.data, arr.shape[1], arr.shape[0],
                                  arr.strides[0], QImage.Format_RGB888)
                self.pm = QPixmap.fromImage(self.img)
                self.setPixmap(self.pm)
                self.setAlignment(QtCore.Qt.AlignTop)
                self.setMinimumSize(100, 100)

            def resizeEvent(self, evt):
                width = self.width()
                pm = QPixmap.fromImage(self.img)
                self.pm = pm.scaledToWidth(width)
                self.setPixmap(self.pm)


        class ImageWindow(QMainWindow):
            def __init__(self, arr, mgr):
                QMainWindow.__init__(self)
                self.setWindowTitle('scikits.image')
                self.mgr = mgr
                self.main_widget = QWidget()
                self.layout = QtGui.QGridLayout(self.main_widget)
                self.setCentralWidget(self.main_widget)

                self.label = ImageLabel(self, arr)
                self.layout.addWidget(self.label, 0, 0)
                self.layout.addLayout
                self.mgr.add_window(self)
                self.main_widget.show()

            def closeEvent(self, event):
                # Allow window to be destroyed by removing any
                # references to it
                self.mgr.remove_window(self)


        def imshow(arr, fancy=False):
            global app
            if not app:
                app = QApplication([])

            arr = prepare_for_display(arr)

            if not fancy:
                iw = ImageWindow(arr, window_manager)
            else:
                from scivi import SciviImageWindow
                iw = SciviImageWindow(arr, window_manager)

            iw.show()


        def _app_show():
            global app
            if app and window_manager.has_windows():
                app.exec_()
            else:
                print 'No images to show.  See `imshow`.'


        def imsave(filename, img):
            # we can support for other than 3D uint8 here...
            img = prepare_for_display(img)
            qimg = QImage(img.data, img.shape[1], img.shape[0],
                                  img.strides[0], QImage.Format_RGB888)
            saved = qimg.save(filename)
            if not saved:
                msg = dedent(
                    '''The image was not saved. Allowable file formats
                    for the QT imsave plugin are:
                    BMP, JPG, JPEG, PNG, PPM, TIFF, XBM, XPM''')
                raise RuntimeError(msg)

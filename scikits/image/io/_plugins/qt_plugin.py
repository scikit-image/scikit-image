from util import prepare_for_display, window_manager, GuiLockError

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
                                 QLabel, QWidget, QVBoxLayout)

    except ImportError:
        print 'PyQT4 libraries not installed.  Plugin not loaded.'
        window_manager._release('qt')

    else:

        app = None

        class LabelImage(QLabel):
            def __init__(self, parent, arr):
                QLabel.__init__(self)
                self.img = QImage(arr.data, arr.shape[1], arr.shape[0],
                                  arr.strides[0], QImage.Format_RGB888)
                self.pm = QPixmap.fromImage(self.img)
                self.setPixmap(self.pm)


        class ImageWindow(QMainWindow):
            def __init__(self, arr, mgr):
                QMainWindow.__init__(self)
                self.mgr = mgr
                self.label = LabelImage(self, arr)
                self.setCentralWidget(self.label)
                self.mgr.add_window(self)

            def closeEvent(self, event):
                # Allow window to be destroyed by removing any
                # references to it
                self.mgr.remove_window(self)


        class FancyImageWindow(ImageWindow):
            def __init__(self, arr, mgr):
                ImageWindow.__init__(self, arr, mgr)

                # we need to hold a reference to arr,
                # if we want to access the data later,
                # because QImage does not copy the data.
                self.arr = arr

                self.statusBar().showMessage('X: Y: ')
                self.label.setScaledContents(True)
                self.label.setMouseTracking(True)
                self.label.mouseMoveEvent = self.label_mouseMoveEvent

            def scale_mouse_pos(self, x, y):
                width = self.label.width()
                height = self.label.height()
                x_frac = 1. * x / width
                y_frac = 1. * y / height
                width = self.arr.shape[1]
                height = self.arr.shape[0]
                new_x = int(width * x_frac)
                new_y = int(height * y_frac)
                return(new_x, new_y)

            def label_mouseMoveEvent(self, evt):
                x = evt.x()
                y = evt.y()
                x, y = self.scale_mouse_pos(x, y)
                msg = 'X: %d, Y: %d  ' % (x, y)
                R = self.arr[y,x,0]
                G = self.arr[y,x,1]
                B = self.arr[y,x,2]
                msg += 'R: %s, G:, %s, B: %s' % (R, G, B)
                self.statusBar().showMessage(msg)


        def imshow(arr, fancy=False):
            global app

            if not app:
                app = QApplication([])

            arr = prepare_for_display(arr)

            if not fancy:
                iw = ImageWindow(arr, window_manager)
            else:
                iw = FancyImageWindow(arr, window_manager)

            iw.show()

        def _app_show():
            global app
            if app and window_manager.has_windows():
                app.exec_()
            else:
                print 'No images to show.  See `imshow`.'

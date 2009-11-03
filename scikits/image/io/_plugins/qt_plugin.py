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
                self.parent = parent
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

            def mouseMoveEvent(self, evt):
                self.parent.label_mouseMoveEvent(evt)


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

            def label_mouseMoveEvent(self, evt):
                pass


        class FancyImageWindow(ImageWindow):
            def __init__(self, arr, mgr):
                ImageWindow.__init__(self, arr, mgr)
                self.arr = arr

                # for image manipulation
                self.arrfloat = np.asarray(arr, dtype=np.float64)
                self.arruint8 = arr.copy()

                self.statusBar().showMessage('X: Y: ')
                self.label.setScaledContents(True)
                self.label.setMouseTracking(True)


            def keyPressEvent(self, evt):
                self.arrfloat[:,:,0] *= 1.1
                np.clip(self.arrfloat, 0, 255, self.arrfloat)
                self.arr[:] = self.arrfloat[:]

                pm = QPixmap.fromImage(self.label.img)
                self.label.setPixmap(pm)
                print 'heard'

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

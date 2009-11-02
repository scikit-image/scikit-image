import plugin
from util import prepare_for_display, window_manager, GuiLockError

import numpy as np
import sys


try:
    # we try to aquire the gui lock first
    # or else the gui import might trample another
    # gui's pyos_inputhook.
    window_manager.acquire('qt')

except GuiLockError, gle:
    print gle

else:
    try:
        from PyQt4.QtGui import (QApplication, QMainWindow, QImage, QPixmap,
                                 QLabel)

    except ImportError:
        print 'pyqt4 libraries not installed.'
        print 'plugin not loaded'
        window_manager._release('qt')

    else:

        app = None

        class ImageWindow(QMainWindow):
            def __init__(self, arr, mgr):
                QMainWindow.__init__(self)
                self.mgr = mgr
                img = QImage(arr.data, arr.shape[1], arr.shape[0],
                             QImage.Format_RGB888)
                pm = QPixmap.fromImage(img)

                label = QLabel()
                label.setPixmap(pm)
                label.show()

                self.label = label
                self.setCentralWidget(self.label)
                self.mgr.add_window(self)

            def closeEvent(self, event):
                # Allow window to be destroyed by removing any
                # references to it
                self.mgr.remove_window(self)

        def qt_imshow(arr, block=True):
            global app

            if not app:
                app = QApplication([])

            arr = prepare_for_display(arr)

            iw = ImageWindow(arr, window_manager)
            iw.show()

        def qt_show():
            global app
            if app and window_manager.has_images():
                app.exec_()
            else:
                print 'no images to show'

        plugin.register('qt', show=qt_imshow, appshow=qt_show)




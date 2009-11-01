import plugin
from util import prepare_for_display

import numpy as np
import sys

app = None
windows = []

try:
    from PyQt4.QtGui import (QApplication, QMainWindow, QImage, QPixmap,
                             QLabel)
except ImportError:
    pass
else:

    class ImageWindow(QMainWindow):
        def __init__(self, arr):
            QMainWindow.__init__(self)

            img = QImage(arr.data, arr.shape[1], arr.shape[0],
                         QImage.Format_RGB888)
            pm = QPixmap.fromImage(img)

            label = QLabel()
            label.setPixmap(pm)
            label.show()

            self.label = label
            self.setCentralWidget(self.label)

        def closeEvent(self, event):
            # Allow window to be destroyed by removing any
            # references to it
            windows.remove(self)

    def show(arr, block=True):
        global app

        if not '-qt4thread' in sys.argv and app is None:
            app = QApplication([])

        arr = prepare_for_display(arr)

        iw = ImageWindow(arr)
        iw.show()

        # Keep track of window so that it doesn't get destroyed
        windows.append(iw)

        if app and block:
            app.exec_()

    plugin.register('qt', show=show)

if __name__ == "__main__":
    import scikits.image.io as io

    io.plugin.use('qt', 'show')

    img = np.empty((200, 200, 3), dtype=np.uint8)
    img[:50, :50, 0] = 100
    img[25:100, 25:100, 1] = 200
    img[:, :, 2] = 155
    io.imshow(img, block=False)
    io.imshow(img)

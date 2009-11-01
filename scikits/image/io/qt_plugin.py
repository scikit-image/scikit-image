import plugin
import numpy as np
import sys

app = [None]
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
        arr = np.ascontiguousarray(arr.astype(np.uint8))
        if arr.ndim != 3:
            raise ValueError("Qt only displays colour images.")

        if arr.shape[-1] == 4:
            raise ValueError("Alpha channels not yet supported.")

        if not '-qt4thread' in sys.argv and app[0] is None:
            app[0] = QApplication([])

        iw = ImageWindow(arr)
        iw.show()

        # Keep track of window so that it doesn't get destroyed
        windows.append(iw)

        if app[0] and block:
            app[0].exec_()

    plugin.register('qt', show=show)

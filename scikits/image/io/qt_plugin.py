import plugin
import numpy as np

try:
    from PyQt4.QtGui import QApplication, QImage, QPixmap, QLabel
except ImportError:
    have_qt = False
else:
    have_qt = True

def show(arr):
    arr = np.ascontiguousarray(arr.astype(np.uint8))
    if arr.ndim != 3:
        raise ValueError("Qt only displays colour images.")

    if arr.shape[-1] == 4:
        raise ValueError("Alpha channels not yet supported.")

    img = QImage(arr.data, arr.shape[1], arr.shape[0], QImage.Format_RGB888)
    pm = QPixmap.fromImage(img)

    app = QApplication([])

    label = QLabel()
    label.setPixmap(pm)
    label.show()

    app.exec_()

if have_qt:
    plugin.register('qt', show=show)

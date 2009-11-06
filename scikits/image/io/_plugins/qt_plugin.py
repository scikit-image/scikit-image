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
                                 QLabel, QWidget, QVBoxLayout, QSlider,
                                 QPainter, QColor, QFrame)
        from PyQt4 import QtCore, QtGui
        from q_color_mixer import MixerPanel
        from q_histogram import QuadHistogram

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
                self.main_widget = QWidget()
                self.layout = QtGui.QGridLayout(self.main_widget)
                self.setCentralWidget(self.main_widget)

                self.label = LabelImage(self, arr)
                self.layout.addWidget(self.label, 0, 0)
                self.mgr.add_window(self)
                self.main_widget.show()

            def closeEvent(self, event):
                # Allow window to be destroyed by removing any
                # references to it
                self.mgr.remove_window(self)

            def label_mouseMoveEvent(self, evt):
                pass


        class RGBHSVDisplay(QWidget):
            def __init__(self):
                QWidget.__init__(self)
                self.posx_label = QLabel('X-pos:')
                self.posx_value = QLabel()
                self.posy_label = QLabel('Y-pos:')
                self.posy_value = QLabel()
                self.r_label = QLabel('R:')
                self.r_value = QLabel()
                self.g_label = QLabel('G:')
                self.g_value = QLabel()
                self.b_label = QLabel('B:')
                self.b_value = QLabel()
                self.h_label = QLabel('H:')
                self.h_value = QLabel()
                self.s_label = QLabel('S:')
                self.s_value = QLabel()
                self.v_label = QLabel('V:')
                self.v_value = QLabel()

                self.layout = QtGui.QGridLayout(self)
                self.layout.addWidget(self.posx_label, 0, 0)
                self.layout.addWidget(self.posx_value, 0, 1)
                self.layout.addWidget(self.posy_label, 1, 0)
                self.layout.addWidget(self.posy_value, 1, 1)
                self.layout.addWidget(self.r_label, 0, 2)
                self.layout.addWidget(self.r_value, 0, 3)
                self.layout.addWidget(self.g_label, 1, 2)
                self.layout.addWidget(self.g_value, 1, 3)
                self.layout.addWidget(self.b_label, 2, 2)
                self.layout.addWidget(self.b_value, 2, 3)
                self.layout.addWidget(self.h_label, 0, 4)
                self.layout.addWidget(self.h_value, 0, 5)
                self.layout.addWidget(self.s_label, 1, 4)
                self.layout.addWidget(self.s_value, 1, 5)
                self.layout.addWidget(self.v_label, 2, 4)
                self.layout.addWidget(self.v_value, 2, 5)

            def update_vals(self, data):
                xpos, ypos, r, g, b, h, s, v = data
                self.posx_value.setText(str(xpos)[:5])
                self.posy_value.setText(str(ypos)[:5])
                self.r_value.setText(str(r)[:5])
                self.g_value.setText(str(g)[:5])
                self.b_value.setText(str(b)[:5])
                self.h_value.setText(str(h)[:5])
                self.s_value.setText(str(s)[:5])
                self.v_value.setText(str(v)[:5])


        class FancyImageWindow(ImageWindow):
            def __init__(self, arr, mgr):
                ImageWindow.__init__(self, arr, mgr)
                self.arr = arr

                self.label.setScaledContents(True)
                self.label.setMouseTracking(True)
                self.label.setMinimumSize(QtCore.QSize(100, 100))

                self.mixer_panel = MixerPanel(self.arr)
                self.layout.addWidget(self.mixer_panel, 0, 2)
                self.mixer_panel.show()
                self.mixer_panel.set_callback(self.refresh_image)

                self.rgbv_hist = QuadHistogram(self.arr)
                self.layout.addWidget(self.rgbv_hist, 0, 1)
                self.rgbv_hist.show()

                self.rgb_hsv_disp = RGBHSVDisplay()
                self.layout.addWidget(self.rgb_hsv_disp, 1, 0)
                self.rgb_hsv_disp.show()

                self.layout.setColumnStretch(0, 1)
                self.layout.setRowStretch(0, 1)

                self.save_file = QtGui.QPushButton('Save to File')
                self.save_variable = QtGui.QPushButton('Save to Variable')
                self.save_file.show()
                self.save_variable.show()

                self.layout.addWidget(self.save_variable, 1, 1)
                self.layout.addWidget(self.save_file, 1, 2)


            def update_histograms(self):
                self.rgbv_hist.update_hists(self.arr)

            def refresh_image(self):
                pm = QPixmap.fromImage(self.label.img)
                self.label.setPixmap(pm)
                self.update_histograms()

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
                r = self.arr[y,x,0]
                g = self.arr[y,x,1]
                b = self.arr[y,x,2]
                h, s, v = self.mixer_panel.mixer.rgb_2_hsv_pixel(r, g, b)
                self.rgb_hsv_disp.update_vals((x, y, r, g, b, h, s, v))


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

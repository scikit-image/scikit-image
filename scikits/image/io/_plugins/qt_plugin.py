from util import prepare_for_display, window_manager, GuiLockError, ColorMixer

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
                                 QLabel, QWidget, QVBoxLayout, QSlider)
        from PyQt4 import QtCore, QtGui

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
                self.layout = QtGui.QHBoxLayout(self.main_widget)
                self.setCentralWidget(self.main_widget)

                self.label = LabelImage(self, arr)
                self.layout.addWidget(self.label)
                self.mgr.add_window(self)
                self.main_widget.show()

            def closeEvent(self, event):
                # Allow window to be destroyed by removing any
                # references to it
                self.mgr.remove_window(self)

            def label_mouseMoveEvent(self, evt):
                pass


        class IntelligentSlider(QSlider):
            ''' A slider that adds a 'name' attribute and calls a callback
            with 'name' as an argument to the registerd callback.

            This allows you to create large groups of sliders in a loop,
            but still keep track of the individual events'''

            def __init__(self, name, callback, conv_fac):
                QSlider.__init__(self)
                self.name = name
                self.callback = callback
                self.conv_fac = conv_fac

                self.name_label = QLabel()
                self.name_label.setText(self.name)
                self.name_label.setAlignment(QtCore.Qt.AlignCenter)

                self.value_label = QLabel()
                self.value_label.setText('')
                self.value_label.setAlignment(QtCore.Qt.AlignCenter)

            # bind this to the valueChanged signal of the slider
            def i_changed(self, val):
                val = self.conv_val()
                self.value_label.setText(str(val)[:4])
                self.callback(self.name, val)

            def conv_val(self):
                return self.value() * self.conv_fac


        class NSliderBlock(QWidget):
            '''Creates a block of n sliders with ranges
            specified as a list of tuples. The fourth entry
            in the tuple will be used as a dictionary key
            so you can access the slider later.
            So the tuple should be (min, max, initial, name)

            The callback is the function to be called
            when a slider value changes. The callback will be
            called with the following arguments (name, value).

            You can get a hook to a specific slider using
            NSliderBlock.sliders[name]

            '''

            def __init__(self, n, ranges_labels, callback):
                QWidget.__init__(self)

                if len(ranges_labels) != n:
                    raise ValueError('not enough or too many ranges supplied')

                self.callback = callback

                # each key will give you
                self.sliders = {}
                self.layout = QtGui.QGridLayout(self)

                for i in range(n):
                    params = ranges_labels[i]
                    if len(params) != 5:
                        raise ValueError('Tuples must be length 4')

                    name = params[3]
                    conv_fac = params[4]

                    slider = IntelligentSlider(name, self.callback, conv_fac)
                    slider.setMinimum(params[0])
                    slider.setMaximum(params[1])
                    slider.setValue(params[2])
                    slider.valueChanged.connect(slider.i_changed)


                    self.sliders[name] = slider

                    self.layout.addWidget(slider.name_label, 0, i)
                    self.layout.addWidget(slider, 1, i, QtCore.Qt.AlignCenter)
                    self.layout.addWidget(slider.value_label, 2, i)

                    self.layout.setColumnMinimumWidth(i, 50)

            def set_sliders(self, vals):
                # vals should a dict to of slider names and set vals
                if len(vals) != len(self.sliders):
                    raise ValueError('Wrong number of values')

                for key, value in vals.iteritems():
                    self.sliders[key].setValue(value)

        class MixerPanel(QWidget):
            '''A color mixer to hook up to an image.
            You pass the image you the panel to operate on
            and it operates on that image in place. You also
            pass a callback to be called to trigger a refresh.
            This callback is called every time the mixer modifies
            your image.'''
            def __init__(self, img, callback):
                QWidget.__init__(self)

                self.img = img
                self.update = callback
                self.mixer = ColorMixer(self.img)

                #---------------------------------------------------------------
                # ComboBox
                #---------------------------------------------------------------

                self.combo_box_entries = ['RGB Color', 'HSV Color',
                                          'Brightness', 'Contrast']
                self.combo_box = QtGui.QComboBox()
                for entry in self.combo_box_entries:
                    self.combo_box.addItem(entry)
                self.combo_box.currentIndexChanged.connect(self.combo_box_changed)

                #---------------------------------------------------------------
                # RGB color sliders
                #---------------------------------------------------------------

                # radio buttons
                self.rgb_add = QtGui.QRadioButton('Additive')
                self.rgb_mul = QtGui.QRadioButton('Multiplicative')
                self.rgb_mul.toggled.connect(self.rgb_radio_changed)
                self.rgb_add.toggled.connect(self.rgb_radio_changed)

                # additive sliders
                self.rgb_add_sliders = NSliderBlock(3, [(-255, 255, 0, 'R', 1),
                                                        (-255, 255, 0, 'G', 1),
                                                        (-255, 255, 0, 'B', 1)],
                                                    self.rgb_add_changed)

                # multiplicative sliders
                self.rgb_mul_sliders = NSliderBlock(3, [(0, 1000, 500, 'R', .002),
                                                        (0, 1000, 500, 'G', .002),
                                                        (0, 1000, 500, 'B', .002)],
                                                    self.rgb_mul_changed)

                # layout
                self.rgb_widget = QWidget()
                self.rgb_widget.layout = QtGui.QGridLayout(self.rgb_widget)
                self.rgb_widget.layout.addWidget(self.rgb_add, 0, 0)
                self.rgb_widget.layout.addWidget(self.rgb_mul, 1, 0)
                self.rgb_widget.layout.addWidget(self.rgb_add_sliders, 2, 0)
                self.rgb_widget.layout.addWidget(self.rgb_mul_sliders, 2, 0)

                #---------------------------------------------------------------
                # Brightness sliders
                #---------------------------------------------------------------

                # sliders
                self.bright_sliders = NSliderBlock(2, [(-255, 255, 0, '+', 1),
                                                    (0, 1000, 500, 'x', 0.002)],
                                                   self.bright_changed)

                # layout
                self.bright_widget = QWidget()
                self.bright_widget.layout = QtGui.QGridLayout(self.bright_widget)
                self.bright_widget.layout.addWidget(self.bright_sliders, 0, 0)

                #---------------------------------------------------------------
                # Buttons
                #---------------------------------------------------------------
                self.commit_button = QtGui.QPushButton('Commit')
                self.commit_button.clicked.connect(self.commit_changes)
                self.revert_button = QtGui.QPushButton('Revert')
                self.revert_button.clicked.connect(self.revert_changes)

                #---------------------------------------------------------------
                # Mixer Layout
                #---------------------------------------------------------------
                self.layout = QtGui.QGridLayout(self)
                self.layout.addWidget(self.combo_box, 0, 0)
                self.layout.addWidget(self.rgb_widget, 1, 0)
                self.layout.addWidget(self.bright_widget, 1, 0)
                self.layout.addWidget(self.commit_button, 2, 0)
                self.layout.addWidget(self.revert_button, 3, 0)

                #---------------------------------------------------------------
                # Initialization
                #---------------------------------------------------------------

                self.combo_box.setCurrentIndex(0)
                self.hide_sliders()
                self.rgb_widget.show()
                self.rgb_add.setChecked(True)

            def rgb_add_changed(self, name, val):
                if not self.rgb_add.isChecked():
                    return
                if name == 'R':
                    self.mixer.add(self.mixer.RED, val)
                elif name == 'G':
                    self.mixer.add(self.mixer.GREEN, val)
                elif name == 'B':
                    self.mixer.add(self.mixer.BLUE, val)
                else:
                    return
                self.update()

            def rgb_mul_changed(self, name, val):
                if not self.rgb_mul.isChecked():
                    return
                if name == 'R':
                    self.mixer.multiply(self.mixer.RED, val)
                elif name == 'G':
                    self.mixer.multiply(self.mixer.GREEN, val)
                elif name == 'B':
                    self.mixer.multiply(self.mixer.BLUE, val)
                else:
                    return
                self.update()

            def bright_changed(self, name, val):
                # doesnt matter which slider changed we need both
                # values
                factor = self.bright_sliders.sliders['x'].conv_val()
                offset = self.bright_sliders.sliders['+'].conv_val()
                self.mixer.brightness(offset, factor)
                self.update()


            def reset_sliders(self):
                self.rgb_add_sliders.set_sliders({'R': 0, 'G': 0, 'B': 0})
                self.rgb_mul_sliders.set_sliders({'R': 500, 'G': 500, 'B': 500})
                self.bright_sliders.set_sliders({'+': 0, 'x': 500})

            def combo_box_changed(self, index):
                self.reset_sliders()
                self.mixer.set_to_stateimg()
                self.update()
                combo_box_map={0: self.show_rgb, 1: self.show_hsv,
                               2: self.show_bright, 3: self.show_contrast}
                combo_box_map[index]()

            def hide_sliders(self):
                self.rgb_widget.hide()
                self.bright_widget.hide()

            def rgb_radio_changed(self):
                if self.rgb_add.isChecked():
                    self.rgb_add_sliders.show()
                    self.rgb_mul_sliders.hide()
                elif self.rgb_mul.isChecked():
                    self.rgb_mul_sliders.show()
                    self.rgb_add_sliders.hide()
                else:
                    pass

                self.reset_sliders()
                self.mixer.set_to_stateimg()
                self.update()

            def show_rgb(self):
                self.hide_sliders()
                self.rgb_widget.show()

            def show_hsv(self):
                self.hide_sliders()

            def show_bright(self):
                self.hide_sliders()
                self.bright_widget.show()

            def show_contrast(self):
                self.hide_sliders()

            def commit_changes(self):
                self.mixer.commit_changes()
                self.update()

            def revert_changes(self):
                self.mixer.revert()
                self.reset_sliders()
                self.update()



        class FancyImageWindow(ImageWindow):
            def __init__(self, arr, mgr):
                ImageWindow.__init__(self, arr, mgr)
                self.arr = arr

                self.statusBar().showMessage('X: Y: ')
                self.label.setScaledContents(True)
                self.label.setMouseTracking(True)

                self.mixer_panel = MixerPanel(self.arr, self.refresh_image)
                self.layout.addWidget(self.mixer_panel)
                self.mixer_panel.show()

            def refresh_image(self):
                pm = QPixmap.fromImage(self.label.img)
                self.label.setPixmap(pm)

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

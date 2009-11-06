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
                                 QLabel, QWidget, QVBoxLayout, QSlider,
                                 QPainter, QColor, QFrame)
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
                    self.layout.addWidget(slider, 1, i, QtCore.Qt.AlignHCenter)
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
                # HSV sliders
                #---------------------------------------------------------------

                # radio buttons
                self.hsv_add = QtGui.QRadioButton('Additive')
                self.hsv_mul = QtGui.QRadioButton('Multiplicative')
                self.hsv_mul.toggled.connect(self.hsv_radio_changed)
                self.hsv_add.toggled.connect(self.hsv_radio_changed)

                # additive sliders
                self.hsv_add_sliders = NSliderBlock(3, [(-180, 180, 0, 'H', 1),
                                                        (-100, 100, 0, 'S', .01),
                                                        (-100, 100, 0, 'V', .01)],
                                                    self.hsv_add_changed)

                # multiplicative sliders
                self.hsv_mul_sliders = NSliderBlock(3, [(-180, 180, 0, 'H', 1),
                                                        (0, 1000, 500, 'S', .002),
                                                        (0, 1000, 500, 'V', .002)],
                                                    self.hsv_mul_changed)

                # layout
                self.hsv_widget = QWidget()
                self.hsv_widget.layout = QtGui.QGridLayout(self.hsv_widget)
                self.hsv_widget.layout.addWidget(self.hsv_add, 0, 0)
                self.hsv_widget.layout.addWidget(self.hsv_mul, 1, 0)
                self.hsv_widget.layout.addWidget(self.hsv_add_sliders, 2, 0)
                self.hsv_widget.layout.addWidget(self.hsv_mul_sliders, 2, 0)

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
                self.layout.addWidget(self.hsv_widget, 1, 0)
                self.layout.addWidget(self.bright_widget, 1, 0)
                self.layout.addWidget(self.commit_button, 2, 0)
                self.layout.addWidget(self.revert_button, 3, 0)

                #---------------------------------------------------------------
                # Initialization
                #---------------------------------------------------------------

                self.combo_box.setCurrentIndex(0)
                self.hide_sliders()
                self.rgb_widget.show()
                self.rgb_mul.setChecked(True)
                self.hsv_mul.setChecked(True)

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

            def hsv_add_changed(self, name, val):
                if not self.hsv_add.isChecked():
                    return
                h = self.hsv_add_sliders.sliders['H'].conv_val()
                s = self.hsv_add_sliders.sliders['S'].conv_val()
                v = self.hsv_add_sliders.sliders['V'].conv_val()
                self.mixer.hsv_add(h, s, v)
                self.update()

            def hsv_mul_changed(self, name, val):
                if not self.hsv_mul.isChecked():
                    return
                h = self.hsv_mul_sliders.sliders['H'].conv_val()
                s = self.hsv_mul_sliders.sliders['S'].conv_val()
                v = self.hsv_mul_sliders.sliders['V'].conv_val()
                self.mixer.hsv_multiply(h, s, v)
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
                self.hsv_add_sliders.set_sliders({'H': 0, 'S': 0, 'V': 0})
                self.hsv_mul_sliders.set_sliders({'H': 0, 'S': 500, 'V': 500})
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
                self.hsv_widget.hide()
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

            def hsv_radio_changed(self):
                if self.hsv_add.isChecked():
                    self.hsv_add_sliders.show()
                    self.hsv_mul_sliders.hide()
                elif self.hsv_mul.isChecked():
                    self.hsv_mul_sliders.show()
                    self.hsv_add_sliders.hide()
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
                self.hsv_widget.show()

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


        class Histogram(QWidget):
            '''A Class which draws a scaling histogram in
            a widget.

            The argument to the constructor 'vals' is a list of tuples
            of the following form:

            vals = [(counts, colormap)]

            where counts are the bin values in the histogram
            and colormap is a tuple of (R, G, B) tuples the same length
            as counts. These are the colors to apply to the histogram bars.
            Colormap can also contain a single tuple, in which case this is
            the color applied to all bars of that histogram.

            Each histogram is drawn in order from left to right in its own
            box and the values are scaled so that max(count) = height.
            This is a linear scaling.

            The histogram assumes the bins were evenly spaced.
            '''

            def __init__(self, counts, colormap):
                QWidget.__init__(self)
                self._validate_input(counts, colormap)
                self.counts = counts
                self.n = np.sum(self.counts)
                self.colormap = colormap
                self.setMinimumSize(100, 50)

            def _validate_input(self, counts, colormap):
                if len(counts) != len(colormap):
                    if len(colormap) != 1:
                        msg = 'Colormap must be same length as count or 1'
                        raise ValueError(msg)

            def paintEvent(self, evt):
                # get the widget dimensions
                orig_width = self.width()
                orig_height = self.height()

                # fill perc % of the widget
                perc =  1.0
                width = int(orig_width * perc)
                height = int(orig_height * perc)

                # get the starting origin
                x_orig = int((orig_width - width) / 2)
                # we want to start at the bottom and draw up.
                y_orig = orig_height - int((orig_height - height) / 2)

                # a running x-position
                running_pos = x_orig

                # calculate to number of bars
                nbars = len(self.counts)


                # calculate the bar widths, this compilcation is
                # necessary because integer trunction severly cripples
                # the layout.
                remainder = width % nbars
                bar_width = [int(width / nbars)] * nbars
                for i in range(remainder):
                    bar_width[i]+=1

                paint = QPainter()
                paint.begin(self)

                if len(self.colormap) == 1:
                    self.colormap = self.colormap * len(self.counts)

                # determine the scaling factor
                max_val = np.max(self.counts)
                scale =  1. * height / max_val

                # draw the bars for this graph
                for i in range(len(self.counts)):
                    bar_height = self.counts[i] * scale
                    r, g, b = self.colormap[i]
                    paint.setPen(QColor(r, g, b))
                    paint.setBrush(QColor(r, g, b))
                    paint.drawRect(running_pos, y_orig, bar_width[i], -bar_height)
                    running_pos += bar_width[i]

                paint.end()


            def update_hist(self, counts, cmap):
                self._validate_input(counts, cmap)
                self.counts = counts
                self.colormap = cmap
                self.repaint()



        class MultiHist(QFrame):
            def __init__(self, vals):
                QFrame.__init__(self)

                self.hists = []
                for counts, cmap in vals:
                    self.hists.append(Histogram(counts, cmap))

                self.layout = QtGui.QGridLayout(self)
                for i in range(len(self.hists))[::-1]:
                    self.layout.addWidget(self.hists[i], i, 0)


            def update_hists(self, vals):
                for i in range(len(vals)):
                    counts, cmap = vals[i]
                    self.hists[i].update_hist(counts, cmap)


        class FancyImageWindow(ImageWindow):
            def __init__(self, arr, mgr):
                ImageWindow.__init__(self, arr, mgr)
                self.arr = arr

                self.label.setScaledContents(True)
                self.label.setMouseTracking(True)
                self.label.setMinimumSize(QtCore.QSize(100, 100))

                self.mixer_panel = MixerPanel(self.arr, self.refresh_image)
                self.layout.addWidget(self.mixer_panel, 0, 2)
                self.mixer_panel.show()

                self.rgb_hist = MultiHist(self.calc_hist())
                self.layout.addWidget(self.rgb_hist, 0, 1)
                self.rgb_hist.show()

                self.rgb_hsv_disp = RGBHSVDisplay()
                self.layout.addWidget(self.rgb_hsv_disp, 1, 0)
                self.rgb_hsv_disp.show()

                self.layout.setColumnStretch(0, 1)
                self.layout.setRowStretch(0, 1)

                # hook up the mixer sliders move events to trigger a
                # histogram redraw.
                self.mixer_panel.rgb_add_sliders.sliders['R'].\
                    sliderMoved.connect(self.update_histogram)
                self.mixer_panel.rgb_add_sliders.sliders['G'].\
                    sliderMoved.connect(self.update_histogram)
                self.mixer_panel.rgb_add_sliders.sliders['B'].\
                    sliderMoved.connect(self.update_histogram)
                self.mixer_panel.rgb_mul_sliders.sliders['R'].\
                    sliderMoved.connect(self.update_histogram)
                self.mixer_panel.rgb_mul_sliders.sliders['G'].\
                    sliderMoved.connect(self.update_histogram)
                self.mixer_panel.rgb_mul_sliders.sliders['B'].\
                    sliderMoved.connect(self.update_histogram)
                self.mixer_panel.hsv_add_sliders.sliders['H'].\
                    sliderMoved.connect(self.update_histogram)
                self.mixer_panel.hsv_add_sliders.sliders['S'].\
                    sliderMoved.connect(self.update_histogram)
                self.mixer_panel.hsv_add_sliders.sliders['V'].\
                    sliderMoved.connect(self.update_histogram)
                self.mixer_panel.hsv_mul_sliders.sliders['H'].\
                    sliderMoved.connect(self.update_histogram)
                self.mixer_panel.hsv_mul_sliders.sliders['S'].\
                    sliderMoved.connect(self.update_histogram)
                self.mixer_panel.hsv_mul_sliders.sliders['V'].\
                    sliderMoved.connect(self.update_histogram)
                self.mixer_panel.bright_sliders.sliders['+'].\
                    sliderMoved.connect(self.update_histogram)
                self.mixer_panel.bright_sliders.sliders['x'].\
                    sliderMoved.connect(self.update_histogram)

            def update_histogram(self):
                self.rgb_hist.update_hists(self.calc_hist())

            def calc_hist(self):
                rvals, gvals, bvals, grays = \
                     self.mixer_panel.mixer.histograms(100)

                vals = ((rvals, ((255,0,0),)),(gvals, ((0,255,0),)),
                        (bvals, ((0,0,255),)), (grays, ((0, 0, 0),)))
                return vals

            def refresh_image(self):
                pm = QPixmap.fromImage(self.label.img)
                self.label.setPixmap(pm)
                self.update_histogram()

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

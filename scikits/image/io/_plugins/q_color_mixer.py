# the module for the qt color_mixer plugin
from PyQt4 import QtGui, QtCore
from PyQt4.QtGui import (QWidget, QStackedWidget, QSlider, QVBoxLayout,
                         QGridLayout, QLabel)

from util import ColorMixer




class IntelligentSlider(QWidget):
    ''' A slider that adds a 'name' attribute and calls a callback
    with 'name' as an argument to the registerd callback.

    This allows you to create large groups of sliders in a loop,
    but still keep track of the individual events

    It also prints a label below the slider.

    The range of the slider is hardcoded from zero - 1000,
    but it supports a conversion factor so you can scale the results'''

    def __init__(self, name, a, b, callback):
        QWidget.__init__(self)
        self.name = name
        self.callback = callback
        self.a = a
        self.b = b

        self.slider = QSlider()
        self.slider.setRange(0, 1000)
        self.slider.setValue(500)
        self.slider.valueChanged.connect(self.slider_changed)

        self.name_label = QLabel()
        self.name_label.setText(self.name)
        self.name_label.setAlignment(QtCore.Qt.AlignCenter)

        self.value_label = QLabel()
        self.value_label.setText(str(self.slider.value() * self.a + self.b))
        self.value_label.setAlignment(QtCore.Qt.AlignCenter)

        self.layout = QVBoxLayout(self)
        self.layout.addWidget(self.name_label)
        self.layout.addWidget(self.slider)
        self.layout.addWidget(self.value_label)

    # bind this to the valueChanged signal of the slider
    def slider_changed(self, val):
        val = self.val()
        self.value_label.setText(str(val)[:4])
        self.callback(self.name, val)

    def set_conv_fac(self, a, b):
        self.a = a
        self.b = b

    def set_value(self, val):
        self.slider.setValue(int((val - self.a) / self.b))

    def val(self):
        return self.value() * self.a + self.b


class MixerPanel(QWidget):
    '''A color mixer to hook up to an image.
    You pass the image you the panel to operate on
    and it operates on that image in place. You also
    pass a callback to be called to trigger a refresh.
    This callback is called every time the mixer modifies
    your image.'''
    def __init__(self, img):
        QWidget.__init__(self)

        self.img = img
        self.mixer = ColorMixer(self.img)

        #---------------------------------------------------------------
        # ComboBox
        #---------------------------------------------------------------

        self.combo_box_entries = ['RGB Color', 'HSV Color',
                                  'Brightness/Contrast',
                                  'Gamma (Sigmoidal)']
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

        # sliders
        rs = IntelligentSlider('R', 0.51, -255, self.rgb_changed)
        gs = IntelligentSlider('G', 0.51, -255, self.rgb_changed)
        bs = IntelligentSlider('B', 0.51, -255, self.rgb_changed)
        self.rs = rs
        self.gs = gs
        self.bs = bs

        self.rgb_widget = QWidget()
        self.rgb_widget.layout = QGridLayout(self.rgb_widget)
        self.rgb_widget.layout.addWidget(self.rgb_add, 0, 0, 1, 3)
        self.rgb_widget.layout.addWidget(self.rgb_mul, 1, 0, 1, 3)
        self.rgb_widget.layout.addWidget(self.rs, 2, 0)
        self.rgb_widget.layout.addWidget(self.gs, 2, 1)
        self.rgb_widget.layout.addWidget(self.bs, 2, 2)


        #---------------------------------------------------------------
        # HSV sliders
        #---------------------------------------------------------------

        # radio buttons
        self.hsv_add = QtGui.QRadioButton('Additive')
        self.hsv_mul = QtGui.QRadioButton('Multiplicative')
        self.hsv_mul.toggled.connect(self.hsv_radio_changed)
        self.hsv_mul.toggled.connect(self.hsv_radio_changed)

        # sliders
        hs = IntelligentSlider('H', 0.36, -180, self.hsv_changed)
        ss = IntelligentSlider('S', 0.002, 0, self.hsv_changed)
        vs = IntelligentSlider('V', 0.002, 0, self.hsv_changed)
        self.hs = hs
        self.ss = ss
        self.vs = vs

        self.hsv_widget = QWidget()
        self.hsv_widget.layout = QGridLayout(self.hsv_widget)
        self.hsv_widget.layout.addWidget(self.hsv_add, 0, 0, 1, 3)
        self.hsv_widget.layout.addWidget(self.hsv_mul, 1, 0, 1, 3)
        self.hsv_widget.layout.addWidget(self.hs, 2, 0)
        self.hsv_widget.layout.addWidget(self.ss, 2, 1)
        self.hsv_widget.layout.addWidget(self.vs, 2, 2)


        #---------------------------------------------------------------
        # Brightness/Contrast sliders
        #---------------------------------------------------------------

        # sliders
        cont = IntelligentSlider('x', 0.002, 0, self.bright_changed)
        bright = IntelligentSlider('+', 0.51, -155, self.bright_changed)
        self.cont = cont
        self.bright = bright

        # layout
        self.bright_widget = QWidget()
        self.bright_widget.layout = QtGui.QGridLayout(self.bright_widget)
        self.bright_widget.layout.addWidget(self.cont, 0, 0)
        self.bright_widget.layout.addWidget(self.bright, 0, 1)


        '''
        #---------------------------------------------------------------
        # Gamma sliders
        #---------------------------------------------------------------
        # sliders
        self.gamma_sliders = NSliderBlock(2,
                                           [(100, 1200, 100, 'alpha', 0.01),
                                            (0, 1200, 0, 'beta', 0.01)],
                                           self.gamma_changed)

        # layout
        self.gamma_widget = QWidget()
        self.gamma_widget.layout = QtGui.QGridLayout(self.gamma_widget)
        self.gamma_widget.layout.addWidget(self.gamma_sliders, 0, 0)
        '''

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
        self.sliders = QStackedWidget()
        self.sliders.addWidget(self.rgb_widget)
        self.sliders.addWidget(self.hsv_widget)
        self.sliders.addWidget(self.bright_widget)

        self.layout = QtGui.QGridLayout(self)
        self.layout.addWidget(self.combo_box, 0, 0)
        self.layout.addWidget(self.sliders, 1, 0)
        self.layout.addWidget(self.commit_button, 2, 0)
        self.layout.addWidget(self.revert_button, 3, 0)

        #---------------------------------------------------------------
        # Initialization
        #---------------------------------------------------------------

        self.combo_box.setCurrentIndex(0)
        self.sliders.setCurrentIndex(2)
        #self.hide_sliders()
        #self.rgb_widget.show()
        #self.rgb_mul.setChecked(True)
        #self.hsv_mul.setChecked(True)

    def rgb_changed(self, name, val):
        pass


    def hsv_changed(self, name, val):
        pass

    def bright_changed(self, name, val):
        # doesnt matter which slider changed we need both
        # values
        factor = self.bright_sliders.sliders['x'].conv_val()
        offset = self.bright_sliders.sliders['+'].conv_val()
        self.mixer.brightness(offset, factor)
        self.update()

    def gamma_changed(self, name, val):
        # doesnt matter which slider changed we need both
        # values
        alpha = self.gamma_sliders.sliders['alpha'].conv_val()
        beta = self.gamma_sliders.sliders['beta'].conv_val()
        self.mixer.sigmoid_gamma(alpha, beta)
        self.update()

    def iter_all_sliders(self):
        pass

    def reset_sliders(self):
        self.rgb_add_sliders.set_sliders({'R': 0, 'G': 0, 'B': 0})
        self.rgb_mul_sliders.set_sliders({'R': 500, 'G': 500, 'B': 500})
        self.hsv_add_sliders.set_sliders({'H': 0, 'S': 0, 'V': 0})
        self.hsv_mul_sliders.set_sliders({'H': 0, 'S': 500, 'V': 500})
        self.bright_sliders.set_sliders({'+': 0, 'x': 500})
        self.gamma_sliders.set_sliders({'alpha': 100, 'beta': 0})

    def combo_box_changed(self, index):
        self.reset_sliders()
        self.mixer.set_to_stateimg()
        self.update()
        combo_box_map={0: self.show_rgb, 1: self.show_hsv,
                       2: self.show_bright, 3: self.show_gamma}
        combo_box_map[index]()

    def hide_sliders(self):
        self.rgb_widget.hide()
        self.hsv_widget.hide()
        self.bright_widget.hide()
        self.gamma_sliders.hide()

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

    def show_gamma(self):
        self.hide_sliders()
        self.gamma_sliders.show()

    def commit_changes(self):
        self.mixer.commit_changes()
        self.update()

    def revert_changes(self):
        self.mixer.revert()
        self.reset_sliders()
        self.update()
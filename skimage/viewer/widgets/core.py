"""
Widgets for interacting with ImageViewer.

These widgets should be added to a Plugin subclass using its `add_widget`
method. The Plugin will delegate action based on the widget's parameter type
specified by its `ptype` attribute, which can be:

    'arg' : positional argument passed to Plugin's `filter_image` method.
    'kwarg' : keyword argument passed to Plugin's `filter_image` method.
    'plugin' : attribute of Plugin. You'll probably need to make the attribute
        a class property that updates the display.

"""
from PyQt4 import QtGui
from PyQt4 import QtCore
from skimage.io._plugins.q_color_mixer import IntelligentSlider


#TODO: Add WidgetBase class (requires reimplementation of IntelligentSlider).

class Slider(IntelligentSlider):
    """Slider widget.

    Parameters
    ----------
    name : str
        Name of slider parameter. If this parameter is passed as a keyword
        argument, it must match the name of that keyword argument (spaces are
        replaced with underscores). In addition, this name is displayed as the
        name of the slider.
    low, high : float
        Range of slider values.
    ptype : {'arg' | 'kwarg' | 'plugin'}
        Parameter type.
    """
    def __init__(self, name, low, high, ptype='kwarg', callback=None, **kwargs):
        self.ptype = ptype
        kwargs.setdefault('orientation', 'horizontal')
        scale = (high - low) / 1000.0
        super(Slider, self).__init__(name, scale, low, callback, **kwargs)


class ComboBox(QtGui.QWidget):
    """ComboBox widget for selecting among a list of choices.

    Parameters
    ----------
    name : str
        Name of slider parameter. If this parameter is passed as a keyword
        argument, it must match the name of that keyword argument (spaces are
        replaced with underscores). In addition, this name is displayed as the
        name of the slider.
    items: list
        Allowed parameter values.
    ptype : {'arg' | 'kwarg' | 'plugin'}
        Parameter type.
    """

    def __init__(self, name, items, ptype='kwarg', callback=None):
        super(ComboBox, self).__init__()
        self.ptype = ptype
        self.callback = callback

        self.name = name
        self.name_label = QtGui.QLabel()
        self.name_label.setText(self.name)
        self.name_label.setAlignment(QtCore.Qt.AlignLeft)

        self._combo_box = QtGui.QComboBox()
        self._combo_box.addItems(items)

        self.layout = QtGui.QHBoxLayout(self)
        self.layout.addWidget(self.name_label)
        self.layout.addWidget(self._combo_box, alignment=QtCore.Qt.AlignLeft)

        self._combo_box.currentIndexChanged.connect(self._value_changed)
        # self.connect(self._combo_box,
                # SIGNAL("currentIndexChanged(int)"), self.updateUi)

    @property
    def val(self):
        return self._combo_box.value()

    def _value_changed(self, value):
        self.callback(self.name, value)

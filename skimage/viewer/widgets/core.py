"""
Widgets for interacting with ImageViewer.

These widgets should be added to a Plugin subclass using its `add_widget`
method or calling::

    plugin += Widget(...)

on a Plugin instance. The Plugin will delegate action based on the widget's
parameter type specified by its `ptype` attribute, which can be:

    'arg' : positional argument passed to Plugin's `filter_image` method.
    'kwarg' : keyword argument passed to Plugin's `filter_image` method.
    'plugin' : attribute of Plugin. You'll probably need to add a class
        property of the same name that updates the display.

"""
try:
    from PyQt4.QtCore import Qt
    from PyQt4 import QtGui
    from PyQt4 import QtCore
    from PyQt4.QtGui import QWidget
except ImportError:
    QWidget = object  # hack to prevent nosetest and autodoc errors
    print("Could not import PyQt4 -- skimage.viewer not available.")

from ..utils import RequiredAttr


__all__ = ['BaseWidget', 'Slider', 'ComboBox']


class BaseWidget(QWidget):

    plugin = RequiredAttr("Widget is not attached to a Plugin.")

    def __init__(self, name, ptype=None, callback=None):
        super(BaseWidget, self).__init__()
        self.name = name
        self.ptype = ptype
        self.callback = callback
        self.plugin = None

    @property
    def val(self):
        msg = "Subclass of BaseWidget requires `val` property"
        raise NotImplementedError(msg)

    def _value_changed(self, value):
        self.callback(self.name, value)


class Slider(BaseWidget):
    """Slider widget for adjusting numeric parameters.

    Parameters
    ----------
    name : str
        Name of slider parameter. If this parameter is passed as a keyword
        argument, it must match the name of that keyword argument (spaces are
        replaced with underscores). In addition, this name is displayed as the
        name of the slider.
    low, high : float
        Range of slider values.
    value : float
        Default slider value. If None, use midpoint between `low` and `high`.
    value : {'float' | 'int'}
        Numeric type of slider value.
    ptype : {'arg' | 'kwarg' | 'plugin'}
        Parameter type.
    callback : function
        Callback function called in response to slider changes. This function
        is typically set when the widget is added to a plugin.
    orientation : {'horizontal' | 'vertical'}
        Slider orientation.
    update_on : {'move' | 'release'}
        Control when callback function is called: on slider move or release.
    """
    def __init__(self, name, low=0.0, high=1.0, value=None, value_type='float',
                 ptype='kwarg', callback=None, max_edit_width=60,
                 orientation='horizontal', update_on='move'):
        super(Slider, self).__init__(name, ptype, callback)

        if value is None:
            value = (high - low) / 2.

        # Set widget orientation
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if orientation == 'vertical':
            self.slider = QtGui.QSlider(Qt.Vertical)
            alignment = QtCore.Qt.AlignHCenter
            align_text = QtCore.Qt.AlignHCenter
            align_value = QtCore.Qt.AlignHCenter
            self.layout = QtGui.QVBoxLayout(self)
        elif orientation == 'horizontal':
            self.slider = QtGui.QSlider(Qt.Horizontal)
            alignment = QtCore.Qt.AlignVCenter
            align_text = QtCore.Qt.AlignLeft
            align_value = QtCore.Qt.AlignRight
            self.layout = QtGui.QHBoxLayout(self)
        else:
            msg = "Unexpected value %s for 'orientation'"
            raise ValueError(msg % orientation)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # Set slider behavior for float and int values.
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if value_type == 'float':
            # divide slider into 1000 discrete values
            slider_max = 1000
            self._scale = float(high - low) / slider_max
            self.slider.setRange(0, slider_max)
            self.value_fmt = '%2.2f'
        elif value_type == 'int':
            self.slider.setRange(low, high)
            self.value_fmt = '%d'
        else:
            msg = "Expected `value_type` to be 'float' or 'int'; received: %s"
            raise ValueError(msg % value_type)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        self.value_type = value_type
        self._low = low
        self._high = high
        # Update slider position to default value
        self.val = value

        if update_on == 'move':
            self.slider.valueChanged.connect(self._on_slider_changed)
        elif update_on == 'release':
            self.slider.sliderReleased.connect(self._on_slider_changed)
        else:
            raise ValueError("Unexpected value %s for 'update_on'" % update_on)

        self.name_label = QtGui.QLabel()
        self.name_label.setText(self.name)
        self.name_label.setAlignment(align_text)

        self.editbox = QtGui.QLineEdit()
        self.editbox.setMaximumWidth(max_edit_width)
        self.editbox.setText(self.value_fmt % self.val)
        self.editbox.setAlignment(align_value)
        self.editbox.editingFinished.connect(self._on_editbox_changed)

        self.layout.addWidget(self.name_label, alignment=align_text)
        self.layout.addWidget(self.slider, alignment=alignment)
        self.layout.addWidget(self.editbox, alignment=align_value)

    def _on_slider_changed(self):
        """Call callback function with slider's name and value as parameters"""
        value = self.val
        self.editbox.setText(str(value)[:4])
        self.callback(self.name, value)

    def _on_editbox_changed(self):
        """Validate input and set slider value"""
        try:
            value = float(self.editbox.text())
        except ValueError:
            self._bad_editbox_input()
            return
        if not self._low <= value <= self._high:
            self._bad_editbox_input()
            return

        self.val = value
        self._good_editbox_input()
        self.callback(self.name, value)

    def _good_editbox_input(self):
        self.editbox.setStyleSheet("background-color: rgb(255, 255, 255)")

    def _bad_editbox_input(self):
        self.editbox.setStyleSheet("background-color: rgb(255, 200, 200)")

    @property
    def val(self):
        value = self.slider.value()
        if self.value_type == 'float':
            value = value * self._scale + self._low
        return value

    @val.setter
    def val(self, value):
        if self.value_type == 'float':
            value = (value - self._low) / self._scale
        self.slider.setValue(value)


class ComboBox(BaseWidget):
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
    callback : function
        Callback function called in response to slider changes. This function
        is typically set when the widget is added to a plugin.
    """

    def __init__(self, name, items, ptype='kwarg', callback=None):
        super(ComboBox, self).__init__(name, ptype, callback)

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

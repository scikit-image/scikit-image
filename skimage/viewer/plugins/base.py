from PyQt4 import QtGui
from skimage.io._plugins.q_color_mixer import IntelligentSlider


class Plugin(QtGui.QDialog):
    """Base class for widgets that interact with the axes.

    Parameters
    ----------
    image_viewer : ImageViewer instance.
        Window containing image used in measurement/manipulation.
    useblit : bool
        If True, use blitting to speed up animation. Only available on some
        backends. If None, set to True when using Agg backend, otherwise False.
    figure : :class:`~matplotlib.figure.Figure`
        If None, create a figure with a single axes.
    no_toolbar : bool
        If True, figure created by plugin has no toolbar. This has no effect
        on figures passed into `Plugin`.

    Attributes
    ----------
    viewer : ImageViewer
        Window containing image used in measurement.
    image : array
        Image used in measurement/manipulation.
    overlay : array
        Image used in measurement/manipulation.
    """
    def __init__(self, image_viewer, callback=None, height=100, width=400):
        self._viewer = image_viewer
        QtGui.QDialog.__init__(self, image_viewer)
        self.setWindowTitle('Image Plugin')
        self.layout = QtGui.QGridLayout(self)
        self.resize(width, height)
        self.row = 0
        if callback is not None:
            self.callback = callback

        self.arguments = [image_viewer.original_image]
        self.keyword_arguments= {}

        self.overlay = self._viewer.overlay
        self.image = self._viewer.image

    def caller(self, *args):
        arguments = [self._get_value(a) for a in self.arguments]
        kwargs = dict([(name, self._get_value(a))
                       for name, a in self.keyword_arguments.iteritems()])
        self.callback(*arguments, **kwargs)

    def _get_value(self, param):
        if hasattr(param, 'val'):
            return param.val()
        else:
            return param

    def add_argument(self, name, low, high, callback):
        name, slider = self.add_slider(name, low, high, callback)
        self.arguments[name] = slider

    def add_keyword_argument(self, name, low, high, callback):
        name, slider = self.add_slider(name, low, high, callback)
        self.keyword_arguments[name] = slider

    def add_slider(self, name, low, high, callback):
        slider = IntelligentSlider(name, low, high, callback,
                                   orientation='horizontal')
        self.layout.addWidget(slider, self.row, 0)
        self.row += 1
        return name.replace(' ', '_'), slider

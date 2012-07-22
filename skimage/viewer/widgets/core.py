from skimage.io._plugins.q_color_mixer import IntelligentSlider

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
    ptype : {'arg' | 'kwarg' | ...}
        Parameter
    """
    def __init__(self, name, low, high, ptype='kwarg', callback=None, **kwargs):
        self.ptype = ptype
        kwargs.setdefault('orientation', 'horizontal')
        scale = (high - low) / 1000.0
        super(Slider, self).__init__(name, scale, low, callback, **kwargs)

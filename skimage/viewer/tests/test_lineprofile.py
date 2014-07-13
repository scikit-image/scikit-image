import numpy as np

from skimage import data
from skimage.filter.rank import median
from skimage.morphology import disk

from skimage.viewer import ImageViewer
from skimage.viewer.widgets import Slider, OKCancelButtons, SaveButtons
from skimage.viewer.plugins.base import Plugin
from skimage.viewer.plugins import LineProfile

from skimage.viewer.qt import qt_api, QtCore
from numpy.testing import assert_almost_equal
from numpy.testing.decorators import skipif


def median_filter(image, radius):
    return median(image, selem=disk(radius))


@skipif(qt_api is None)
def test_custom_plugin():

    image = data.coins()[:-50, :]  # shave some off to make the line lower
    viewer = ImageViewer(image)

    plugin = Plugin(image_filter=median_filter)
    sld = Slider('radius', 2, 10, value_type='int')
    plugin += sld
    sv = SaveButtons()
    plugin += sv
    ok = OKCancelButtons()
    plugin += ok

    viewer += plugin

    lp = LineProfile()
    viewer += lp

    def validate_nogui():
        line = lp.get_profiles()[-1][0]
        assert line.size == 129
        assert_almost_equal(np.std(viewer.image), 49.998, 3)
        assert_almost_equal(np.std(line), 56.012, 3)
        assert_almost_equal(np.max(line) - np.min(line), 159.0, 1)

    def validate_gui():
        line = lp.get_profiles()[-1][0]
        assert_almost_equal(np.std(viewer.image), 51.364, 3)
        assert_almost_equal(np.std(line), 56.3555, 3)
        assert_almost_equal(np.max(line) - np.min(line), 172.0, 1)

    assert plugin in viewer.plugins
    assert lp in viewer.plugins

    validate_nogui()

    timer = QtCore.QTimer()
    timer.singleShot(1000, lambda: setattr(sld, 'val', 1))
    timer.singleShot(1100, validate_gui)
    timer.singleShot(1200, lambda: ok.update_original_image())
    timer.singleShot(1300, lambda: viewer.close())

    viewer.show()


import os
from skimage import data, img_as_float, io
from numpy.testing import assert_almost_equal, assert_equal
from numpy.testing.decorators import skipif

try:
    from skimage.viewer.qt import qt_api, QtGui, QtCore
    from skimage.viewer import ImageViewer
    from skimage.viewer.widgets import (
        Slider, OKCancelButtons, SaveButtons, ComboBox, Text)
    from skimage.viewer.plugins.base import Plugin
except ImportError:
    skip_all = True
else:
    skip_all = False


def get_image_viewer():
    image = data.coins()
    viewer = ImageViewer(img_as_float(image))
    viewer += Plugin()
    return viewer


@skipif(skip_all or qt_api is None)
def test_combo_box():
    viewer = get_image_viewer()
    cb = ComboBox('hello', ('a', 'b', 'c'))
    viewer.plugins[0] += cb

    assert_equal(str(cb.val), 'a')
    assert_equal(cb.index, 0)
    cb.index = 2
    assert_equal(str(cb.val), 'c'),
    assert_equal(cb.index, 2)


@skipif(skip_all or qt_api is None)
def test_text_widget():
    viewer = get_image_viewer()
    txt = Text('hello', 'hello, world!')
    viewer.plugins[0] += txt

    assert_equal(str(txt.text), 'hello, world!')
    txt.text = 'goodbye, world!'
    assert_equal(str(txt.text), 'goodbye, world!')


@skipif(skip_all or qt_api is None)
def test_slider_int():
    viewer = get_image_viewer()
    sld = Slider('radius', 2, 10, value_type='int')
    viewer.plugins[0] += sld

    assert_equal(sld.val, 4)
    sld.val = 6
    assert_equal(sld.val, 6)
    sld.editbox.setText('5')
    sld._on_editbox_changed()
    assert_equal(sld.val, 5)


@skipif(skip_all or qt_api is None)
def test_slider_float():
    viewer = get_image_viewer()
    sld = Slider('alpha', 2.1, 3.1, value=2.1, value_type='float',
                 orientation='vertical', update_on='move')
    viewer.plugins[0] += sld

    assert_equal(sld.val, 2.1)
    sld.val = 2.5
    assert_almost_equal(sld.val, 2.5, 2)
    sld.editbox.setText('0.1')
    sld._on_editbox_changed()
    assert_almost_equal(sld.val, 2.5, 2)


@skipif(skip_all or qt_api is None)
def test_save_buttons():
    viewer = get_image_viewer()
    sv = SaveButtons()
    viewer.plugins[0] += sv

    import tempfile
    _, filename = tempfile.mkstemp(suffix='.png')
    os.remove(filename)

    timer = QtCore.QTimer()
    timer.singleShot(100, lambda: QtGui.QApplication.quit())

    sv.save_to_stack()
    sv.save_to_file(filename)

    img = img_as_float(data.imread(filename))
    assert_almost_equal(img, viewer.image)

    img = io.pop()
    assert_almost_equal(img, viewer.image)


@skipif(skip_all or qt_api is None)
def test_ok_buttons():
    viewer = get_image_viewer()
    ok = OKCancelButtons()
    viewer.plugins[0] += ok

    ok.update_original_image(),
    ok.close_plugin()


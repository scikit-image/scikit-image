from numpy.testing import *

from skimage import io
from skimage.io._plugins import plugin
from numpy.testing.decorators import skipif

try:
    io.use_plugin('pil')
    PIL_available = True
    priority_plugin = 'pil'
except ImportError:
    PIL_available = False

try:
    io.use_plugin('freeimage')
    FI_available = True
    priority_plugin = 'freeimage'
except RuntimeError:
    FI_available = False


def setup_module(self):
    plugin.use('test')  # see ../_plugins/test_plugin.py


def teardown_module(self):
    io.reset_plugins()


class TestPlugin:
    def test_read(self):
        io.imread('test.png', as_grey=True, dtype='i4', plugin='test')

    def test_save(self):
        io.imsave('test.png', [1, 2, 3], plugin='test')

    def test_show(self):
        io.imshow([1, 2, 3], plugin_arg=(1, 2), plugin='test')

    def test_collection(self):
        io.imread_collection('*.png', conserve_memory=False, plugin='test')

    def test_use(self):
        plugin.use('test')
        plugin.use('test', 'imshow')

    @raises(ValueError)
    def test_failed_use(self):
        plugin.use('asd')

    @skipif(not PIL_available and not FI_available)
    def test_use_priority(self):
        plugin.use(priority_plugin)
        plug, func = plugin.plugin_store['imread'][0]
        assert_equal(plug, priority_plugin)

        plugin.use('test')
        plug, func = plugin.plugin_store['imread'][0]
        assert_equal(plug, 'test')

    @skipif(not PIL_available)
    def test_use_priority_with_func(self):
        plugin.use('pil')
        plug, func = plugin.plugin_store['imread'][0]
        assert_equal(plug, 'pil')

        plugin.use('test', 'imread')
        plug, func = plugin.plugin_store['imread'][0]
        assert_equal(plug, 'test')

        plug, func = plugin.plugin_store['imsave'][0]
        assert_equal(plug, 'pil')

        plugin.use('test')
        plug, func = plugin.plugin_store['imsave'][0]
        assert_equal(plug, 'test')

    def test_plugin_order(self):
        p = io.plugin_order()
        assert 'imread' in p
        assert 'test' in p['imread']

    def test_available(self):
        assert 'qt' in io.plugins()
        assert 'test' in io.plugins(loaded=True)

if __name__ == "__main__":
    run_module_suite()

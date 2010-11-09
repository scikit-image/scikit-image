from numpy.testing import *

from scikits.image import io
from scikits.image.io._plugins import plugin

from copy import deepcopy


def setup_module(self):
    self.backup_plugin_store = deepcopy(plugin.plugin_store)
    plugin.use('test') # see ../_plugins/test_plugin.py

def teardown_module(self):
    plugin.plugin_store = self.backup_plugin_store

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

    def test_use_priority(self):
        plugin.use('pil')
        plug, func = plugin.plugin_store['imread'][0]
        print plugin.plugin_store
        assert_equal(plug, 'pil')

        plugin.use('test')
        plug, func = plugin.plugin_store['imread'][0]
        print plugin.plugin_store
        assert_equal(plug, 'test')

    def test_available(self):
        assert 'qt' in io.plugins()
        assert 'test' in io.plugins(loaded=True)

if __name__ == "__main__":
    run_module_suite()

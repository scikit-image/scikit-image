from contextlib import contextmanager

from numpy.testing import assert_equal, raises

from skimage import io
from skimage.io import manage_plugins


io.use_plugin('pil')
priority_plugin = 'pil'


def setup_module():
    manage_plugins.use_plugin('test')  # see ../_plugins/test_plugin.py


def teardown_module():
    io.reset_plugins()


@contextmanager
def protect_preferred_plugins():
    """Contexts where `preferred_plugins` can be modified w/o side-effects."""
    preferred_plugins = manage_plugins.preferred_plugins.copy()
    try:
        yield
    finally:
        manage_plugins.preferred_plugins = preferred_plugins


def test_read():
    io.imread('test.png', as_grey=True, dtype='i4', plugin='test')


def test_save():
    io.imsave('test.png', [1, 2, 3], plugin='test')


def test_show():
    io.imshow([1, 2, 3], plugin_arg=(1, 2), plugin='test')


def test_collection():
    ic = io.imread_collection('*.png', conserve_memory=False, plugin='test')
    io.imshow_collection(ic)


def test_use():
    manage_plugins.use_plugin('test')
    manage_plugins.use_plugin('test', 'imshow')


@raises(ValueError)
def test_failed_use():
    manage_plugins.use_plugin('asd')


def test_use_priority():
    manage_plugins.use_plugin(priority_plugin)
    plug, func = manage_plugins.plugin_store['imread'][0]
    assert_equal(plug, priority_plugin)

    manage_plugins.use_plugin('test')
    plug, func = manage_plugins.plugin_store['imread'][0]
    assert_equal(plug, 'test')


def test_use_priority_with_func():
    manage_plugins.use_plugin('pil')
    plug, func = manage_plugins.plugin_store['imread'][0]
    assert_equal(plug, 'pil')

    manage_plugins.use_plugin('test', 'imread')
    plug, func = manage_plugins.plugin_store['imread'][0]
    assert_equal(plug, 'test')

    plug, func = manage_plugins.plugin_store['imsave'][0]
    assert_equal(plug, 'pil')

    manage_plugins.use_plugin('test')
    plug, func = manage_plugins.plugin_store['imsave'][0]
    assert_equal(plug, 'test')


def test_plugin_order():
    p = io.plugin_order()
    assert 'imread' in p
    assert 'test' in p['imread']


def test_available():
    assert 'qt' in io.available_plugins
    assert 'test' in io.find_available_plugins(loaded=True)


def test_load_preferred_plugins_all():
    from skimage.io._plugins import pil_plugin, matplotlib_plugin

    with protect_preferred_plugins():
        manage_plugins.preferred_plugins = {'all': ['pil'],
                                            'imshow': ['matplotlib']}
        manage_plugins.reset_plugins()

        for plugin_type in ('imread', 'imsave'):
            plug, func = manage_plugins.plugin_store[plugin_type][0]
            assert func == getattr(pil_plugin, plugin_type)
        plug, func = manage_plugins.plugin_store['imshow'][0]
        assert func == getattr(matplotlib_plugin, 'imshow')


def test_load_preferred_plugins_imread():
    from skimage.io._plugins import pil_plugin, matplotlib_plugin

    with protect_preferred_plugins():
        manage_plugins.preferred_plugins['imread'] = ['pil']
        manage_plugins.reset_plugins()

        plug, func = manage_plugins.plugin_store['imread'][0]
        assert func == pil_plugin.imread
        plug, func = manage_plugins.plugin_store['imshow'][0]
        assert func == matplotlib_plugin.imshow, func.__module__


if __name__ == "__main__":
    from numpy.testing import run_module_suite
    run_module_suite()

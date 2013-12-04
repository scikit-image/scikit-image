from skimage.io.inherited_config import InheritedConfig


def test_get_non_existent():
    config = InheritedConfig()
    assert config.get('size') is None


def test_get_simple():
    config = InheritedConfig({'imread': 'matplotlib'})
    assert config.get('imread') == 'matplotlib'


def test_get_default():
    config = InheritedConfig()
    assert config.get('size', 10) == 10


def test_get_best():
    config = InheritedConfig({'size': 0, 'size.text': 1})
    assert config.get('size.text') == 1


def test_get_multi_level():
    config = InheritedConfig({'size': 0})
    assert config.get('size.text.title') == 0
    config['size.text'] = 1
    assert config.get('size.text.title') == 1
    config['size.text.title'] = 2
    assert config.get('size.text.title') == 2


def test_contains():
    config = InheritedConfig({'imread': 'matplotlib'})
    assert 'imread' in config
    assert 'imread.jpg' in config


if __name__ == '__main__':
    from numpy import testing
    testing.run_module_suite()

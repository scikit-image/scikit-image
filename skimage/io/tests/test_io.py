import os
from urllib.error import URLError

import numpy as np
from skimage import io, data_dir

from skimage._shared import testing
from skimage._shared.testing import assert_array_equal


one_by_one_jpeg = (
    b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01'
    b'\x00\x01\x00\x00\xff\xdb\x00C\x00\x03\x02\x02\x02\x02'
    b'\x02\x03\x02\x02\x02\x03\x03\x03\x03\x04\x06\x04\x04'
    b'\x04\x04\x04\x08\x06\x06\x05\x06\t\x08\n\n\t\x08\t\t'
    b'\n\x0c\x0f\x0c\n\x0b\x0e\x0b\t\t\r\x11\r\x0e\x0f\x10'
    b'\x10\x11\x10\n\x0c\x12\x13\x12\x10\x13\x0f\x10\x10'
    b'\x10\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11'
    b'\x00\xff\xc4\x00\x14\x00\x01\x00\x00\x00\x00\x00\x00'
    b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\t\xff\xc4\x00'
    b'\x14\x10\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
    b'\x00\x00\x00\x00\x00\x00\xff\xda\x00\x08\x01\x01\x00'
    b'\x00?\x00*\x9f\xff\xd9'
)


def test_stack_basic():
    x = np.arange(12).reshape(3, 4)
    io.push(x)

    assert_array_equal(io.pop(), x)


def test_stack_non_array():
    with testing.raises(ValueError):
        io.push([[1, 2, 3]])


def test_imread_file_url():
    # tweak data path so that file URI works on both unix and windows.
    data_path = data_dir.lstrip(os.path.sep)
    data_path = data_path.replace(os.path.sep, '/')
    image_url = 'file:///{0}/camera.png'.format(data_path)
    image = io.imread(image_url)
    assert image.shape == (512, 512)


def test_imread_http_url(httpserver):
    # httpserver is a fixture provided by pytest-localserver
    # https://bitbucket.org/pytest-dev/pytest-localserver/
    httpserver.serve_content(one_by_one_jpeg)
    # it will serve anything you provide to it on its url.
    # we add a /test.jpg so that we can identify the content
    # by extension
    image = io.imread(httpserver.url + '/test.jpg' + '?' + 's' * 266)
    assert image.shape == (1, 1)


def test_imread_unreachable_url_handle():
    with testing.raises(URLError):
        io.imread('http://fake_url.com/image.jpg')

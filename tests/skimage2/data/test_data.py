import ast
import os
import warnings
from pathlib import Path

import numpy as np
import pytest

import _skimage2
import _skimage2.data as data
import _skimage2.data._fetchers as _fetchers
from _skimage2.data._fetchers import _image_fetcher
from _skimage2 import io
from _skimage2._shared.testing import (
    assert_equal,
    fetch,
)


@pytest.mark.thread_unsafe(reason="worker threads would share a download directory")
def test_download_all_with_pooch():
    # jni first wrote this test with the intention of
    # fully deleting the files in the data_dir,
    # then ensure that the data gets downloaded accordingly.
    # hmaarrfk raised the concern that this test wouldn't
    # play well with parallel testing since we
    # may be breaking the global state that certain other
    # tests require, especially in parallel testing

    # The second concern is that this test essentially uses
    # a lot of bandwidth, which is not fun for developers on
    # lower speed connections.
    # https://github.com/scikit-image/scikit-image/pull/4666/files/26d5138b25b958da6e97ebf979e9bc36f32c3568#r422604863
    data_dir = data.data_dir
    if _image_fetcher is not None:
        data.download_all()
        assert 'astronaut.png' in os.listdir(data_dir)
        assert len(os.listdir(data_dir)) > 50
    else:
        with pytest.raises(ModuleNotFoundError):
            data.download_all()


def test_astronaut():
    """Test that "astronaut" image can be loaded."""
    astronaut = data.fetch_astronaut()
    assert_equal(astronaut.shape, (512, 512, 3))


def test_camera():
    """Test that "camera" image can be loaded."""
    cameraman = data.fetch_camera()
    assert_equal(cameraman.ndim, 2)


def test_checkerboard():
    """Test that "checkerboard" image can be loaded."""
    data.fetch_checkerboard()


def test_chelsea():
    """Test that "chelsea" image can be loaded."""
    data.fetch_chelsea()


def test_clock():
    """Test that "clock" image can be loaded."""
    data.fetch_clock()


def test_coffee():
    """Test that "coffee" image can be loaded."""
    data.fetch_coffee()


def test_eagle():
    """Test that "eagle" image can be loaded."""
    # Fetching the data through the testing module will
    # cause the test to skip if pooch isn't installed.
    fetch('data/eagle.png')
    eagle = data.fetch_eagle()
    assert_equal(eagle.ndim, 2)
    assert_equal(eagle.dtype, np.dtype('uint8'))


def test_horse():
    """Test that "horse" image can be loaded."""
    horse = data.fetch_horse()
    assert_equal(horse.ndim, 2)
    assert_equal(horse.dtype, np.dtype('bool'))


def test_hubble():
    """Test that "Hubble" image can be loaded."""
    data.fetch_hubble_deep_field()


def test_immunohistochemistry():
    """Test that "immunohistochemistry" image can be loaded."""
    data.fetch_immunohistochemistry()


def test_logo():
    """Test that "logo" image can be loaded."""
    logo = data.fetch_logo()
    assert_equal(logo.ndim, 3)
    assert_equal(logo.shape[2], 4)


def test_moon():
    """Test that "moon" image can be loaded."""
    data.fetch_moon()


def test_page():
    """Test that "page" image can be loaded."""
    data.fetch_page()


def test_rocket():
    """Test that "rocket" image can be loaded."""
    data.fetch_rocket()


def test_text():
    """Test that "text" image can be loaded."""
    data.fetch_text()


def test_stereo_motorcycle():
    """Test that "stereo_motorcycle" image can be loaded."""
    data.fetch_stereo_motorcycle()


def test_lfw_subset():
    """Test that "lfw_subset" can be loaded."""
    data.fetch_lfw_subset()


def test_skin():
    """Test that "skin" image can be loaded.

    Needs internet connection.
    """
    skin = data.fetch_skin()
    assert skin.ndim == 3


def test_cell():
    """Test that "cell" image can be loaded."""
    data.fetch_cell()


def test_cells3d():
    """Needs internet connection."""
    path = fetch('data/cells3d.tif')
    image = io.imread(path)
    assert image.shape == (60, 2, 256, 256)


def test_brain_3d():
    """Needs internet connection."""
    path = fetch('data/brain.tiff')
    image = io.imread(path)
    assert image.shape == (10, 256, 256)


def test_kidney_3d_multichannel():
    """Test that 3D multichannel image of kidney tissue can be loaded.

    Needs internet connection.
    """
    fetch('data/kidney.tif')
    kidney = data.fetch_kidney()
    assert kidney.shape == (16, 512, 512, 3)


def test_lily_multichannel():
    """Test that microscopy image of lily of the valley can be loaded.

    Needs internet connection.
    """
    fetch('data/lily.tif')
    lily = data.fetch_lily()
    assert lily.shape == (922, 922, 4)


def test_vortex():
    fetch('data/pivchallenge-B-B001_1.tif')
    fetch('data/pivchallenge-B-B001_2.tif')
    image0, image1 = data.fetch_vortex()
    for image in [image0, image1]:
        assert image.shape == (512, 512)


@pytest.mark.parametrize(
    'function_name',
    [
        'fetch',
        'file_hash',
    ],
)
def test_fetchers_are_public(function_name):
    # Check that the following functions that are only used indirectly in the
    # above tests are public.
    assert hasattr(data, function_name)


# --- public fetch()/fetch_<name>() API surface ---


FETCH_FUNCTION_NAMES = [
    'fetch_astronaut',
    'fetch_brain',
    'fetch_brick',
    'fetch_camera',
    'fetch_cat',
    'fetch_cell',
    'fetch_cells3d',
    'fetch_checkerboard',
    'fetch_chelsea',
    'fetch_clock',
    'fetch_coffee',
    'fetch_coins',
    'fetch_colorwheel',
    'fetch_eagle',
    'fetch_grass',
    'fetch_gravel',
    'fetch_horse',
    'fetch_hubble_deep_field',
    'fetch_human_mitosis',
    'fetch_immunohistochemistry',
    'fetch_kidney',
    'fetch_lbp_frontal_face_cascade_filename',
    'fetch_lfw_subset',
    'fetch_lily',
    'fetch_logo',
    'fetch_microaneurysms',
    'fetch_moon',
    'fetch_nickel_solidification',
    'fetch_page',
    'fetch_palisades_of_vogt',
    'fetch_protein_transport',
    'fetch_retina',
    'fetch_rocket',
    'fetch_shepp_logan_phantom',
    'fetch_skin',
    'fetch_stereo_motorcycle',
    'fetch_text',
    'fetch_vortex',
]


@pytest.mark.parametrize('function_name', FETCH_FUNCTION_NAMES)
def test_bare_v1_names_are_unadvertised_aliases(function_name):
    """fetch_<name>() is public API; the bare v1 name (e.g. astronaut for
    fetch_astronaut) resolves to the same function but is absent from
    __all__ and __dir__."""
    bare_name = function_name.removeprefix('fetch_')
    assert hasattr(data, function_name)
    assert hasattr(data, bare_name)
    assert getattr(data, function_name) is getattr(data, bare_name)
    assert function_name in data.__all__
    assert bare_name not in data.__all__
    assert bare_name not in dir(data)


def test_bare_v1_name_is_importable():
    """The v1 alias imports the same way the fetch_ name does."""
    from _skimage2.data import astronaut, fetch_astronaut

    assert astronaut is fetch_astronaut


def test_public_skimage2_data_surface():
    """The v1 aliases and __all__ reach users through skimage2.data."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', _skimage2.ExperimentalAPIWarning)
        import skimage2

    public_data = skimage2.data
    assert public_data is data
    assert public_data.astronaut is public_data.fetch_astronaut
    assert 'fetch_astronaut' in public_data.__all__
    assert 'astronaut' not in public_data.__all__


def test_runtime_all_matches_stub():
    """Runtime __all__ lists exactly the names the stub declares in __all__."""
    stub_path = Path(data.__file__).with_name('__init__.pyi')
    tree = ast.parse(stub_path.read_text())
    stub_all = next(
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == '__all__'
            for target in node.targets
        )
    )
    stub_names = [element.value for element in stub_all.elts]
    assert data.__all__ == stub_names


def test_fetch_public_wrapper_matches_internal_fetch():
    """The public fetch() is a thin wrapper: it must resolve to the exact
    same path _fetch() would for the same registry key."""
    assert data.fetch('data/camera.png') == _fetchers._fetch('data/camera.png')

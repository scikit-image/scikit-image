import hashlib
from pathlib import Path

import numpy as np
import _skimage2.data as data
import _skimage2.data._fetchers as _fetchers
from _skimage2.data._fetchers import _image_fetcher
from _skimage2 import io
from _skimage2._shared.testing import (
    assert_equal,
    fetch,
)
from _skimage2._shared._dependency_checks import is_wasm
import os
import pytest


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


@pytest.mark.thread_unsafe(reason="mutates process-wide environment variables")
def test_skip_pytest_case_requiring_pooch_fires_during_collection(monkeypatch):
    """``PYTEST_VERSION`` is set for the whole session (including collection),
    unlike ``PYTEST_CURRENT_TEST`` (only set while a test's call/setup/teardown
    phase runs). Module- and class-level data fetches (e.g. ``IMG =
    data.fetch_astronaut()``) execute during collection, so the guard must
    also fire on ``PYTEST_VERSION`` alone."""
    monkeypatch.delenv('PYTEST_CURRENT_TEST', raising=False)
    monkeypatch.setenv('PYTEST_VERSION', '9.0.0')
    with pytest.raises(pytest.skip.Exception):
        _fetchers._skip_pytest_case_requiring_pooch('data/does_not_matter.png')


@pytest.mark.thread_unsafe(reason="mutates process-wide environment variables")
def test_skip_pytest_case_requiring_pooch_noop_outside_pytest(monkeypatch):
    """Without any pytest marker env var, the guard must not skip; it
    should only intervene when actually running under pytest."""
    monkeypatch.delenv('PYTEST_CURRENT_TEST', raising=False)
    monkeypatch.delenv('PYTEST_VERSION', raising=False)
    # Should return normally; raises if it unexpectedly tries to skip.
    _fetchers._skip_pytest_case_requiring_pooch('data/does_not_matter.png')


@pytest.mark.thread_unsafe(reason="monkeypatches shared fetcher module state")
def test_fetch_without_pooch_raises_clear_error(monkeypatch):
    """Without pooch, and without the file being bundled/cached/legacy, `_fetch`
    must raise a clear `ModuleNotFoundError` rather than attempting a download."""
    monkeypatch.setattr(_fetchers, '_image_fetcher', None)
    monkeypatch.setattr(
        _fetchers, '_skip_pytest_case_requiring_pooch', lambda *a, **kw: None
    )
    test_key = 'data/_test_no_pooch_no_bundle.bin'
    monkeypatch.setitem(_fetchers.registry, test_key, '0' * 64)

    with pytest.raises(ModuleNotFoundError, match='pooch'):
        _fetchers._fetch(test_key)


# --- optional `scikit-image-data` package (bundles a curated data subset) ---

requires_pooch = pytest.mark.skipif(
    _image_fetcher is None, reason="pooch is not installed / unavailable"
)


@requires_pooch
@pytest.mark.thread_unsafe(reason="monkeypatches shared fetcher module state")
def test_fetch_prefers_scikit_image_data_package(tmp_path, monkeypatch):
    """When installed, `scikit-image-data` is used instead of downloading --
    and nothing gets written to the download cache for a file it provides."""
    skimage_data = pytest.importorskip('skimage_data')
    monkeypatch.setattr(_fetchers._image_fetcher, 'path', tmp_path)

    result_path = _fetchers._fetch('data/astronaut.png')

    assert Path(result_path) == skimage_data.data_dir / 'astronaut.png'
    assert not (tmp_path / 'data' / 'astronaut.png').exists()


@requires_pooch
@pytest.mark.thread_unsafe(reason="monkeypatches shared fetcher module state")
@pytest.mark.skipif(is_wasm, reason="no access to pytest-localserver")
def test_fetch_downloads_file_not_in_scikit_image_data_package(
    httpserver, tmp_path, monkeypatch
):
    """A file `scikit-image-data` doesn't bundle still falls through to a
    normal download, and ends up cached on disk as usual."""
    pytest.importorskip('skimage_data')
    content = b'not bundled in scikit-image-data test content'
    expected_hash = hashlib.sha256(content).hexdigest()
    httpserver.serve_content(content)

    test_key = 'data/_test_not_in_scikit_image_data.bin'
    monkeypatch.setitem(_fetchers.registry, test_key, expected_hash)
    monkeypatch.setitem(_fetchers.registry_urls, test_key, httpserver.url)
    # `Pooch.urls` is copied at `pooch.create()` time. `Pooch.registry` is
    # also a separate dict now (filtered down to only CDN-hosted keys,
    # rather than being the same object as `_registry.py`'s `registry`) --
    # so both need patching separately from the module-level dicts above.
    monkeypatch.setitem(_fetchers._image_fetcher.urls, test_key, httpserver.url)
    monkeypatch.setitem(_fetchers._image_fetcher.registry, test_key, expected_hash)
    monkeypatch.setattr(_fetchers._image_fetcher, 'path', tmp_path)

    result_path = _fetchers._fetch(test_key)

    cached_path = tmp_path / 'data' / '_test_not_in_scikit_image_data.bin'
    assert Path(result_path) == cached_path
    assert cached_path.read_bytes() == content


# --- public fetch()/fetch_<name>() API matrix ---


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
def test_fetch_functions_are_public_and_bare_names_are_not(function_name):
    """Every dataset getter is only reachable as fetch_<name>(); the old
    bare name (e.g. astronaut() for fetch_astronaut()) must not resolve on
    _skimage2.data, even though the underlying function of that name still
    exists in _fetchers.py for skimage1's shim to import directly."""
    assert hasattr(data, function_name)
    assert not hasattr(data, function_name.removeprefix('fetch_'))


@requires_pooch
@pytest.mark.thread_unsafe(reason="monkeypatches shared fetcher module state")
@pytest.mark.parametrize('scikit_image_data_available', [True, False])
@pytest.mark.parametrize(
    'fetch_name, expected_shape',
    [
        ('fetch_astronaut', (512, 512, 3)),
        ('fetch_camera', (512, 512)),
    ],
)
def test_fetch_name_matrix_with_and_without_scikit_image_data(
    fetch_name, expected_shape, scikit_image_data_available, tmp_path, monkeypatch
):
    """fetch_<name>() functions bundled by scikit-image-data return correct
    data whether or not that package is actually installed: served from
    the bundle when available, downloaded using pooch and cached otherwise."""
    pytest.importorskip('skimage_data')
    if not scikit_image_data_available:
        monkeypatch.setattr(_fetchers, 'skimage_data', None)
    monkeypatch.setattr(_fetchers._image_fetcher, 'path', tmp_path)

    result = getattr(data, fetch_name)()

    assert result.shape == expected_shape
    basename = fetch_name.removeprefix('fetch_') + '.png'
    cached_path = tmp_path / 'data' / basename
    assert cached_path.exists() == (not scikit_image_data_available)


@requires_pooch
@pytest.mark.thread_unsafe(reason="monkeypatches shared fetcher module state")
def test_fetch_public_wrapper_matches_internal_fetch(tmp_path, monkeypatch):
    """The public fetch() is a thin wrapper: it must resolve to the exact
    same path _fetch() would for the same registry key."""
    monkeypatch.setattr(_fetchers._image_fetcher, 'path', tmp_path)
    assert data.fetch('data/camera.png') == _fetchers._fetch('data/camera.png')


# --- SKIMAGE_TEST_OFFLINE: opt-in test-time skip for un-cached fetches ---


@requires_pooch
@pytest.mark.thread_unsafe(reason="mutates process-wide environment variables")
def test_fetch_offline_env_var_skips_when_not_local(monkeypatch, tmp_path):
    """With SKIMAGE_TEST_OFFLINE set, _fetch() must skip (not attempt a
    download) for a file that isn't bundled or already cached, even though
    pooch itself is installed and available."""
    monkeypatch.setenv('SKIMAGE_TEST_OFFLINE', '1')
    monkeypatch.setattr(_fetchers, 'skimage_data', None)
    monkeypatch.setattr(_fetchers._image_fetcher, 'path', tmp_path)
    test_key = 'data/_test_offline_not_local.bin'
    monkeypatch.setitem(_fetchers.registry, test_key, '0' * 64)

    with pytest.raises(pytest.skip.Exception):
        _fetchers._fetch(test_key)


@requires_pooch
@pytest.mark.thread_unsafe(
    reason="mutates process-wide environment variables and shared fetcher state"
)
def test_download_all_ignores_offline_toggle(monkeypatch, tmp_path):
    """download_all() is an explicit request to download everything; it
    must ignore SKIMAGE_TEST_OFFLINE, unlike implicit per-dataset fetches."""
    monkeypatch.setenv('SKIMAGE_TEST_OFFLINE', '1')
    monkeypatch.setattr(_fetchers, 'skimage_data', None)
    test_key = 'data/_test_download_all_offline.bin'
    monkeypatch.setitem(_fetchers.registry, test_key, '0' * 64)
    monkeypatch.setattr(_fetchers._image_fetcher, 'registry', {test_key: '0' * 64})
    monkeypatch.setattr(_fetchers._image_fetcher, 'path', tmp_path)
    fake_path = tmp_path / 'data' / '_test_download_all_offline.bin'
    fake_path.parent.mkdir(parents=True, exist_ok=True)
    fake_path.write_bytes(b'x')
    monkeypatch.setattr(_fetchers._image_fetcher, 'fetch', lambda name: str(fake_path))

    # Should not raise/skip: proves _force_online=True bypassed the toggle.
    _fetchers.download_all()


def test_local_only_registry_files_are_present():
    """Regression guard for scikit-image's own CI (which always checks out
    the full repo, so `tests/` is always present regardless of whether the
    package itself is installed from a wheel or sdist; see
    `.github/workflows/_test_linux_for_python_x.yaml` etc., which all run
    `actions/checkout` before installing/testing).

    Every registry key with no `registry_urls` (CDN) entry is a test-only
    fixture expected to be tracked in git and read directly off disk (using
    `test_root_dir`/`local_data_path`, not `fetch()`). `local_data_path`
    skips gracefully when such a file is missing, which is the *correct*
    behavior for a downstream repackager testing against an installed
    wheel without `tests/`. This test guards the other case: a fixture
    going missing or corrupted by accident in an environment (like our own
    CI) that's supposed to have it, where a silent skip would hide a real
    regression.

    Only `tests/skimage2/` is checked: `tests/skimage/conftest.py`'s own
    `test_root_dir` fixture redirects there too (there's a single shared
    copy of these fixtures, not one per test suite; confirmed by reading
    that fixture's implementation, not assumed).
    """
    test_root = Path(__file__).resolve().parents[2] / 'skimage2'
    local_only_keys = sorted(set(_fetchers.registry) - set(_fetchers.registry_urls))
    assert local_only_keys, 'sanity check: expected some local-only registry keys'

    def matches(path, expected_hash):
        if _fetchers._has_hash(path, expected_hash):
            return True
        if not path.exists():
            return False
        # A plain-text fixture checked out on Windows may have git's
        # core.autocrlf normalize LF to CRLF, changing its hash without
        # changing its actual content; retry against the LF-normalized
        # bytes before declaring it corrupted.
        content = path.read_bytes()
        normalized = content.replace(b'\r\n', b'\n')
        if normalized == content:
            return False
        return hashlib.sha256(normalized).hexdigest() == expected_hash

    missing = [
        key
        for key in local_only_keys
        if not matches(test_root / key, _fetchers.registry[key])
    ]
    assert (
        not missing
    ), f'{test_root}: missing or corrupted local test-data file(s): {missing}'


# --- no GitHub raw-URL fallback: pooch only ever knows about CDN keys ---


@requires_pooch
def test_image_fetcher_registry_is_restricted_to_cdn_keys():
    """`_image_fetcher` must only know about registry keys that have a CDN
    url, never the full registry, so nothing can silently fall back to
    the (removed) GitHub raw-URL base_url for internal test-only fixtures.
    This also fixes download_all(), since it iterates
    `_image_fetcher.registry`."""
    assert set(_fetchers._image_fetcher.registry) == set(_fetchers.registry_urls)


@requires_pooch
@pytest.mark.thread_unsafe(reason="mutates shared fetcher module state")
def test_fetch_non_cdn_key_raises_pooch_registry_error_not_github(
    monkeypatch, tmp_path
):
    """Fetching a key with no CDN url must fail with pooch's own clear
    registry-membership error, not silently attempt (and possibly
    succeed at) a GitHub raw-URL download."""
    monkeypatch.setattr(_fetchers, 'skimage_data', None)
    monkeypatch.setattr(_fetchers._image_fetcher, 'path', tmp_path)
    non_cdn_key = next(iter(set(_fetchers.registry) - set(_fetchers.registry_urls)))

    with pytest.raises(ValueError, match='not in the registry'):
        _fetchers._fetch(non_cdn_key)

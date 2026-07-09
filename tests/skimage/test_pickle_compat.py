import pickle
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from skimage._pickle_compat import (
    PickleRef,
    SkimageUnpickler,
    pickle_dumps,
    pickle_loads,
    register_pickle_type,
    registered_pickle_refs,
)
from skimage.measure import regionprops

PROTOCOL = 4
GOLDEN_DIR = Path(__file__).parent / 'data' / 'pickles'
REGIONPROPS_PATH = 'skimage.measure._regionprops'
REGIONPROPS_IMPL_PATH = '_skimage2.measure._regionprops'
GEOMETRIC_PATH = 'skimage.transform._geometric'
GEOMETRIC_IMPL_PATH = '_skimage2.transform._geometric'

GOLDEN_CASES = (
    pytest.param(
        'regionproperties_0.25.pkl',
        REGIONPROPS_PATH,
        lambda obj: obj.area == 9.0 and obj.label == 1 and obj.bbox == (2, 2, 5, 5),
        id='regionproperties',
    ),
    pytest.param(
        'affine_transform_0.25.pkl',
        GEOMETRIC_PATH,
        lambda obj: np.allclose(obj.params, [[1, 0, 0], [0, 2, 0], [0, 0, 1]]),
        id='affine_transform',
    ),
    pytest.param(
        'thin_plate_spline_transform_0.25.pkl',
        'skimage.transform._thin_plate_splines',
        lambda obj: obj.src.shape == (4, 2),
        id='thin_plate_spline_transform',
    ),
    pytest.param(
        'image_collection_0.25.pkl',
        'skimage.io.collection',
        lambda obj: len(obj) == 2,
        id='image_collection',
    ),
)


def _sample_region():
    label_image = np.zeros((10, 10), dtype=int)
    label_image[2:5, 2:5] = 1
    regions = regionprops(label_image)
    assert len(regions) == 1
    return regions[0]


def _unpickler() -> SkimageUnpickler:
    return SkimageUnpickler.__new__(SkimageUnpickler)


def _assert_public_module(blob: bytes, module: str, *, forbid: str | None = None):
    assert module.encode() in blob
    if forbid is not None:
        assert forbid.encode() not in blob


def _stdlib_pickle_loads(blob: bytes):
    import skimage  # noqa: F401

    return pickle.loads(blob)


def test_regionproperties_exported_on_shim():
    from skimage.measure._regionprops import RegionProperties
    from _skimage2.measure._regionprops import RegionProperties as Impl

    assert RegionProperties is Impl


@pytest.mark.parametrize(
    ('dumps', 'loads'),
    [
        (pickle.dumps, pickle.loads),
        (pickle_dumps, pickle_loads),
    ],
)
def test_regionproperties_roundtrip_uses_public_path(dumps, loads):
    region = _sample_region()
    blob = dumps(region, protocol=PROTOCOL)
    _assert_public_module(blob, REGIONPROPS_PATH, forbid=REGIONPROPS_IMPL_PATH)
    assert loads(blob) == region


def test_regionproperties_impl_module_unchanged():
    from _skimage2.measure._regionprops import RegionProperties

    _sample_region()
    assert RegionProperties.__module__ == REGIONPROPS_IMPL_PATH


def test_affine_impl_pickle_out_uses_public_path():
    from _skimage2.transform._geometric import AffineTransform as Impl

    transform = Impl(scale=(1.0, 2.0))
    blob = pickle_dumps(transform, protocol=PROTOCOL)
    _assert_public_module(blob, GEOMETRIC_PATH, forbid=GEOMETRIC_IMPL_PATH)
    assert Impl.__module__ == GEOMETRIC_IMPL_PATH
    assert np.allclose(pickle_loads(blob).params, transform.params)


def test_pickle_loads_resolves_registered_path(monkeypatch):
    region = _sample_region()
    from skimage.measure import _regionprops as shim_mod

    blob = pickle_dumps(region, protocol=PROTOCOL)
    monkeypatch.delattr(shim_mod, 'RegionProperties', raising=False)
    assert pickle_loads(blob) == region


def test_legacy_ref_lookup():
    class Demo:
        pass

    ref = PickleRef('skimage.tests.demo', 'Demo')
    legacy = PickleRef('skimage.tests.legacy', 'Demo')
    register_pickle_type(ref, Demo, legacy=(legacy,))

    assert _unpickler().find_class(legacy.module, legacy.qualname) is Demo


@pytest.mark.parametrize(
    ('module', 'name', 'impl'),
    [
        ('skimage.transform._geometric', 'AffineTransform', 'AffineTransform'),
        (
            'skimage.transform._thin_plate_splines',
            'ThinPlateSplineTransform',
            'ThinPlateSplineTransform',
        ),
        ('skimage.io.collection', 'ImageCollection', 'ImageCollection'),
        ('skimage.io.collection', 'MultiImage', 'MultiImage'),
    ],
)
def test_registered_type_resolves(module, name, impl):
    import importlib

    shim = importlib.import_module(module)
    impl_mod = importlib.import_module(module.replace('skimage.', '_skimage2.', 1))
    impl_cls = getattr(impl_mod, impl)

    assert _unpickler().find_class(module, name) is impl_cls
    assert issubclass(getattr(shim, name), impl_cls)


def test_top_level_pickle_loads_api():
    import skimage as ski

    region = _sample_region()
    blob = ski.pickle_dumps(region, protocol=PROTOCOL)
    assert ski.pickle_loads(blob) == region


@pytest.mark.parametrize(
    'loads', [pickle_loads, _stdlib_pickle_loads], ids=['compat', 'stdlib']
)
@pytest.mark.parametrize(('filename', 'module', 'check'), GOLDEN_CASES)
def test_golden_0_25_pickle(
    loads, filename: str, module: str, check: Callable[[object], bool]
):
    blob = (GOLDEN_DIR / filename).read_bytes()
    assert module.encode() in blob
    assert check(loads(blob))


def test_registered_refs_cover_golden_types():
    refs = registered_pickle_refs()
    for module, qualname in (
        (REGIONPROPS_PATH, 'RegionProperties'),
        (GEOMETRIC_PATH, 'AffineTransform'),
        ('skimage.transform._thin_plate_splines', 'ThinPlateSplineTransform'),
        ('skimage.io.collection', 'ImageCollection'),
    ):
        assert PickleRef(module, qualname) in refs

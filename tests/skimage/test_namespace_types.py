"""Objects built by each namespace have that namespace's classes as types."""

import importlib
import inspect
import os

import numpy as np
import pytest

import _skimage2
import skimage
from _skimage2.transform._geometric import TRANSFORMS
from skimage._doctest_adapters import _IMPL_TO_SHIM


SRC = np.array(
    [
        [-12.3705, -10.5075],
        [-10.7865, 15.4305],
        [8.6985, 10.8675],
        [11.4975, -9.5715],
        [7.8435, 7.4835],
        [-5.3325, 6.5025],
        [6.7905, -6.3765],
        [-6.1695, -0.8235],
    ]
)
DST = np.array(
    [
        [0, 0],
        [0, 5800],
        [4900, 5800],
        [4900, 0],
        [4479, 4580],
        [1176, 3660],
        [3754, 790],
        [1024, 1931],
    ]
)
IMAGE = np.random.default_rng(0).random((8, 8, 3))
LABELS = np.arange(64).reshape(8, 8) // 16


def _camera_path(ns):
    return os.path.join(ns.data.data_dir, 'camera.png')


# Every v1 function and method that returns an object of a class that v1
# subclasses.  The v1 API does not grow, so this list stays complete.
# (case name, subpackage, class name, factory taking the namespace module)
CASES = [
    (
        'rag_mean_color',
        'graph',
        'RAG',
        lambda ns: ns.graph.rag_mean_color(IMAGE, LABELS),
    ),
    (
        'RAG.copy',
        'graph',
        'RAG',
        lambda ns: ns.graph.rag_mean_color(IMAGE, LABELS).copy(),
    ),
    (
        'rag_boundary',
        'graph',
        'RAG',
        lambda ns: ns.graph.rag_boundary(LABELS, IMAGE[..., 0]),
    ),
    (
        'imread_collection',
        'io',
        'ImageCollection',
        lambda ns: ns.io.imread_collection(_camera_path(ns)),
    ),
    (
        'imread_collection_wrapper',
        'io',
        'ImageCollection',
        lambda ns: ns.io.imread_collection_wrapper(ns.io.imread)(_camera_path(ns)),
    ),
    (
        'ImageCollection-slice',
        'io',
        'ImageCollection',
        lambda ns: ns.io.ImageCollection(_camera_path(ns))[0:1],
    ),
    (
        'regionprops',
        'measure._regionprops',
        'RegionProperties',
        lambda ns: ns.measure.regionprops(LABELS)[0],
    ),
    (
        'ransac',
        'measure',
        'LineModelND',
        lambda ns: ns.measure.ransac(SRC, ns.measure.LineModelND, 2, 1.0)[0],
    ),
    (
        'Affine+Euclidean',
        'transform',
        'ProjectiveTransform',
        lambda ns: ns.transform.AffineTransform() + ns.transform.EuclideanTransform(),
    ),
    (
        'PiecewiseAffine-affines',
        'transform',
        'AffineTransform',
        lambda ns: ns.transform.PiecewiseAffineTransform.from_estimate(
            SRC, DST
        ).affines[0],
    ),
    (
        'PiecewiseAffine-inverse_affines',
        'transform',
        'AffineTransform',
        lambda ns: ns.transform.PiecewiseAffineTransform.from_estimate(
            SRC, DST
        ).inverse_affines[0],
    ),
    (
        'Affine-inverse',
        'transform',
        'AffineTransform',
        lambda ns: ns.transform.AffineTransform(scale=2).inverse,
    ),
]
for _ttype, _cls in TRANSFORMS.items():
    _name = _cls.__name__
    CASES += [
        (
            f'estimate_transform-{_ttype}',
            'transform',
            _name,
            lambda ns, t=_ttype: ns.transform.estimate_transform(t, SRC, DST),
        ),
        (
            f'{_name}.from_estimate',
            'transform',
            _name,
            lambda ns, n=_name: getattr(ns.transform, n).from_estimate(SRC, DST),
        ),
    ]


# `skimage2` forwards to `_skimage2`; importing `skimage2` here would silence
# the import warning that `test_skimage2_modules_match` expects.
@pytest.mark.parametrize('ns', [skimage, _skimage2], ids=lambda m: m.__name__)
@pytest.mark.parametrize(
    'subpackage, cls_name, make', [pytest.param(*c[1:], id=c[0]) for c in CASES]
)
def test_factory_returns_namespace_class(ns, subpackage, cls_name, make):
    module = importlib.import_module(f'{ns.__name__}.{subpackage}')
    assert type(make(ns)) is getattr(module, cls_name)


SUBPACKAGES = [
    'color',
    'data',
    'draw',
    'exposure',
    'feature',
    'filters',
    'future',
    'graph',
    'io',
    'measure',
    'metrics',
    'morphology',
    'registration',
    'restoration',
    'segmentation',
    'transform',
    'util',
]


def _v1_classes():
    classes = {}
    for subpackage in SUBPACKAGES:
        module = importlib.import_module(f'skimage.{subpackage}')
        for name in module.__all__:
            obj = getattr(module, name)
            if inspect.isclass(obj):
                classes[f'{subpackage}.{name}'] = obj
    return classes


def _impl_class(cls):
    # Classes defined in `skimage` itself are their own implementation.
    return next((c for c in cls.__mro__ if c.__module__.startswith('_skimage2')), cls)


def test_v1_classes_are_v1():
    v2 = []  # Classes in v1 that are actually v2.
    for name, cls in _v1_classes().items():
        if cls.__module__.split('.')[0] == 'skimage':
            continue  # cls already v1
        # Exceptions stay shared, so that v1 code can catch them from v2 code.
        if not issubclass(cls, BaseException):
            v2.append(name)
    assert v2 == []


def test_v1_classes_keep_v2_hierarchy():
    classes = _v1_classes()
    lost = []  # Classes that are subclasses in v2, but not v1.
    for a, cls_a in classes.items():
        for b, cls_b in classes.items():
            if cls_a is cls_b:
                continue
            if not issubclass(_impl_class(cls_a), _impl_class(cls_b)):
                continue  # Not subclass in v2 namespace.
            if not issubclass(cls_a, cls_b):
                # Subclass in v2 namespace, but not in v1 namespace.
                lost.append(f'{a} -> {b}')
    assert lost == []


# Subclass relationships among public v1 classes.  The v1 API is frozen, so
# these must not change, whatever the `_skimage2` class hierarchy.
V1_SUBCLASSES = {
    ('graph.MCP_Connect', 'graph.MCP'),
    ('graph.MCP_Flexible', 'graph.MCP'),
    ('graph.MCP_Geometric', 'graph.MCP'),
    ('io.MultiImage', 'io.ImageCollection'),
    ('transform.AffineTransform', 'transform.ProjectiveTransform'),
    ('transform.EssentialMatrixTransform', 'transform.FundamentalMatrixTransform'),
    ('transform.EuclideanTransform', 'transform.ProjectiveTransform'),
    ('transform.SimilarityTransform', 'transform.EuclideanTransform'),
    ('transform.SimilarityTransform', 'transform.ProjectiveTransform'),
    # Structural: classes with the methods of the runtime-checkable protocol.
    ('measure.CircleModel', 'measure.RansacModelProtocol'),
    ('measure.EllipseModel', 'measure.RansacModelProtocol'),
    ('measure.LineModelND', 'measure.RansacModelProtocol'),
    ('transform.AffineTransform', 'measure.RansacModelProtocol'),
    ('transform.EssentialMatrixTransform', 'measure.RansacModelProtocol'),
    ('transform.EuclideanTransform', 'measure.RansacModelProtocol'),
    ('transform.FundamentalMatrixTransform', 'measure.RansacModelProtocol'),
    ('transform.PiecewiseAffineTransform', 'measure.RansacModelProtocol'),
    ('transform.PolynomialTransform', 'measure.RansacModelProtocol'),
    ('transform.ProjectiveTransform', 'measure.RansacModelProtocol'),
    ('transform.SimilarityTransform', 'measure.RansacModelProtocol'),
}


def test_v1_class_hierarchy_frozen():
    classes = _v1_classes()
    found = set()
    for a, cls_a in classes.items():
        for b, cls_b in classes.items():
            if cls_a is not cls_b and issubclass(cls_a, cls_b):
                found.add((a, b))
    assert found == V1_SUBCLASSES


def test_v1_shim_classes_findable():
    # Every public shim class is at its module and name, for pickle and doctests.
    _v1_classes()  # Import all v1 shim modules.
    lost = []
    for shim_class in _IMPL_TO_SHIM.values():
        if shim_class.__module__.startswith('skimage.tests'):
            continue  # Made by tests of the shim machinery.
        if shim_class.__qualname__.startswith('_'):
            continue  # Private classes need not be importable.
        module = importlib.import_module(shim_class.__module__)
        if getattr(module, shim_class.__qualname__, None) is not shim_class:
            lost.append(f'{shim_class.__module__}.{shim_class.__qualname__}')
    assert lost == []

"""Objects built by each namespace have that namespace's classes as types."""

import os

import numpy as np
import pytest

import _skimage2
import skimage
from _skimage2.transform._geometric import TRANSFORMS


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


# Factory paths that return objects whose classes have v1 shims. Keep this list
# aligned with the v1 factories as they are added.
# (case name, subpackage, class name, factory taking the namespace module)
CASES = [
    (
        'rag_mean_color',
        'graph',
        'RAG',
        lambda ns: ns.graph.rag_mean_color(IMAGE, LABELS),
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
    cls = getattr(getattr(ns, subpackage), cls_name)
    assert type(make(ns)) is cls

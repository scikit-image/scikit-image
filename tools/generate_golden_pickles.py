#!/usr/bin/env python3
"""Generate golden pickle fixtures from scikit-image 0.25.x.

Run inside an environment with ``scikit-image==0.25.2`` installed::

    pip install 'scikit-image==0.25.2'
    python tools/generate_golden_pickles.py

Fixtures are written to ``tests/skimage/data/pickles/``.
"""

from __future__ import annotations

import os
import pickle
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / 'tests' / 'skimage' / 'data' / 'pickles'
PROTOCOL = 4


def _write(name: str, obj) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    path.write_bytes(pickle.dumps(obj, protocol=PROTOCOL))
    print(f'wrote {path} ({path.stat().st_size} bytes)')


def main() -> None:
    from skimage import data
    from skimage.io import ImageCollection, imsave
    from skimage.measure import regionprops
    from skimage.transform import AffineTransform, ThinPlateSplineTransform

    label_image = np.zeros((10, 10), dtype=int)
    label_image[2:5, 2:5] = 1
    _write('regionproperties_0.25.pkl', regionprops(label_image)[0])

    _write('affine_transform_0.25.pkl', AffineTransform(scale=(1.0, 2.0)))

    src = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    tps = ThinPlateSplineTransform()
    tps.estimate(src, src + 0.1)
    _write('thin_plate_spline_transform_0.25.pkl', tps)

    with tempfile.TemporaryDirectory() as tmpdir:
        for index, image in enumerate((data.camera(), data.camera())):
            imsave(os.path.join(tmpdir, f'img{index}.png'), image)
        pattern = os.path.join(tmpdir, '*.png')
        _write('image_collection_0.25.pkl', ImageCollection(pattern))


if __name__ == '__main__':
    main()

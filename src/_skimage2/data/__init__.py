"""Example images and datasets.

A curated set of general purpose and scientific images used in tests, examples,
and documentation.

Newer datasets are no longer included as part of the package, but are
downloaded on demand. To make data available offline, use :func:`download_all`.

"""

import lazy_loader as _lazy

__getattr__, *_ = _lazy.attach_stub(__name__, __file__)


# Don't use the `__all__` and `__dir__` returned by `attach_stub`; those
# also advertise the bare v1 dataset names (e.g. `astronaut`), which stay
# importable as aliases of the `fetch_*()` functions but are not public
# API. Keep this list in sync with `__all__` in `__init__.pyi`.
__all__ = [
    'binary_blobs',
    'data_dir',
    'legacy_data_dir',
    'download_all',
    'fetch',
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
    'file_hash',
]


def __dir__():
    return __all__.copy()

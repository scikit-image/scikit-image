"""Example images and datasets.

A curated set of general purpose and scientific images used in tests, examples,
and documentation.

Newer datasets are no longer included as part of the package, but are
downloaded on demand. To make data available offline, use :func:`download_all`.

"""

import functools
import warnings
from collections.abc import Callable

import lazy_loader as _lazy

__getattr__, *_ = _lazy.attach_stub(__name__, __file__)
_stub_getattr = __getattr__

# Override `attach_stub`'s `__all__`/`__dir__` to hide the bare v1 names, which
# stay importable but are not public API. Keep in sync with `__init__.pyi`.
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

# Bare v1 names: deprecated aliases of the `fetch_*()` wrappers, dropped when
# `skimage` (v1) support is removed.
_DEPRECATED_FETCHERS = frozenset(
    name.removeprefix('fetch_') for name in __all__ if name.startswith('fetch_')
)

_deprecated_wrappers: dict[str, Callable] = {}


def _make_deprecation_wrapper(name, func):
    """Wrap a bare-named fetcher to warn about its `fetch_` replacement."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        warnings.warn(
            f"`skimage2.data.{name}` is deprecated in favor of "
            f"`skimage2.data.fetch_{name}` and will be removed when support "
            f"for `skimage` (v1) is dropped. Use `skimage2.data.fetch_{name}` "
            f"instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return func(*args, **kwargs)

    return wrapper


def _deprecated_wrapper(name):
    wrapper = _deprecated_wrappers.get(name)
    if wrapper is None:
        # Resolve through the canonical `fetch_*` name so `attach_stub` caches
        # that name, then cache the wrapper here so the raw bare name never
        # shadows it on later accesses.
        func = _stub_getattr(f'fetch_{name}')
        wrapper = _deprecated_wrappers[name] = _make_deprecation_wrapper(name, func)
        globals()[name] = wrapper
    return wrapper


def __getattr__(name):
    if name in _DEPRECATED_FETCHERS:
        return _deprecated_wrapper(name)
    return _stub_getattr(name)


# `attach_stub` eagerly imports every stub name when EAGER_IMPORT is set,
# binding the raw bare names before this module's `__getattr__` can wrap them.
for _name in _DEPRECATED_FETCHERS:
    if _name in globals():
        globals()[_name] = _deprecated_wrapper(_name)


def __dir__():
    return __all__.copy()

"""Pickle compatibility for the ``skimage`` / ``_skimage2`` megamove.

Maps stable public ``skimage.*`` import paths to ``_skimage2`` implementations so
pickles survive the internal package move. Docstring shims in
:mod:`skimage._docutils` stay separate and must not touch implementation metadata.
"""

from __future__ import annotations

import copyreg
import io
import pickle
from dataclasses import dataclass
from functools import partial
from typing import Any

__all__ = [
    'PickleRef',
    'SkimagePickler',
    'SkimageUnpickler',
    'apply_pickle_exports',
    'pickle_dump',
    'pickle_dumps',
    'pickle_load',
    'pickle_loads',
    'register_pickle_type',
    'registered_pickle_refs',
]

_REGISTRY: dict[PickleRef, Any] = {}  # public path -> implementation
_LEGACY: dict[PickleRef, PickleRef] = {}  # old public path -> current path
_EXPORTS: dict[str, dict[str, Any]] = {}  # shim module -> {name: implementation}
_REF_BY_TYPE: dict[type, PickleRef] = {}  # implementation type -> public path


@dataclass(frozen=True, slots=True)
class PickleRef:
    """Stable public ``(module, qualname)`` identity stored in pickle blobs."""

    module: str
    qualname: str

    def __str__(self) -> str:
        return f'{self.module}.{self.qualname}'


def register_pickle_type(
    ref: PickleRef,
    obj: Any,
    *,
    legacy: tuple[PickleRef, ...] = (),
) -> None:
    """Register ``obj`` under the stable public pickle path ``ref``."""
    _REGISTRY[ref] = obj
    _EXPORTS.setdefault(ref.module, {})[ref.qualname] = obj
    for old_ref in legacy:
        _LEGACY[old_ref] = ref
    if isinstance(obj, type):
        _REF_BY_TYPE[obj] = ref
        copyreg.pickle(obj, partial(_pickle_reduce, ref))


def registered_pickle_refs() -> tuple[PickleRef, ...]:
    """Return all registered public pickle paths."""
    return tuple(sorted(_REGISTRY, key=str))


def apply_pickle_exports(ns: dict[str, Any]) -> None:
    """Add registered exports missing from a shim module namespace.

    Existing names are left alone so docutils subclass proxies keep adapted
    docstrings.
    """
    module_name = ns.get('__name__')
    if not isinstance(module_name, str):
        return
    for name, obj in _EXPORTS.get(module_name, {}).items():
        if name not in ns:
            ns[name] = obj


def _resolve_ref(module: str, qualname: str) -> PickleRef:
    ref = PickleRef(module, qualname)
    return _LEGACY.get(ref, ref)


def _ref_for_type(cls: type) -> PickleRef | None:
    ref = _REF_BY_TYPE.get(cls)
    if ref is not None:
        return ref
    for reg_cls, reg_ref in _REF_BY_TYPE.items():
        if issubclass(cls, reg_cls):
            return reg_ref
    return None


def _extract_state(obj: Any) -> Any:
    if hasattr(obj, '__getstate__'):
        return obj.__getstate__()
    reduced = obj.__reduce_ex__(pickle.HIGHEST_PROTOCOL)
    if len(reduced) >= 3 and reduced[2] is not None:
        return reduced[2]
    msg = f'cannot extract pickle state from {type(obj)!r}'
    raise TypeError(msg)


def _pickle_reduce(ref: PickleRef, obj: Any):
    # PickleRef carries the public skimage.* path; rebuild via _restore_registered_instance.
    return _restore_registered_instance, (ref, _extract_state(obj))


def _restore_registered_instance(ref: PickleRef, state: Any) -> Any:
    """Rebuild a registered object from ``(ref, state)`` stored in a pickle."""
    cls = _REGISTRY[ref]
    obj = cls.__new__(cls)
    if isinstance(state, tuple):
        obj.__setstate__(*state)
    elif isinstance(state, dict):
        obj.__dict__.update(state)
    else:
        obj.__setstate__(state)
    return obj


class SkimagePickler(pickle.Pickler):
    """Pickler that emits stable public paths for registered types."""

    def reducer_override(self, obj: Any):
        ref = _ref_for_type(type(obj))
        if ref is None:
            return NotImplemented
        return _pickle_reduce(ref, obj)


class SkimageUnpickler(pickle.Unpickler):
    """Unpickler that resolves registered public paths to implementations."""

    def find_class(self, module: str, name: str) -> Any:
        ref = _resolve_ref(module, name)
        if ref in _REGISTRY:
            return _REGISTRY[ref]
        return super().find_class(module, name)


def _pickle_protocol(protocol: int | None) -> int:
    return pickle.DEFAULT_PROTOCOL if protocol is None else protocol


def pickle_dumps(obj: Any, *, protocol: int | None = None) -> bytes:
    """Pickle ``obj`` with :class:`SkimagePickler`."""
    buffer = io.BytesIO()
    SkimagePickler(buffer, protocol=_pickle_protocol(protocol)).dump(obj)
    return buffer.getvalue()


def pickle_dump(obj: Any, file, *, protocol: int | None = None) -> None:
    """Pickle ``obj`` to ``file`` with :class:`SkimagePickler`."""
    SkimagePickler(file, protocol=_pickle_protocol(protocol)).dump(obj)


def pickle_loads(data: bytes) -> Any:
    """Unpickle ``data`` with :class:`SkimageUnpickler`."""
    return SkimageUnpickler(io.BytesIO(data)).load()


def pickle_load(file) -> Any:
    """Unpickle from ``file`` with :class:`SkimageUnpickler`."""
    return SkimageUnpickler(file).load()


def _register(module: str, qualname: str, obj: Any) -> None:
    register_pickle_type(PickleRef(module, qualname), obj)


def _register_default_types() -> None:
    from _skimage2.io.collection import ImageCollection, MultiImage
    from _skimage2.measure._regionprops import RegionProperties
    from _skimage2.transform._geometric import (
        AffineTransform,
        EssentialMatrixTransform,
        EuclideanTransform,
        FundamentalMatrixTransform,
        PiecewiseAffineTransform,
        PolynomialTransform,
        ProjectiveTransform,
        SimilarityTransform,
    )
    from _skimage2.transform._thin_plate_splines import ThinPlateSplineTransform

    _register('skimage.measure._regionprops', 'RegionProperties', RegionProperties)

    geometric = 'skimage.transform._geometric'
    for cls in (
        AffineTransform,
        EssentialMatrixTransform,
        EuclideanTransform,
        FundamentalMatrixTransform,
        PiecewiseAffineTransform,
        PolynomialTransform,
        ProjectiveTransform,
        SimilarityTransform,
    ):
        _register(geometric, cls.__name__, cls)

    _register(
        'skimage.transform._thin_plate_splines',
        'ThinPlateSplineTransform',
        ThinPlateSplineTransform,
    )

    collection = 'skimage.io.collection'
    _register(collection, 'ImageCollection', ImageCollection)
    _register(collection, 'MultiImage', MultiImage)


_register_default_types()

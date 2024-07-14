import functools
import logging
from collections.abc import Callable
from types import ModuleType


__all__ = ["_dispatchable"]

_logger = logging.getLogger(__name__)

cp = None


def _setup_backends() -> dict[str, ModuleType]:
    """
    Setup available backends.

    Returns
    -------
    Dict[str, ModuleType]
        Available backends.
    """
    import skimage as ski

    backends = {
        "numpy": ski,
    }

    try:
        global cp
        import cupy as cp

        try:
            import cucim.skimage as cu_ski

            backends["cupy"] = cu_ski

        except ImportError:
            _logger.info("Cupy found but cuCIM not found. Ignoring cupy backend.")

    except (ImportError, ModuleNotFoundError):
        pass

    return backends


BACKENDS = _setup_backends()


def _get_submodule(module: ModuleType, submodule_path: str) -> ModuleType:
    """
    Get submodule from a module.

    Example:
    >>> import skimage
    >>> _get_submodule(skimage, "morphology.binary")

    Parameters
    ----------
    module : ModuleType
        Parent module.
    submodule_path : str
        Submodule path, for example "morphology.binary".

    Returns
    -------
    ModuleType
        Requested submodule.
    """
    submodules = submodule_path.split(".")

    while submodules:
        module = getattr(module, submodules.pop(0))

    return module


def _is_valid_cucim_backend(*args, **kwargs) -> bool:
    """
    Check if the cuCIM backend is valid.

    Parameters
    ----------
    args : Any
        Arguments.
    kwargs : Any
        Keyword arguments.

    Returns
    -------
    bool
        True if the cuCIM backend is valid.

    Raises
    ------
    ValueError
        If the arguments are not on a single backend.
    """
    if cp is None:
        return False

    if cp is not None:  # cupy not found
        #  check if has an array interface, it could be numpy, dask, zarr, etc.
        n_arrs = sum(
            [hasattr(a, "__array__") for a in args]
            + [hasattr(v, "__array__") for v in kwargs.values()]
        )
        n_cu_arrs = sum(
            [isinstance(a, cp.ndarray) for a in args]
            + [isinstance(v, cp.ndarray) for v in kwargs.values()]
        )
        if n_cu_arrs > 0 and n_cu_arrs != n_arrs:
            raise ValueError(
                "All arguments must be on the same backend. "
                f"Found {n_arrs} arrays with {n_cu_arrs} being cupy arrays."
            )

    # if everything is numpy, then use the numpy backend
    return n_cu_arrs > 0


def _dispatchable(func: Callable) -> Callable:
    """
    Decorator to dispatch a function to different backends depending on the input types.

    Parameters
    ----------
    func : Callable
        Function to be run with different backends.

    Returns
    -------
    Callable
        Backend compatible function.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        backend = "numpy"

        if _is_valid_cucim_backend(*args, **kwargs):
            backend = "cupy"
            _logger.debug(f"Using cupy backend for '{func.__name__}'")

        _, submodule_path = func.__module__.split(".", maxsplit=1)

        submodule = _get_submodule(BACKENDS[backend], submodule_path)

        # IMPORTANT: otherwise it does infinite recursion
        if backend == "numpy":
            return func(*args, **kwargs)

        return getattr(submodule, func.__name__)(*args, **kwargs)

    if not hasattr(func, "__name__"):
        # if it instead a class
        wrapper.__name__ = func.__class__.__name__

    return wrapper

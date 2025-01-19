import functools
from importlib.metadata import entry_points
from functools import cache
import os
import warnings


def get_backend_priority():
    """Returns the backend priority list, or `False` if the dispatching is disabled.

    The function interprets the value of the environment variable 
    `SKIMAGE_BACKENDS` as follows:
    - If unset or explicitly `False`, return `False`.
    - If a comma-separated string, return it as a list of backend names.
    - If a single string, return it as a list with that single backend name.
    """
    backend_priority = os.environ.get("SKIMAGE_BACKENDS", False)

    if backend_priority in ["False", False]:
        return False
    elif "," in backend_priority:
        return [item.strip() for item in backend_priority.split(",")]
    else:
        return [backend_priority,]


def public_api_name(func):
    """Get the name of the public module for a scikit-image function.

    This computes the name of the public submodule in which the function can
    be found.
    """
    full_name = func.__module__
    # This relies on the fact that scikit-image does not use
    # sub-submodules in its public API, except in one case.
    # This means that public name can be atmost `skimage.foobar`
    # for everything else

    sub_submodules = ["skimage.filters.rank"]
    candidates = [name for name in sub_submodules if full_name.startswith(name)]
    if len(candidates) == 0:
        # Assume first two parts of the name are where the function is in our public API
        parts = full_name.split(".")
        if len(parts) <= 2:
            msg = f"expected {func.__module__=} with more than 2 dot-delimited parts"
            raise ValueError(msg)
        public_name = ".".join(parts[:2])
    elif len(candidates) == 1:
        public_name = candidates[0]
    else:
        msg = f"{func!r} matches more than one sub-submodule: {candidates!r}"
        raise ValueError(msg)

    # It would be nice to sanity check things by doing something like the
    # following. However we can't because this code is executed while the
    # module is being imported, which means this would create a circular
    # import
    # mod = importlib.import_module(public_name)
    # assert getattr(mod, func.__name__) is func

    return public_name


@cache
def all_backends():
    """List all installed backends and information about them"""
    backends = {}
    backends_ = entry_points(group="skimage_backends")
    backend_infos = entry_points(group="skimage_backend_infos")

    for backend in backends_:
        backends[backend.name] = {"implementation": backend}
        try:
            info = backend_infos[backend.name]
            # Double () to load and then call the backend information function
            backends[backend.name]["info"] = info.load()()
        except KeyError:
            pass

    return backends


def dispatchable(func):
    """Mark a function as dispatchable.

    When a decorated function is called the installed backends are
    searched for an implementation. If no backend implements the function
    then the scikit-image implementation is used.
    """
    func_name = func.__name__
    func_module = public_api_name(func)

    # If no backends are installed or dispatching is disabled,
    # return the original function.
    if not all_backends():
        if get_backend_priority():
            # no installed backends but `SKIMAGE_BACKENDS` is not False
            warnings.warn(
                f"Call to '{func_module}:{func_name}' was not dispatched."
                " No backends installed and SKIMAGE_BACKENDS is not 'False'."
                " Falling back to scikit-image.",
                DispatchNotification,
                stacklevel=2,
            )
        return func

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        backend_priority = get_backend_priority()
        installed_backends = all_backends()
        for backend_name in backend_priority:
            if backend_name not in installed_backends:
                continue
            backend = installed_backends[backend_name]
            # Check if the function we are looking for is implemented in
            # the backend
            if f"{func_module}:{func_name}" not in backend["info"].supported_functions:
                continue

            backend_impl = backend["implementation"].load()

            # Allow the backend to accept/reject a call based on the function
            # name and the arguments
            wants_it = backend_impl.can_has(
                f"{func_module}:{func_name}", *args, **kwargs
            )
            if not wants_it:
                continue

            func_impl = backend_impl.get_implementation(f"{func_module}:{func_name}")
            warnings.warn(
                f"Call to '{func_module}:{func_name}' was dispatched to"
                f" the '{backend_name}' backend. Set SKIMAGE_BACKENDS='False' to"
                " disable dispatching.",
                DispatchNotification,
                # XXX from where should this warning originate?
                # XXX from where the function that was dispatched was called?
                # XXX or from where the user called a function that called
                # XXX a function that was dispatched?
                stacklevel=2,
            )
            return func_impl(*args, **kwargs)

        else:
            if backend_priority:
                warnings.warn(
                    f"Call to '{func_module}:{func_name}' was not dispatched."
                    " All backends rejected the call. Falling back to scikit-image",
                    DispatchNotification,
                    stacklevel=2,
                )
            return func(*args, **kwargs)

    return wrapper


class BackendInformation:
    """Information about a backend

    A backend that wants to provide additional information about itself
    should return an instance of this from its information entry-point.
    """

    def __init__(self, supported_functions):
        self.supported_functions = supported_functions


class DispatchNotification(RuntimeWarning):
    """Notification issued when a function is dispatched to a backend."""

    pass

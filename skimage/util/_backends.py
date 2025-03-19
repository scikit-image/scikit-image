import functools
from importlib.metadata import entry_points
from functools import cache
import os
import warnings


__all__ = [
    "set_backends",
]


class set_backends:
    _active_instance = None

    def __init__(self, *backends, dispatch=False):
        if not isinstance(dispatch, bool):
            raise ValueError(
                f"Invalid value for 'dispatch': {dispatch}. Expected True or False."
            )

        self.dispatch = dispatch
        self.backend_priority = list(backends) if backends else None

        set_backends._previous_instance = None  # for nested context managers
        set_backends._active_instance = self

    def __enter__(self):
        self._previous_instance = set_backends._active_instance
        set_backends._active_instance = self
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        set_backends._active_instance = self._previous_instance

    @classmethod
    def get_dispatch_and_priority(cls):
        env_var_dispatch = get_skimage_dispatching()
        env_var_priority = get_skimage_backend_priority()

        if cls._active_instance:
            if env_var_dispatch or env_var_priority:
                warnings.warn(
                    "`set_backends` instance is currently active. Ignoring values in"
                    "`SKIMAGE_DISPATCHING` and `SKIMAGE_BACKEND_PRIORITY`.",
                    DispatchNotification,
                    stacklevel=3,
                )
            return cls._active_instance.dispatch, cls._active_instance.backend_priority
        else:
            return env_var_dispatch, env_var_priority

    @classmethod
    def delete_active_instance(cls):
        """Deletes the currently active instance."""
        if cls._active_instance:
            del cls._active_instance
            cls._active_instance = None


def get_skimage_dispatching():
    """Returns the value of the `SKIMAGE_DISPATCHING` environment variable."""
    dispatch_flag = os.environ.get("SKIMAGE_DISPATCHING", False)
    if dispatch_flag in ["False", False]:
        return False
    elif dispatch_flag in ["True", True]:
        return True
    else:
        warnings.warn(
            f"Invalid value for SKIMAGE_DISPATCHING: {dispatch_flag}."
            "Expected 'True' or 'False'."
            f"Setting SKIMAGE_DISPATCHING to 'False'.",
            DispatchNotification,
            stacklevel=4,
        )
        return False


def get_skimage_backend_priority():
    """Returns the backend priority list stored in `SKIMAGE_BACKEND_PRIORITY`
    environment variable, or `False`.

    This function interprets the value of the environment variable
    `SKIMAGE_BACKEND_PRIORITY` as follows:
    - If unset or explicitly set to `"False"`, return `False`.
    - If a comma-separated string, return it as a list of backend names.
    - If a single string, return it as a list with that single backend name.
    """
    backend_priority = os.environ.get("SKIMAGE_BACKEND_PRIORITY", False)

    if backend_priority in ["False", False]:
        return False
    elif "," in backend_priority:
        return [item.strip() for item in backend_priority.split(",")]
    else:
        return [
            backend_priority,
        ]


def public_api_module(func):
    """Returns the public module in which the given skimage `func` is present.

    Since scikit-image does not use sub-submodules in its public API
    (except `skimage.filters.rank`), the function infers the public module name
    based on the function's module path.

    Parameters
    ----------
    func : function
        A function from the scikit-image library.

    Returns
    -------
    public_name : str
        The name of the public module in scikit-image where the `func` resides.
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
def all_backends_with_eps_combined():
    """Returns a dictionary with all the installed scikit-image backends and the infos
    stored in their two entry-points.

    Returns
    -------
    backends : dict
        A dictionary where keys are backend names, and values are dictionaries with:
        - `skimage_backends_ep_obj` : EntryPoint
          The backend's entry point object from the `skimage_backends` group.
        - `info` : object
          `BackendInformation` object stored in the `skimage_backend_infos` entry-point.

    For example::

        {
            'backend1': {
                'skimage_backends_ep_obj': EntryPoint(...),
                'info': <BackendInformation object at ...>
            },
            ...
        }

    """
    backends = {}
    backends_ = entry_points(group="skimage_backends")
    backend_infos = entry_points(group="skimage_backend_infos")

    for backend in backends_:
        backends[backend.name] = {"skimage_backends_ep_obj": backend}
        info = backend_infos[backend.name]
        # Only loading and calling the infos ep bcoz it is
        # assumed to be cheap operation --> saves time
        backends[backend.name]["info"] = info.load()()

    return backends


def dispatchable(func):
    """Mark a function as dispatchable.

    When a decorated function is called, the installed backends are
    searched for an implementation. If no backend implements the function
    then the scikit-image implementation is used.
    """
    func_name = func.__name__
    func_module = public_api_module(func)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        dispatch, backend_priority = set_backends.get_dispatch_and_priority()
        if not dispatch:
            return func(*args, **kwargs)

        installed_backends = all_backends_with_eps_combined()

        if not installed_backends:
            # no backends installed falling back to scikit-image
            warnings.warn(
                f"Call to '{func_module}:{func_name}' was not dispatched."
                " No backends installed and `SKIMAGE_DISPATCHING` is set to"
                f"'{dispatch}'. Falling back to scikit-image.",
                DispatchNotification,
                stacklevel=2,
            )
            return func(*args, **kwargs)

        if not backend_priority:
            # backend priority is not set; using default priority--
            # i.e. backend names sorted in alphabetical order
            default_backend_priority = sorted(installed_backends.keys())
            warnings.warn(
                f"`SKIMAGE_BACKEND_PRIORITY` was set to {backend_priority}. Defaulting to priority: "
                f"'{default_backend_priority}'. Use `SKIMAGE_BACKEND_PRIORITY` to set a custom backend priority.",
                DispatchNotification,
                stacklevel=2,
            )
            backend_priority = default_backend_priority

        for backend_name in backend_priority:
            if backend_name not in installed_backends:
                continue
            backend = installed_backends[backend_name]
            # Check if the function we are looking for is implemented in
            # the backend
            if f"{func_module}:{func_name}" not in backend["info"].supported_functions:
                continue

            backend_impl = backend["skimage_backends_ep_obj"].load()

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
                f" the '{backend_name}' backend. Set SKIMAGE_DISPATCHING='False' to"
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
                    " All backends rejected the call. Falling back to scikit-image."
                    f" Installed backends : {list(installed_backends.keys())}",
                    DispatchNotification,
                    stacklevel=2,
                )

            return func(*args, **kwargs)

    return wrapper


class BackendInformation:
    """To store the information about a backend.

    An instance of this class is expected to be returned by the
    `skimage_backend_infos` entry-point.

    Parameters
    ----------
    supported_functions : list of strings
        A list of all the functions supported by a backend. The functions are
        present in the list as strings of the form `"public_module_name:func_name"`.
        For example: `["skimage.metrics:mean_squared_error", ...]`.

    In future, a backend would be able to provide more additional information
    about itself.
    """

    def __init__(self, supported_functions):
        self.supported_functions = supported_functions


class DispatchNotification(RuntimeWarning):
    """This type of runtime warning is issued when a function is dispatched to
    a backend."""

    pass

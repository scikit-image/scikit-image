import functools
from importlib.metadata import entry_points
from functools import cache
import os
import warnings


def get_skimage_backends():
    """Returns the backend priority list stored in `SKIMAGE_BACKENDS`
    environment variable, or `False` if the dispatching is disabled.

    This function interprets the value of the environment variable 
    `SKIMAGE_BACKENDS` as follows:
    - If unset or explicitly set to `"False"`, return `False`.
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

    When a decorated function is called the installed backends are
    searched for an implementation. If no backend implements the function
    then the scikit-image implementation is used.
    """
    func_name = func.__name__
    func_module = public_api_name(func)

    # If no backends are installed or dispatching is disabled,
    # return the original function.
    if not all_backends_with_eps_combined():
        if get_skimage_backends():
            # no installed backends but `SKIMAGE_BACKENDS` is not False
            warnings.warn(
                f"Call to '{func_module}:{func_name}' was not dispatched."
                " No backends installed and SKIMAGE_BACKENDS is not 'False'."
                " Falling back to scikit-image.",
                DispatchNotification,
                stacklevel=2,
            )
        return func
    elif not get_skimage_backends():
        # backends installed but `SKIMAGE_BACKENDS` is False
        return func

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        backend_priority = get_skimage_backends()
        installed_backends = all_backends_with_eps_combined()
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

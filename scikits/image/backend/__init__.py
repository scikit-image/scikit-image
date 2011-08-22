import os, sys
import scikits.image
from scikits.image import log
import warnings
import ast
from nose.plugins.skip import SkipTest

class BackendTester(object):
    """
    Base class for nose backend testing.
    """
    def wrapped(self, function, backend):
        try:
            use_backend(backend)
            function()
        except ImportError:
            raise SkipTest
    
    def test_all_backends(self):
        for backend in scikits.image.backends:
            if backend == "default": 
                continue
            for function_name in dir(self):
                if function_name.startswith("test") and function_name != "test_all_backends":
                    yield (self.wrapped, getattr(self, function_name), backend)


class BackendManager(object):
    """
    Backend manager handles backend registry and switching.
    """
    def __init__(self):
        # add default backend to the namespace
        scikits.image.backends = ["default"]
        self.current_backend = "default"
        self.fallback_backends = []
        self.backend_listing = {}
        self.backend_modules = {}
        self.backend_imported = {}

    def register(self, backend=None, module=None, source=None, functions=[], unlisted=False):
        """
        Registers backend support for a function or functions in a specified module.
        
        Parameters
        ----------
            backend: str
                Name of backend.
            module: str
                Name of module that is supported.
            source: str
                Backend implementation module located in the backends subdirectory.
            functions: list of str
                Functions to be registered.
            unlisted: boolean, optional
                Omit backend from listing (used in testing).
        """
        if not backend:
            raise ValueError("Specify backend name.")
        if not module:
            raise ValueError("Specify module name.")
        backend_name = backend
        module_elements = module.split(".")
        backend_module_str = ".".join([module, "backend"])
        if source:
            backend_module_str += "." + source
        module_name = module
        for function_name in functions:
            function_name = function_name.split(".")
            ending_module = function_name[:-1]
            if ending_module:
                backend_module_string = ".".join([backend_module_str] + ending_module)
            else:
                backend_module_string = backend_module_str
            if module_name not in self.backend_listing:
                # initialize default backend
                self.backend_listing[module_name] = {"default" : {}}
                self.backend_modules[module_name] = {"default" : {}}
            if backend_name not in self.backend_listing[module_name]:
                self.backend_listing[module_name][backend_name] = {}
                self.backend_modules[module_name][backend_name] = {}
            self.backend_listing[module_name][backend_name][function_name[-1]] = None
            self.backend_modules[module_name][backend_name][function_name[-1]] = backend_module_string
            self.backend_imported[backend_module_string] = False
    
        if not unlisted:
            if backend_name not in scikits.image.backends:
                scikits.image.backends.append(backend_name)
            
    def _scan_backends(self):
        """
        Scans through the source tree to extract all available backends from file names.
        """        
        root = "scikits.image"
        location = os.path.split(sys.modules[root].__file__)[0]
        backends = []
        # visit each backend directory in every scikits.image submodule
        for f in os.listdir(location):
            submodule = os.path.join(location, f)
            if os.path.isdir(submodule):
                submodule_dir = submodule
                module_name = root + "." + f
                backend_dir = os.path.join(location, f, "backend")
                if os.path.exists(backend_dir):
                    try:
                        __import__(module_name + ".backend", fromlist=[module_name])
                    except ImportError:
                        pass
                        
    def _ensure_backend_loaded(self, backend_name, module_name, function_name):
        """
        Ensures a backend is imported.
        
        Parameters
        ----------
            backend_name: str
                Name of backend.
            module_name: str
                Name of function module.
            function_name: str
                Function name.
        Returns
        -------
            output: boolean
                Backend supports for a function.
        """
        # check if backend has been imported and if not do so
        if not (backend_name in self.backend_listing[module_name] and\
            function_name in self.backend_listing[module_name][backend_name]):
            return False
        
        if not self.backend_listing[module_name][backend_name][function_name]:
            module_location = self.backend_modules[module_name][backend_name][function_name]
            if not self.backend_imported[module_location]:
                module = __import__(module_location, fromlist=[module_location])
                self.backend_imported[module_location] = True
                for f_name in self.backend_listing[module_name][backend_name]:
                    self.backend_listing[module_name][backend_name][f_name] = \
                        getattr(module, f_name)
        return True
                        
    def use_backend(self, backend=None):
        """
        Selects a new backend and update modules as needed.
        
        Parameters
        ----------
        backend: str, optional
            Selects a new backend globally.
            Default backend is selected if None or if the parameter is omitted.
        """
        if isinstance(backend, list):
            if backend:            
                current_backend = backend[0]
                fallback_backends = backend[1:]
            else:
                current_backend = "default"                
        else:
            if backend == None:
                current_backend = "default"
            else:
                current_backend = backend
            fallback_backends = []
        self.current_backend = current_backend
        self.fallback_backends = fallback_backends

    def backing(self, function):
        """
        Returns the backends that supports a function.
        
        Parameters
        ----------
            function: function type 
                Queried function.
        """
        module_name = ".".join(function.__module__.split('.')[:3])
        backends = []
        if module_name in self.backend_listing:
            for backend in self.backend_listing[module_name]:
                if function.__name__ in self.backend_listing[module_name][backend]:
                    backends.append(backend)
        return backends


def add_backends(function):
    """
    A decorator that adds backend support to a function.
    """
    function_name = function.__name__
    module_name = ".".join(function.__module__.split(".")[:3])
    if module_name not in _manager.backend_listing:
        # initialize default backend
        _manager.backend_listing[module_name] = {"default" : {}}
        _manager.backend_modules[module_name] = {"default" : {}}    
    # register default implementation               
    listing = _manager.backend_listing[module_name]
    listing["default"][function_name] = function
    # add documentation to function doc strings
    if len(listing) > 1:
        if not function.__doc__:
            function.__doc__ = ""
        else:
            function.__doc__ += "\n"
        function.__doc__ += "    Backends supported:\n"
        function.__doc__ += "    -------------------\n"
        for backend in listing:
            if backend == "default" or function_name not in listing[backend]:
                continue
            function.__doc__ += "    %s\n" % backend
            function.__doc__ += "       See also: %s\n" % _manager.backend_modules[module_name][backend][function_name]
                                
    def wrapper(*args, **kwargs):
        if "backend" in kwargs:
            backend = kwargs.get("backend")
            if backend == None:
                backend = "default"
            del kwargs["backend"]
        else:
            backend = _manager.current_backend
        # fall back to default if backend not supported
        if not _manager._ensure_backend_loaded(backend, module_name, function_name):
            for fallback in _manager.fallback_backends:
                if fallback in listing and \
                function_name in listing[fallback]:
                    backend = fallback
                    log.warn("Falling back to %s implementation" % fallback)
                    # make sure backend imported
                    if _manager._ensure_backend_loaded(backend, module_name, function_name):
                        break                
            else:
                log.warn("Falling back to default implementation from backend %s" % backend)
                backend = "default"
        # execute the backend function and return result
        return listing[backend][function_name](*args, **kwargs)
    
    wrapper.__doc__ = function.__doc__
    wrapper.__module__ = function.__module__        
    wrapper.__name__ = function_name
    return wrapper

_manager = BackendManager()
register = _manager.register
use_backend = _manager.use_backend        
backing = _manager.backing
_manager._scan_backends()

@add_backends
def _test1():
    """
    Test function test1 documentation
    """
    return "default test1"

@add_backends
def _test2():
    """
    Test function test2 documentation
    """    
    return "default test2"

@add_backends
def _test3():
    """
    Test function test3 documentation
    """    
    return "default test3"

__all__ = ["add_backends", "use_backend", "backing", "BackendTester", "register"]


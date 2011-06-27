import os, sys
from scikits.image import log
import scikits.image.backends
import warnings
import ast

class ModuleParser(ast.NodeVisitor):
    """
    Parser that extracts all defined methods from source without importing.
    """
    def parse(self, code):
        """
        Parses source code and visit definitions.
        """
        tree = ast.parse(code)
        self.functions = []
        self.visit(tree)
        return self.functions
    
    def visit_FunctionDef(self, statement):
        """
        Function visitation of parser.
        """
        self.functions.append(statement.name)


class BackendTester():
    def test_all_backends(self):
        for backend in scikits.image.backends.list:
            if backend == "default": 
                continue
            for function_name in dir(self):
                if function_name.startswith("test") and function_name != "test_all_backends":
                    yield (getattr(self, function_name), backend)


def import_backend(backend, module_name):
    """
    Imports the backend counterpart of a module.
    """
    print "importing", backend
    mods = module_name.split(".")
    module_name = ".".join(mods[:-1] + ["backend"] + [mods[-1]])
    name = module_name + "_%s" % backend
    try:
        return __import__(name, fromlist=[name])
    except ImportError:
        return None


class BackendManager(object):
    """
    Backend manager handles backend registry and switching.
    """
    def __init__(self):
        # add default backend to the namespace
        mod = sys.modules["scikits.image.backends"]
        scikits.image.backends.list = ["default"]
        setattr(mod, "default", "default")        
        self.current_backend = "default"
        self.fallback_backends = []
        self.backend_listing = {}
        self.backend_imported = {}
        self.module_members = {}        
        self.scan_backends()
        self.parser = ModuleParser()
    
    def scan_backends(self):
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
                    submodule_files = [f for f in os.listdir(submodule_dir) \
                        if os.path.isfile(os.path.join(submodule_dir, f)) and f.endswith(".py")]
                    backend_files = [f for f in os.listdir(backend_dir) \
                        if os.path.isfile(os.path.join(backend_dir, f)) and f.endswith(".py")]
                    # math file in backend directory with file in parent directory
                    for f in backend_files:
                        split = f.split("_")
                        backend = split[-1][:-3]
                        target = "_".join(split[:-1])
                        if target + ".py" in submodule_files:
                            if backend not in backends:
                                backends.append(backend)
                            mod_name = module_name + "." + target
                            if mod_name not in self.backend_listing:
                                # initialize default default backend
                                self.backend_listing[mod_name] = {}
                                self.backend_listing[mod_name]["default"] = {}
                                self.backend_imported[mod_name] = {}
                                self.backend_imported[mod_name]["default"] = True
                            self.backend_listing[mod_name][backend] = {}
                            self.backend_imported[mod_name][backend] = False
        # create references for each backend in backends namespace
        backends_mod = sys.modules["scikits.image.backends"]
        for backend_name in backends:
            setattr(backends_mod, backend_name, backend_name)
            scikits.image.backends.list.append(backend_name)

    def ensure_backend_loaded(self, backend, module=None):
        """
        Ensures a backend is imported.
        """
        if module:
            modules = [module]
        else:
            modules = self.backend_imported.keys()
        for module_name in modules:
            # check if backend has been imported and if not do so
            if backend in self.backend_imported[module_name] \
            and not self.backend_imported[module_name][backend]:
                backend_module = import_backend(backend, module_name)
                self.backend_imported[module_name][backend] = True
                for function_name in self.backend_listing[module_name][backend]:
                    self.backend_listing[module_name][backend][function_name] = \
                        getattr(backend_module, function_name)
                        
    def use_backend(self, backend):
        """
        Selects a new backend and update modules as needed.
        """
        if isinstance(backend, list):
            if backend:            
                self.current_backend = backend[0]
                self.fallback_backends = backend[1:]
            else:
                self.current_backend = "default"                
        else:
            self.current_backend = backend
            self.fallback_backends = []
        self.ensure_backend_loaded(self.current_backend)

    def backing(self, function):
        module_name = function.__module__
        backends = []
        if module_name in self.backend_listing:
            for backend in self.backend_listing[module_name]:
                if function.__name__ in self.backend_listing[module_name][backend]:
                    backends.append(backend)
        return backends

    def scan_backend_functions(self, module_name):
        """
        Scans through the registered backends of a module and extract the defined functions
        """
        module_path = os.path.split(sys.modules[module_name].__file__)[0]
        main_name = module_name.split('.')[-1]
        functions = {}
        for backend in self.backend_listing[module_name]:
            if backend != "default":
                backend_path = os.path.join(module_path, "backend", main_name + "_" + backend + ".py")
                #functions[backend] = self.parser.parse(open(backend_path).read())
                functions[backend] = compile(open(backend_path).read(), '', mode='exec').co_names
        return functions

    def backend_function_name(self, function, backend):
        module_elements = function.__module__.split(".")
        return ".".join(module_elements[:-1] + ["backend"] + \
            [module_elements[-1] + "_" + backend] + [function.__name__])
    
    def register_function(self, module_name, function):
        """
        Register functions for a specific module
        """
        function_name = function.__name__
        if module_name not in self.backend_listing:
            self.backend_listing[module_name] = {}
            self.backend_listing[module_name]["default"] = {}
        # parse backend files and initialize implemented functions
#        if len(self.backend_listing[module_name]["default"]) == 0:
#            functions = self.scan_backend_functions(module_name)
#            for backend, backend_functions in functions.items():
#                for backend_function in backend_functions:
#                    if backend_function in funcs:
#                        print ">", backend_function
#                        self.backend_listing[module_name][backend][backend_function] = None
#                print self.backend_listing[module_name][backend]

        if module_name not in self.module_members:
            self.module_members[module_name] = self.scan_backend_functions(module_name)
        # register default implementation
        self.backend_listing[module_name]["default"][function_name] = function
        for backend, members in self.module_members[module_name].items():
            if function_name in members:
                self.backend_listing[module_name][backend][function_name] = None
        
        self.ensure_backend_loaded(self.current_backend)
        # if current backend is other than default, do the required backend imports
#        if not self.backend_imported[module_name][self.current_backend]:
#            # register backend function
#            backend_module = import_backend(self.current_backend, module_name)
#            self.backend_imported[module_name][self.current_backend] = True
#            self.backend_listing[module_name][self.current_backend][function_name] = \
#                    getattr(backend_module, function_name)
          
        
class backend_function(object):
    """
    A decorator that adds backend support to a function.
    """
    def __init__(self, function):
        self.function = function
        self.function_name = function.__name__
        self.module_name = function.__module__
        # iterate through backend directory and find backends that match
        manager.register_function(self.module_name, function)
        # add documentation to function doc strings
        if len(manager.backend_listing[self.module_name]) > 1:
            if not function.__doc__:
                function.__doc__ = ""
            else:
                function.__doc__ += "\n"
            function.__doc__ += "    Backends supported:\n"
            function.__doc__ += "    -------------------\n"
            for backend in manager.backend_listing[self.module_name]:
                if backend == "default":
                    continue
                function.__doc__ += "    %s\n" % backend
                function.__doc__ += "       See also: %s\n" % manager.backend_function_name(function, backend)
        self.__doc__ = function.__doc__
        self.__module__ = function.__module__        
        self.__name__ = function.__name__ 
        self.func_name = function.func_name
                                
    def __call__(self, *args, **kwargs):
        if "backend" in kwargs:
            backend = kwargs.get("backend")
            manager.ensure_backend_loaded(backend, module=self.module_name)
            del kwargs["backend"]
        else:
            backend = manager.current_backend
        # fall back to default if backend not supported
        if backend not in manager.backend_listing[self.module_name] or \
        self.function_name not in manager.backend_listing[self.module_name][backend]:
            for fallback in manager.fallback_backends:
                if fallback in manager.backend_listing[self.module_name] and \
                self.function_name in manager.backend_listing[self.module_name][fallback]:
                    backend = fallback
                    log.warn("Falling back to %s implementation" % fallback)
                    # make sure backend imported
                    manager.register_backend(backend)
                    break
            else:
                log.warn("Falling back to default implementation from backend %s" % backend)
                backend = "default"
        return manager.backend_listing[self.module_name][backend][self.function_name](*args, **kwargs)

def add_backends(function):
    """
    A decorator that adds backend support to a function.
    """
    function_name = function.__name__
    module_name = function.__module__
    # iterate through backend directory and find backends that match
    manager.register_function(module_name, function)
    # add documentation to function doc strings
    if len(manager.backend_listing[module_name]) > 1:
        if not function.__doc__:
            function.__doc__ = ""
        else:
            function.__doc__ += "\n"
        function.__doc__ += "    Backends supported:\n"
        function.__doc__ += "    -------------------\n"
        for backend in manager.backend_listing[module_name]:
            if backend == "default":
                continue
            function.__doc__ += "    %s\n" % backend
            function.__doc__ += "       See also: %s\n" % manager.backend_function_name(function, backend)
                                
    def wrapper(*args, **kwargs):
        if "backend" in kwargs:
            backend = kwargs.get("backend")
            if not backend:
                backend = "default"
            manager.ensure_backend_loaded(backend, module=module_name)
            del kwargs["backend"]
        else:
            backend = manager.current_backend
        # fall back to default if backend not supported
        if backend not in manager.backend_listing[module_name] or \
        function_name not in manager.backend_listing[module_name][backend]:
            for fallback in manager.fallback_backends:
                if fallback in manager.backend_listing[module_name] and \
                function_name in manager.backend_listing[module_name][fallback]:
                    backend = fallback
                    log.warn("Falling back to %s implementation" % fallback)
                    # make sure backend imported
                    manager.register_backend(backend)
                    break
            else:
                log.warn("Falling back to default implementation from backend %s" % backend)
                backend = "default"
        return manager.backend_listing[module_name][backend][function_name](*args, **kwargs)
    
    wrapper.__doc__ = function.__doc__
    wrapper.__module__ = function.__module__        
    return wrapper

manager = BackendManager()
use_backend = manager.use_backend        
backing = manager.backing


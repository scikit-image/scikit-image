import os, sys


def import_backend(backend, module_name):
    mods = module_name.split(".")
    module_name = ".".join(mods[:-1] + ["backend"] + [mods[-1]])
    name = module_name + "_%s" % backend
    try:
        return __import__(name, fromlist=[name])
    except ImportError:
        return None


class BackendManager(object):
    def __init__(self):
        self.backends = []
        self.add_backend("numpy")
        self.backend_listing = {}
        
    def add_backend(self, backend_name):
        self.backends.append(backend_name)
        setattr(self, backend_name, backend_name)

    def use_backend(self, backend):
        self.current_backend = backend
            
#    def use_backend(self, backend):
#        self.current_backend = backend
#        for module_name, backends in self.backend_listing.items():
#            if backend not in backends:
#                # import backend module and register its functions
#                backend_module = import_backend(backend, module_name)
#                if backend_module:
#                    self.backend_listing[module_name][backend] = {}
#                    # iterate through module functions, register, and replace with their backend counterparts
#                    for function_name, function in backends["numpy"].items():                
#                        try:
#                            print "registering backend", backend, function_name
#                            self.backend_listing[module_name][current_backend][function_name] = \
#                                getattr(backend_module, function_name)                 
#                        except AttributeError:
#                            pass

    def get_module_backends(self, module_name):
        """
        Iterate through a module's backend directory and find all files that matches the signature.
        """
        module_elements = module_name.split(".")
        target_module = module_elements[-1]
#        target_module = module_name.split(".")[-1]
#        backend_directory = os.path.join(os.path.split(sys.modules[module_name].__file__)[0], "backend")
#        if not os.path.exists(backend_directory):
#            return []
        module_name = ".".join(module_elements[:-1] + ["backend"])
        backend_group_module = __import__(module_name, fromlist=[module_name])
        backend_names = []
        for mod in dir(backend_group_module):
#        for mod in [x for x in os.listdir(backend_directory) if x.endswith(".py")]:
#            b = mod[:-3].split("_")
            b = mod.split("_")
            module_name = "_".join(b[:-1])
            if module_name == target_module:
                backend_names.append(b[-1])
        return backend_names
                
    def register_function(self, module_name, function, backends=None):
        """
        Register functions for a specific module
        """
        function_name = function.__name__
        if module_name not in self.backend_listing:
            self.backend_listing[module_name] = {}
            self.backend_listing[module_name]["numpy"] = {}
        # register numpy implementation
        self.backend_listing[module_name]["numpy"][function_name] = function
        function_backends = []
        for backend in backends:
            if backend not in self.backends:
                self.add_backend(backend)
            
            if backend not in self.backend_listing[module_name]:
                backend_module = import_backend(backend, module_name)
                if backend_module:
                    self.backend_listing[module_name][backend] = {}
            try:
                # register backend function
                self.backend_listing[module_name][backend][function_name] = \
                    getattr(backend_module, function_name)
                function_backends.append(backend)
            except AttributeError:
                pass      
        function.__doc__ += "Backends available: %s" % function_backends
            

manager = BackendManager()
use_backend = manager.use_backend        
        
    
class add_backends(object):
    def __init__(self, function):
        
        self.function = function
        self.function_name = function.__name__
        self.module_name = function.__module__
        # iterate through backend directory and find backends that match        
        backends = manager.get_module_backends(self.module_name)
        manager.register_function(self.module_name, function, backends=backends)
        print manager.backend_listing
                
    def __call__(self, *args, **kwargs):
        if "backend" in kwargs:
            backend = kwargs.get("backend")        
            del kwargs["backend"]
        else:
            backend = manager.current_backend
        # fall back to numpy if function not provided
        if backend not in manager.backend_listing[self.module_name] or \
        self.function_name not in manager.backend_listing[self.module_name][backend]:
            backend = "numpy"
        return manager.backend_listing[self.module_name][backend][self.function_name](*args, **kwargs)
    

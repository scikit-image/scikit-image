import sys

#imports = {}

#def get_backend(backend):
#    submodule = __name__ + "_%s" % backend
#    name = "backend.%s" % submodule
#    if name not in imports:
#        module = __import__(name, fromlist=[submodule])
#        imports[name] = module
#        return module
#    else:
#        return imports[name]

   
# backend decorator
#def add_backends(function):
#    def new_function(*args, **kwargs):
#        if "backend" in kwargs:
#            backend = kwargs.get("backend")            
#            del kwargs["backend"]
#        else:
#            backend = current_backend
#        if not backend:
#            return function(*args, **kwargs)
#        else:
#            return getattr(get_backend(backend), function.__name__)(*args, **kwargs)
#    return new_function

current_backend = "numpy"
backend_listing = {}

def use_backend(backend):
    global current_backend
    current_backend = backend
    for module_name, backends in backend_listing.items():
        if backend not in backends:
            # import backend module and register its functions
            backend_module = import_backend(backend, module_name)
            if backend_module:
                backend_listing[module_name][backend] = {}
                # iterate through module functions, register, and replace with their backend counterparts
                for function_name, function in backends["numpy"].items():                
                    try:
                        print "registering backend", backend, function_name
                        backend_listing[module_name][current_backend][function_name] = \
                            getattr(backend_module, function_name)                 
                    except AttributeError:
                        pass

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
    
    def add_backend(self, backend_name):
        b = Backend(name=backend_name)
        self.backends.append(b)
        return b
    
    def get_module_backends(self, module_name):
        """
        Iterate through a module's backend directory and find all files that matches the signature.
        """
        module_elements = module_name.split(".")
        module_name = ".".join(module_elements[:-1] + ["backend"])
        backend_group_module = __import__(module_name, fromlist=[module_name])
        backend_names = []
        target_module = module_elements[-1]
        for mod in dir(backend_group_module):
            b = mod.split("_")
            module_name = "_".join(b[:-1])
            if module_name == target_module:
                backend_names.append(b[-1])
                
    def support_function(self, module_name):
        pass

class Backend(object):
    def __init__(self, name=None):
        self.name = name
        self.functions = {}

class add_backends(object):
    def __init__(self, function):
        self.function = function
        self.function_name = function.__name__
        self.module_name = function.__module__
        # check if module already registered in listing
        if self.module_name not in backend_listing:
            backend_listing[self.module_name] = {}
            backend_listing[self.module_name]["numpy"] = {}
        # register numpy implementation
        backend_listing[self.module_name]["numpy"][self.function_name] = function
        
        # iterate through backend directory and find backends that match
#        backends = backend_manager.get_module_backends(self.module_name)
#        if backend not in backend_manager.backends:
#            backend = backend_manager.add_backend(backend_name)
#        backend_manager.support_function(self.module_name, function_name, backends=backends)
        # inject documentation here
        # scan through backend directory registering functions
        
        # if other backend is selected, import needed modules
        if current_backend != "numpy":
            if current_backend not in backend_listing[self.module_name]:
                backend_module = import_backend(current_backend, self.module_name)
                if backend_module:
                    backend_listing[self.module_name][current_backend] = {}
            try:
                # register backend function
                backend_listing[self.module_name][current_backend][self.function_name] = \
                    getattr(backend_module, self.function_name)
            except AttributeError:
                pass
        print "registering function", self.module_name, self.function_name, 
                
    def __call__(self, *args, **kwargs):
        if "backend" in kwargs:
            backend = kwargs.get("backend")            
            del kwargs["backend"]
        else:
            backend = current_backend
        # fall back to numpy if function not provided
        if backend not in backend_listing[self.module_name] or \
        self.function_name not in backend_listing[self.module_name][backend]:
            backend = "numpy"
        return backend_listing[self.module_name][backend][self.function_name](*args, **kwargs)
    

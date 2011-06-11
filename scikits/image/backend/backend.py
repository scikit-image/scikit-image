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
    


imports = {}
current_backend = "numpy"

def get_backend(backend):
    submodule = __name__ + "_%s" % backend
    name = "backend.%s" % submodule
    if name not in imports:
        module = __import__(name, fromlist=[submodule])
        imports[name] = module
        return module
    else:
        return imports[name]

def use_backend(backend):
    current_backend = backend
    
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

def import_backend(backend, module):
    submodule = module.__name__ + "_%s" % backend
    name = "backend.%s" % submodule
    try:
        return __import__(name, fromlist=[submodule])
    except ImportError:
        return None

class add_backends(object):
    def __init__(self, function):
        self.function = function
        self.function_name = function.__name__
        self.module_name = __name__
        self.module = sys.modules[__name__] 
        # check if module already registered in listing
        if self.module_name not in backend_listing:
            backend_listing[self.module_name] = {}
            backend_listing[self.module_name]["numpy"] = [self.module, {}]
        # register numpy implementation
        backend_listing[self.module_name]["numpy"][1][self.function_name] = function
        # if other backend is selected, import needed modules
        if current_backend != "numpy":
            if current_backend not in backend_listing[self.module_name]:
                backend_module = import_backend(current_backend, self.module)
                if backend_module:
                    backend_listing[self.module_name][current_backend] = [backend_module, {}]
            try:
                # register backend function
                backend_listing[self.module_name][current_backend][1][self.function_name] = \
                    getattr(backend_module, self.function_name)
            except AttributeError:
                pass
        print "registering function", self.module_name, self.function_name, 
                
    def __call__(self, *args, **kwargs):
        if "backend" in kwargs:
            backend = kwargs.get("backend")            
            del kwargs["backend"]
        else:
            if current_backend not in self.backend_listing[self.module]:
                backend_module = import_backend(current_backend, self.module)
                if backend_module:
                    backend_listing[self.module_name][current_backend] = [backend_module, {}]
            try:
                # register backend function
                backend_listing[self.module_name][current_backend][1][self.function_name] = \
                    getattr(backend_module, self.function_name)
            except AttributeError:
                backend = "numpy"
        
        return backend_listing[self.module_name][backend](*args, **kwargs)
    
    

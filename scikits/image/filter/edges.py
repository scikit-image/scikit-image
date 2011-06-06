"""edges.py - Sobel edge filter

Originally part of CellProfiler, code licensed under both GPL and BSD licenses.
Website: http://www.cellprofiler.org
Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2011 Broad Institute
All rights reserved.
Original author: Lee Kamentsky

"""
import numpy as np
from scipy.ndimage import convolve
import sys

# backend implementation 1

imports = {}
def get_backend(backend):
    submodule = __file__.split(".")[0] + "_%s" % backend
    name = "backend.%s" % submodule
    if name not in imports:
        module = __import__(name, fromlist=[submodule])
        imports[name] = module
        return module
    else:
        return imports[name]
    
    
# backend decorator
def add_backends1(function):
    def new_function(*args, **kwargs):
        backend = kwargs.get("backend")
        if "backend" in kwargs:
            del kwargs["backend"]
        if not backend:
            return function(*args, **kwargs)
        else:
            return getattr(get_backend(backend), function.__name__)(*args, **kwargs)
    return new_function

# backend implementation 2

backend_listing = {}
current_backend = "numpy"


def import_backend(backend, module):
    submodule = module.__name__ + "_%s" % backend
    name = "backend.%s" % submodule
    try:
        return __import__(name, fromlist=[submodule])
    except ImportError:
        return None


# backend decorator
class add_backends2(object):
    def __init__(self, function):
        self.function = function
        function_name = function.__name__
        this_module_name = __name__
        this_module = sys.modules[__name__] 
        # check if module already registered in listing
        if this_module_name not in backend_listing:
            backend_listing[this_module_name] = {}
            backend_listing[this_module_name]["numpy"] = [this_module, {}]
        # register numpy implementation
        backend_listing[this_module_name]["numpy"][1][function_name] = function
        # assign an alias to the default numpy implementation
        setattr(this_module, '_%s_numpy' % function_name, function)
        # if current backend is not numpy, import the backend module
        if current_backend != "numpy":
            if current_backend not in backend_listing[this_module_name]:
                backend_module = import_backend(current_backend, this_module)
                if backend_module:
                    backend_listing[this_module_name][current_backend] = [backend_module, {}]
            try:
                # register backend function
                backend_listing[this_module_name][current_backend][1][function_name] = \
                    getattr(backend_module, function_name)
            except AttributeError:
                pass
        print "registering function", this_module_name, function_name, 
                
    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)
    

def use_backend(backend):
    for module_name, backends in backend_listing.items():
        module = backends["numpy"][0]
        if backend in backends:
            # numpy module is the default module            
            backend_module, functions = backends[backend]
            # assign the specified backend as default on all functions            
            for function_name, function in functions.items():
                if backend == "numpy":
                    function_alias = "_%s_numpy" % function_name
                    setattr(module, function_name, getattr(backend_module, function_alias))
                else:
                    setattr(module, function_name, getattr(backend_module, function_name))
        else:
            # import backend module and register its functions
            backend_module = import_backend(backend, module)
            if backend_module:
                backend_listing[module.__name__][backend] = [backend_module, {}]
                # iterate through module functions, register, and replace with their backend counterparts
                for function_name, function in backends["numpy"][1].items():                
                    try:
                        print "registering backend", backend, function_name
                        backend_listing[module.__name__][current_backend][1][function_name] = \
                            getattr(backend_module, function_name)
                        setattr(module, function_name, \
                            getattr(backend_module, function_name))                        
                    except AttributeError:
                        pass  
                              
                        

@add_backends2
def sobel(image, axis=None, output=None):
    """Calculate the absolute magnitude Sobel to find the edges.

    Parameters
    ----------
    image : array_like, dtype=float
        Image to process
    mask : array_like, dtype=bool, optional
        An optional mask to limit the application to a certain area
    
    Returns
    -------
    output : ndarray
      The Sobel edge map.

    Notes
    -----
    Take the square root of the sum of the squares of the horizontal and
    vertical Sobels to get a magnitude that's somewhat insensitive to
    direction.
    
    Note that scipy's Sobel returns a directional Sobel which isn't
    useful for edge detection in its raw form.
    """
    print "running numpy sobel"
    if image.dtype == np.uint8:
        output_type = np.int16
    elif image.dtype == np.float32:
        output_type = np.float32
    if axis is None:
        dx = np.empty(image.shape, dtype=np.float32)
        dy = np.empty(image.shape, dtype=np.float32)
        convolve(image, np.array([[ 1, 2, 1],
                                  [ 0, 0, 0],
                                  [-1,-2,-1]]), output=dx)
        convolve(image, np.array([[ 1, 0,-1],
                                  [ 2, 0,-2],
                                  [ 1, 0,-1]]), output=dy)
        if output:
            output[:] = np.sqrt(dx ** 2 + dy ** 2)
            return output
        else:
            return np.sqrt(dx ** 2 + dy ** 2)
    elif axis == 0:
        dx = np.empty(image.shape, dtype=output_type)
        convolve(image, np.array([[ 1, 2, 1],
                                  [ 0, 0, 0],
                                  [-1,-2,-1]]), output=dx)
        return dx
    elif axis == 1:
        dy = np.empty(image.shape, dtype=output_type)
        convolve(image, np.array([[ 1, 0,-1],
                                  [ 2, 0,-2],
                                  [ 1, 0,-1]]), output=dy)
        return dy


if __name__ == "__main__":
    import os, time
    from scikits.image import data_dir
    from scikits.image import io
    from scikits.image.color import rgb2gray
    import edges
    
    io.use_plugin("gtk")
    #image = rgb2gray(io.imread(os.path.join(data_dir,"lena512.png"))).astype(np.uint8)
    image = rgb2gray(io.imread(os.path.join(data_dir,"lena512.png"))).astype(np.float32)
    #image = np.zeros((2000, 2000), dtype=np.float32)
    t = time.time()
    # imports the opencv backend making subsequential calls faster
    
    import edges
    use_backend("opencv")
    qwer
    
    get_backend("opencv")
    backends = [None, "opencv", "opencl"]
    for backend in backends:
        t = time.time()

        output0 = sobel(image, axis=None)
        print backend, time.time() - t
        io.imshow(output0.astype(np.uint8))
       
    io.show()
    
#    hprewitt = np.abs(convolve(image, np.array([[ 1, 1, 1],
#                                              [ 0, 0, 0],
#                                              [-1,-1,-1]]).astype(float) / 3.0))                              
#    vprewitt = np.abs(convolve(image, np.array([[ 1, 0,-1],
#                                              [ 1, 0,-1],
#                                              [ 1, 0,-1]]).astype(float) / 3.0))


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
    
def add_backends(function):
    def new_function(*args, **kwargs):
        backend = kwargs.get("backend")
        if "backend" in kwargs:
            del kwargs["backend"]
        if not backend:
            return function(*args, **kwargs)
        else:
            return getattr(get_backend(backend), function.__name__)(*args, **kwargs)
    return new_function
    
@add_backends
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
    io.use_plugin("gtk")
    #image = rgb2gray(io.imread(os.path.join(data_dir,"lena512.png"))).astype(np.uint8)
    image = rgb2gray(io.imread(os.path.join(data_dir,"lena512.png"))).astype(np.float32)
    #image = np.zeros((2000, 2000), dtype=np.float32)
    t = time.time()
    # imports the opencv backend making subsequential calls faster
    get_backend("opencv")
    backends = [None, "opencv", "opencl"]
    mod =__import__("backend.edges_opencv", fromlist=["edges_opencv"])
    for backend in backends:
        t = time.time()
        output0 = sobel(image, axis=None, backend=backend)
        print backend, time.time() - t
        io.imshow(output0.astype(np.uint8))
       
    io.show()
    
#    hprewitt = np.abs(convolve(image, np.array([[ 1, 1, 1],
#                                              [ 0, 0, 0],
#                                              [-1,-1,-1]]).astype(float) / 3.0))                              
#    vprewitt = np.abs(convolve(image, np.array([[ 1, 0,-1],
#                                              [ 1, 0,-1],
#                                              [ 1, 0,-1]]).astype(float) / 3.0))


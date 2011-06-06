import numpy as np
import os, time
from scikits.image import data_dir
from scikits.image import io
from scikits.image.color import rgb2gray
import edges
from edges import use_backend
from edges import sobel

if __name__ == "__main__":
    io.use_plugin("gtk")
    image = rgb2gray(io.imread(os.path.join(data_dir,"lena512.png"))).astype(np.float32)
    
    # implementation 2    
    backends = ["numpy", "opencv", "opencl", "numpy"]
    for backend in backends:
        use_backend(backend)
        t = time.time()
        output0 = edges.sobel(image, axis=None)       
        print backend, time.time() - t
        io.imshow(output0.astype(np.uint8))
    #io.show()

#    # implementation 1
#    # imports the opencv backend making subsequential calls faster
#    get_backend("opencv")
#    backends = [None, "opencv", "opencl"]
#    for backend in backends:
#        t = time.time()
#        output0 = sobel(image, axis=None, backend=backend)
#        print backend, time.time() - t
#        io.imshow(output0.astype(np.uint8))
#    io.show()
#    

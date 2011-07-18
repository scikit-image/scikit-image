from scikits.image.backend.backend import register_function
register_function("opencl", "scikits.image.filter.sobel", "scikits.image.filter.backend.edges_opencl.sobel")
register_function("opencv", "scikits.image.filter.sobel", "scikits.image.filter.backend.edges_opencv.sobel")
register_function("opencv", "scikits.image.filter.test", "scikits.image.filter.backend.edges_opencv.test")

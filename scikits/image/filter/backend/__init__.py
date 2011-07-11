from scikits.image.backend.backend import register_backend
register_backend("opencl", "scikits.image.filter", "scikits.image.filter.backend.edges_opencl.sobel")

register_backend("opencv", "scikits.image.filter", "scikits.image.filter.backend.edges_opencv.sobel")
register_backend("opencv", "scikits.image.filter", "scikits.image.filter.backend.edges_opencv.test")

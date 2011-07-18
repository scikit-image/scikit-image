from scikits.image.backend.backend import register
register(backend="opencl", module="scikits.image.filter.backend.edges_opencl", functions=["sobel"])
register(backend="opencv", module="scikits.image.filter.backend.edges_opencv", functions=["sobel", "test"])

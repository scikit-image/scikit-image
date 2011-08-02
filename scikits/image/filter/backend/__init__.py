from scikits.image.backend import register
register(backend="opencl", module="scikits.image.filter", functions=["edges_opencl.sobel"])
register(backend="opencv", module="scikits.image.filter", source="edges_opencv", functions=["sobel", "test"])

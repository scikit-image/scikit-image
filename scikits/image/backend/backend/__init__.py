from scikits.image.backend import register
register(backend="backend1", unlisted="True", module="scikits.image.backend", functions=["backend1._test1"])
register(backend="backend2", unlisted="True", module="scikits.image.backend", source="backend2", functions=["_test1", "_test2"])

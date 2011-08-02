from scikits.image.backend import register
register(backend="backend1", unlisted="True", module="scikits.image.backend", functions=["backend1.test1"])
register(backend="backend2", unlisted="True", module="scikits.image.backend", source="backend2", functions=["test1", "test2"])

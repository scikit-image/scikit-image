from _skimage2.util.noise import random_noise as random_noise  # noqa: F401

__all__ = ['random_noise']

from skimage._docutils import bind_namespace

bind_namespace(globals())

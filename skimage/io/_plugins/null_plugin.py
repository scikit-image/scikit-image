__all__ = ['imshow', 'imread', 'imsave', '_app_show']

import warnings

message = '''\
No plugin has been loaded.  Please refer to the docstring for ``skimage.io``
for a list of available plugins.  You may specify a plugin explicitly as
an argument to ``imread``, e.g. ``imread("image.jpg", plugin='pil')``.

'''


def imshow(*args, **kwargs):
    warnings.warn(RuntimeWarning(message))


def imread(*args, **kwargs):
    warnings.warn(RuntimeWarning(message))


def imsave(*args, **kwargs):
    warnings.warn(RuntimeWarning(message))


_app_show = imshow

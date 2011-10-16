__all__ = ['imshow', 'imread', '_app_show']

import warnings

message = '''\
No plugin has been loaded.  Please refer to

skimage.io.plugins()

for a list of available plugins.'''


def imshow(*args, **kwargs):
    warnings.warn(RuntimeWarning(message))


def imread(*args, **kwargs):
    warnings.warn(RuntimeWarning(message))

_app_show = imshow

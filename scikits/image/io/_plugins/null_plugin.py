__all__ = ['imshow', '_app_show']

import warnings

message = '''\
No plugin has been loaded.  Please refer to

scikits.image.io.plugins()

for a list of available plugins.'''

def imshow(*args, **kwargs):
    warnings.warn(RuntimeWarning(message))

_app_show = imshow

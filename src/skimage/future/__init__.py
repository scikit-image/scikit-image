"""Functionality with an experimental API.

.. warning::
    Note that we will not port these functions to the ``skimage2`` namespace.
    Please adapt your code accordingly.
"""

import lazy_loader as _lazy
from .._migration import ski2_migration_decorator as _smd

# This string used to decorate functions in module, define before defining
# functions in lazy loading clause below.
_PENDING_SKIMAGE2_NO_FUTURE = """\
``skimage.future.%(qual)s`` is deprecated, and will be removed in ``skimage2``.
Please adapt your code, perhaps by vendoring the code in this module.
"""

__getattr__, __dir__, __all__ = _lazy.attach_stub(__name__, __file__)

# For migration doc build.
_smd.extra_params['future_funcs'] = [f'skimage.future.{name}' for name in __all__]

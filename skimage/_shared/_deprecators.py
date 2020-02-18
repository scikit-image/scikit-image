from functools import partial
from ..version import __version__
from wabisabi import kwonly_change, default_parameter_change

kwonly_change = partial(kwonly_change,
                        library_name='scikit-image',
                        current_library_version=__version__)

default_parameter_change = partial(default_parameter_change,
                                   library_name='scikit-image',
                                   current_library_version=__version__)

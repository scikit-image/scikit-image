from functools import partial
from ..version import __version__
from deprecation_factory import kwonly_change
from deprecation_factory import default_parameter_change

kwonly_change = partial(kwonly_change,
                        library_name='scikit-image',
                        current_library_version=__version__)

default_parameter_change = partial(default_parameter_change,
                                   library_name='scikit-image',
                                   current_library_version=__version__)

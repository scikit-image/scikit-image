from warnings import warn
from skimage._shared.version_requirements import is_installed

from .viewers import ImageViewer, CollectionViewer
from .qt import qt_api

viewer_available = not qt_api is None and is_installed('matplotlib')
if not viewer_available:
    warn('Viewer requires matplotlib and Qt')

del qt_api, is_installed, warn

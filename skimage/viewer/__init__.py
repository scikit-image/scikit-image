from .._shared.utils import warn
from .._shared.version_requirements import is_installed

from .qt import has_qt
if not has_qt:
    warn('Viewer requires Qt')

if is_installed('matplotlib'):
    from .viewers import ImageViewer, CollectionViewer
    __all__ = ['ImageViewer', 'CollectionViewer']
else:
    __all__ = []
    warn('Viewer requires matplotlib.')

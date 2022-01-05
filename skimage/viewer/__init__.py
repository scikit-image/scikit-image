from .._shared.utils import warn
from .._shared._dependency_checks import has_mpl
from .qt import has_qt

if not has_qt:
    warn('viewer requires Qt')

if has_mpl:
    from .viewers import ImageViewer, CollectionViewer
else:
    ImageViewer, CollectionViewer = None, None
    warn('viewer requires matplotlib<3.5; '
         'please consider moving to another tool such as napari as the '
         'viewer module will be removed in version 1.0.', stacklevel=2)

__all__ = ['ImageViewer', 'CollectionViewer']

try:
    from .qt import QtGui as _QtGui
except ImportError as e:
    raise ImportError('Viewer requires Qt')

from .viewers import ImageViewer, CollectionViewer

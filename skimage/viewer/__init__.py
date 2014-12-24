import warnings
try:
    from .viewers import ImageViewer, CollectionViewer
except ImportError as e:
    warnings.warn('Viewer requires Qt')

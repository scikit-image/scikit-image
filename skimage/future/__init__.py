"""Functionality with an experimental API. Although you can count on the
functions in this package being around in the future, the API may change with
any version update **and will not follow the skimage two-version deprecation
path**. Therefore, use the functions herein with care, and do not use them in
production code that will depend on updated skimage versions.
"""
from .manual_segmentation import manual_polygon_segmentation
from .manual_segmentation import manual_lasso_segmentation


_remove_error = (
    'skimage.future.graph has been moved to skimage.segmentation.graph, and '
    'the API has changed. Please see the API documentation at '
    'https://scikit-image.org/docs/stable/api/skimage.segmentation.html'
)

class NotImported:
    def __init__(self, error_message=_remove_error):
        self.error_message = error_message

    def __getattr__(self, item):
        raise ImportError(self.error_message)


graph = NotImported(_remove_error)


del NotImported

__all__ = ['manual_lasso_segmentation',
           'manual_polygon_segmentation']

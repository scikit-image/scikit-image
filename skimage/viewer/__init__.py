"""skimage viewer subpackage.

``skimage.viewer`` provides a matplotlib-based canvas for displaying images and 
a Qt-based GUI-toolkit, with the goal of making it easy to create interactive 
image editors [1]_.

.. [1] http://scikit-image.org/docs/dev/user_guide/viewer.html

"""


from .._shared.utils import warn
from .viewers import ImageViewer, CollectionViewer
from .qt import has_qt

if not has_qt:
    warn('Viewer requires Qt')

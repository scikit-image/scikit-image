__all__ = ['label']

from ..measure._ccomp import label as _label
from skimage._shared.utils import deprecated

label = deprecated('skimage.measure.label')(_label)

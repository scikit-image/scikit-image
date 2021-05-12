import warnings

from .gray import *


warnings.warn(
    "Importing from skimage.morphology.grey is deprecated. "
    "Please import from skimage.morphology instead.",
    FutureWarning, stacklevel=2
)

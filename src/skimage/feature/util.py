from _skimage2.feature.util import (
    DescriptorExtractor as DescriptorExtractor,
    FeatureDetector as FeatureDetector,
    plot_matched_features as plot_matched_features,
)  # noqa: F401

__all__ = [
    'DescriptorExtractor',
    'FeatureDetector',
    'plot_matched_features',
]

from _skimage2.feature.util import (  # noqa: F401
    _mask_border_keypoints,
    _prepare_grayscale_input_2D,
)

from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())

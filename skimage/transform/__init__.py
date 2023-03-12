"""This module includes tools to transform images and volumetric data.

- Geometric Transformations:
 These transforms change the shape or position of an image.
 They are useful for tasks such as image registration,
 alignment, and geometric correction.<br>
 Examples: `~skimage.transform.AffineTransform`,
 `~skimage.transform.ProjectiveTransform`,
 `~skimage.transform.EuclideanTransform`.

- Image Resizing and Rescaling:
 These transforms change the size or resolution of an image.<br>
 They are useful for tasks such as down-sampling an image to
 reduce its size or up-sampling an image to increase its resolution.
 Examples: :func:`~skimage.transform.resize`,
 :func:`~skimage.transform.rescale`.

- Feature Detection and Extraction:
 These transforms identify and extract specific features or
 patterns in an image. They are useful for tasks s object detection,
 image segmentation, and  feature matching.<br>
 Examples: `~skimage.transform.hough_circle` and other Hough transforms,
 '~skimage.transform.pyramid_expand` and other Pyramid Transforms,
 `~skimage.transform.radon` and other Radon transforms.

- Image Transformation:
 These transforms change the appearance of an image without changing its
 content. They are useful for tasks such a creating image mosaics,
 applying artistic effects, and visualizing image data.
 Examples:`~skimage.transform.warp`, Inverse transforms.

The following transforms are available:

"""

from .hough_transform import (hough_line, hough_line_peaks,
                              probabilistic_hough_line, hough_circle,
                              hough_circle_peaks, hough_ellipse)
from .radon_transform import (radon, iradon, iradon_sart,
                              order_angles_golden_ratio)
from .finite_radon_transform import frt2, ifrt2
from .integral import integral_image, integrate
from ._geometric import (estimate_transform,
                         matrix_transform, EuclideanTransform,
                         SimilarityTransform, AffineTransform,
                         ProjectiveTransform, FundamentalMatrixTransform,
                         EssentialMatrixTransform, PolynomialTransform,
                         PiecewiseAffineTransform)
from ._warps import (swirl, resize, rotate, rescale,
                     downscale_local_mean, warp, warp_coords, warp_polar,
                     resize_local_mean)
from .pyramids import (pyramid_reduce, pyramid_expand,
                       pyramid_gaussian, pyramid_laplacian)


__all__ = ['hough_circle',
           'hough_ellipse',
           'hough_line',
           'probabilistic_hough_line',
           'hough_circle_peaks',
           'hough_line_peaks',
           'radon',
           'iradon',
           'iradon_sart',
           'order_angles_golden_ratio',
           'frt2',
           'ifrt2',
           'integral_image',
           'integrate',
           'warp',
           'warp_coords',
           'warp_polar',
           'estimate_transform',
           'matrix_transform',
           'EuclideanTransform',
           'SimilarityTransform',
           'AffineTransform',
           'ProjectiveTransform',
           'EssentialMatrixTransform',
           'FundamentalMatrixTransform',
           'PolynomialTransform',
           'PiecewiseAffineTransform',
           'swirl',
           'resize',
           'resize_local_mean',
           'rotate',
           'rescale',
           'downscale_local_mean',
           'pyramid_reduce',
           'pyramid_expand',
           'pyramid_gaussian',
           'pyramid_laplacian']

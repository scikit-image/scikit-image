Announcement: scikit-image 0.X.0
================================

We're happy to announce the release of scikit-image v0.X.0!

scikit-image is an image processing toolbox for SciPy that includes algorithms
for segmentation, geometric transformations, color space manipulation,
analysis, filtering, morphology, feature detection, and more.

For more information, examples, and documentation, please visit our website:

https://scikit-image.org

Starting from this release, scikit-image will follow the spirit of the recently
introduced numpy deprecation policy -- NEP 29
(https://github.com/numpy/numpy/blob/master/doc/neps/nep-0029-deprecation_policy.rst). 
Accordingly, scikit-image 0.16 drops the support for Python 3.5.
This release of scikit-image officially supports Python 3.6 and 3.7.

New Features
------------
- Added majority rank filter - ``filters.rank.majority``.


Improvements
------------


API Changes
-----------
- Deprecated subpackage ``skimage.novice`` has been removed.
- Default value of ``multichannel`` parameters has been set to False in
  ``skimage.transform.rescale``, ``skimage.transform.pyramid_reduce``,
  ``skimage.transform.pyramid_laplacian``,
  ``skimage.transform.pyramid_gaussian``, and
  ``skimage.transform.pyramid_expand``. No guessing is performed for 3D arrays
  anymore, so, please, make sure that the parameter is fixed to a proper value.
- Deprecated argument ``visualise`` has been removed from
  ``skimage.feature.hog``. Use ``visualize`` instead.¨
- ``skimage.transform.seam_carve`` has been completely removed from the
  library due to licensing restrictions.
- Parameter ``as_grey`` has been removed from ``skimage.data.load`` and
  ``skimage.io.imread``. Use ``as_gray`` instead.
- Parameter ``min_size`` has been removed from
  ``skimage.morphology.remove_small_holes``. Use ``area_threshold`` instead.
- Deprecated ``correct_mesh_orientation`` in ``skimage.measure`` has been
  removed.
- ``skimage.measure._regionprops`` has been completely switched to using
  row-column coordinates. Old x-y interface is not longer available.
- Default value of ``behavior`` parameter has been set to ``ndimage`` in
  ``skimage.filters.median``.
- Parameter ``flatten`` in `skimage.io.imread` has been removed in
  favor of ``as_gray``.
- Parameters ``Hxx, Hxy, Hyy`` have been removed from
  ``skimage.feature.corner.hessian_matrix_eigvals`` in favor of ``H_elems``.
- Default value of ``order`` parameter has been set to ``rc`` in
  ``skimage.feature.hessian_matrix``.
- ``skimage.util.img_as_*`` functions no longer raise precision and/or loss warnings.
- When used with floating point inputs, ``denoise_wavelet`` no longer rescales
  the range of the data or clips the output to the range [0, 1] or [-1, 1].
  For non-float inputs, rescaling and clipping still occurs as in prior
  releases (although with a bugfix related to the scaling of ``sigma``).


Bugfixes
--------
- ``denoise_wavelet``: For user-supplied `sigma`, if the input image gets
  rescaled via ``img_as_float``, the same scaling will be applied to `sigma` to
  preserve the relative scale of the noise estimate. To restore the old,
  behaviour, the user can manually specify ``rescale_sigma=False``.


Deprecations
------------
- Parameter ``neighbors`` in ``skimage.measure.convex_hull_object`` has been
  deprecated in favor of ``connectivity`` and will be removed in version 0.18.0.
- Parameter ``inplace`` in skimage.morphology.flood_fill has been deprecated
  in favor of ``in_place`` and will be removed in version scikit-image 0.19.0.


Contributors to this release
----------------------------

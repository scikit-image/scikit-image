Announcement: scikit-image 0.X.0
================================

We're happy to announce the release of scikit-image v0.X.0!

scikit-image is an image processing toolbox for SciPy that includes algorithms
for segmentation, geometric transformations, color space manipulation,
analysis, filtering, morphology, feature detection, and more.

For more information, examples, and documentation, please visit our website:

https://scikit-image.org



New Features
------------
- Added majority rank filter - ``filters.rank.majority``.

- Image affine registration (``skimage.transform.registration``)


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


Bugfixes
--------


Deprecations
------------


Contributors to this release
----------------------------

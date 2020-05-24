Announcement: scikit-image 0.X.0
================================

We're happy to announce the release of scikit-image v0.X.0!

scikit-image is an image processing toolbox for SciPy that includes algorithms
for segmentation, geometric transformations, color space manipulation,
analysis, filtering, morphology, feature detection, and more.

For more information, examples, and documentation, please visit our website:

https://scikit-image.org

Starting from release 0.16, scikit-image follows the spirit of the recently
introduced NumPy deprecation policy -- NEP 29
(https://github.com/numpy/numpy/blob/master/doc/neps/nep-0029-deprecation_policy.rst).
This release of scikit-image officially supports Python 3.6, 3.7, and
3.8.

New Features
------------
- Added majority rank filter - ``filters.rank.majority``.
- Added 3D implementation of rank autolevel filter - ``filters.rank.autolevel``.
# - Added 3D implementation of rank autolevel_percentile filter - ``filters.rank.autolevel_percentile``.
- Added 3D implementation of rank equalize filter - ``filters.rank.equalize``.
- Added 3D implementation of rank gradient filter - ``filters.rank.gradient``.
# - Added 3D implementation of rank gradient_percentile filter - ``filters.rank.gradient_percentile``.
- Added 3D implementation of rank majority filter - ``filters.rank.majority``.
- Added 3D implementation of rank maximum filter - ``filters.rank.maximum``.
- Added 3D implementation of rank mean filter - ``filters.rank.mean``.
- Added 3D implementation of rank geometric_mean filter - ``filters.rank.geometric_mean``.
# - Added 3D implementation of rank mean_percentile filter - ``filters.rank.mean_percentile``.
# - Added 3D implementation of rank mean_bilateral filter - ``filters.rank.mean_bilateral``.
- Added 3D implementation of rank subtract_mean filter - ``filters.rank.subtract_mean``.
# - Added 3D implementation of rank subtract_mean_percentile filter - ``filters.rank.subtract_mean_percentile``.
- Added 3D implementation of rank median filter - ``filters.rank.median``.
- Added 3D implementation of rank minimum filter - ``filters.rank.minimum``.
- Added 3D implementation of rank modal filter - ``filters.rank.modal``.
- Added 3D implementation of rank enhance_contrast filter - ``filters.rank.enhance_contrast``.
# - Added 3D implementation of rank enhance_contrast_percentile filter - ``filters.rank.enhance_contrast_percentile``.
- Added 3D implementation of rank pop filter - ``filters.rank.pop``.
# - Added 3D implementation of rank pop_percentile filter - ``filters.rank.pop_percentile``.
# - Added 3D implementation of rank pop_bilateral filter - ``filters.rank.pop_bilateral``.
- Added 3D implementation of rank sum filter - ``filters.rank.sum``.
# - Added 3D implementation of rank sum_bilateral filter - ``filters.rank.sum_bilateral``.
# - Added 3D implementation of rank sum_percentile filter - ``filters.rank.sum_percentile``.
- Added 3D implementation of rank threshold filter - ``filters.rank.threshold``.
# - Added 3D implementation of rank threshold_percentile filter - ``filters.rank.threshold_percentile``.
- Added 3D implementation of rank noise_filter filter - ``filters.rank.noise_filter``.
- Added 3D implementation of rank entropy filter - ``filters.rank.entropy``.
- Added 3D implementation of rank otsu filter - ``filters.rank.otsu``.
# - Added 3D implementation of rank percentile filter - ``filters.rank.percentile``.
# - Added 3D implementation of rank windowed_histogram filter - ``filters.rank.windowed_histogram``.


Improvements
------------



API Changes
-----------



Bugfixes
--------



Deprecations
------------



Contributors to this release
----------------------------

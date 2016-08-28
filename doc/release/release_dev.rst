Announcement: scikit-image 0.X.0
================================

We're happy to announce the release of scikit-image v0.X.0!

scikit-image is an image processing toolbox for SciPy that includes algorithms
for segmentation, geometric transformations, color space manipulation,
analysis, filtering, morphology, feature detection, and more.

For more information, examples, and documentation, please visit our website:

http://scikit-image.org


New Features
------------




Improvements
------------




API Changes
-----------
- The function ``filters.gaussian``, which wraps the SciPy NDImage function
  ``gaussian_filter``, incorrectly changed the default padding mode from
  'reflect' to 'nearest'. This was potentially quite misleading for users, and
  'reflect' is more correct. The default padding mode 'reflect' was restored.

Deprecations
------------


Contributors to this release
----------------------------

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
- Added 3D support for many filters in skimage.filters.rank.

- A new doc tutorial presenting a cell biology example has been added to the
  gallery (#4648). The scientific content benefited from a much appreciated
  review by Pierre Poulain and Fred Bernard, both assistant professors at
  Université de Paris and Institut Jacques Monod.

Improvements
------------



API Changes
-----------



Bugfixes
--------



Deprecations
------------

- In ``skimage.feature.structure_tensor``, an ``order`` argument has been
  introduced which will default to 'rc' starting in version 0.20.
- ``skimage.feature.structure_tensor_eigvals`` has been deprecated and will be
  removed in version 0.20. Use ``skimage.feature.structure_tensor_eigenvalues``
  instead.


Contributors to this release
----------------------------

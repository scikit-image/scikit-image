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

- Added a new perimeter function - ``measure.perimeter_crofton``.

- A new doc tutorial presenting a cell biology example has been added to the
  gallery (#4648). The scientific content benefited from a much appreciated
  review by Pierre Poulain and Fred Bernard, both assistant professors at
  Université de Paris and Institut Jacques Monod.

Improvements
------------

- In ``skimage.restoration.richardson_lucy``, computations are now be done in
  single-precision when the input image is single-precision. This can give a
  substantial performance improvement when working with single precision data.

- The performance of the SLIC superpixels algorithm
  (``skimage.segmentation.slice``) was improved for the case where a mask
  is supplied by the user (#4903). The specific superpixels produced by
  masked SLIC will not be identical to those produced by prior releases.

API Changes
-----------

- ``skimage.restoration.richardson_lucy`` returns a single-precision output
  when the input is single-precision. Prior to this release, double-precision
  was always used.


Bugfixes
--------

- Euler characteristic property of ``skimage.measure.regionprops`` was erroneous
  for 3D objects, since it did not take tunnels into account. A new implementation
  based on integral geometry fixes this bug.
- In ``skimage.morphology.selem.rectangle`` the ``height`` argument
  controlled the width and the ``width`` argument controlled the height.
  They have been replaced with ``nrow`` and ``ncol``.
- ``skimage.segmentation.flood_fill`` and ``skimage.segmentation.flood``
  now consistently handle negative values for ``seed_point``.
- In `skimage.draw.polygon`, segmentation fault caused by 0d inputs.

Deprecations
------------

- In ``skimage.feature.structure_tensor``, an ``order`` argument has been
  introduced which will default to 'rc' starting in version 0.20.
- ``skimage.feature.structure_tensor_eigvals`` has been deprecated and will be
  removed in version 0.20. Use ``skimage.feature.structure_tensor_eigenvalues``
  instead.
- In ``skimage.morphology.selem.rectangle`` the arguments ``width`` and 
  ``height`` have been deprecated. Use ``nrow`` and ``ncol`` instead.


Contributors to this release
----------------------------

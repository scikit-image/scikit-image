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
- The explicit setting ``threshold_rel=0` was removed from the Examples of the
  following docstrings: ``skimage.feature.BRIEF``,
  ``skimage.feature.corner_harris``, ``skimage.feature.corner_shi_tomasi``,
  ``skimage.feature.corner_foerstner``, ``skimage.feature.corner_fast``,
  ``skimage.feature.corner_subpix``, ``skimage.feature.corner_peaks``,
  ``skimage.feature.corner_orientations``, and
  ``skimage.feature._detect_octave``.
- In ``skimage.restoration._denoise``, the warning regarding
  ``rescale_sigma=None`` was removed.
- In ``skimage.restoration._cycle_spin``, the ``# doctest: +SKIP`` was removed.


Bugfixes
--------

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

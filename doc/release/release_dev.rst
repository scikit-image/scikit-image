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

- A new function `segmentation.expand_labels` has been added in order to dilate
  labels without overlap ([4795](https://github.com/scikit-image/scikit-image/pull/4795))
- It is now possible to pass extra measurement functions to
  `measure.regionprops` and `regionprops_table`
  ([#4810](https://github.com/scikit-image/scikit-image/pull/4810))
- Added a new perimeter function - ``measure.perimeter_crofton``.
- Added 3D support for many filters in skimage.filters.rank.

Documentation
-------------

- A new doc tutorial presenting a cell biology example has been added to the
  gallery (#4648). The scientific content benefited from a much appreciated
  review by Pierre Poulain and Fred Bernard, both assistant professors at
  Université de Paris and Institut Jacques Monod.
- New tutorial on [visualizing 3D data](https://scikit-image.org/docs/dev/auto_examples/applications/plot_3d_image_processing.html) ([#4850](https://github.com/scikit-image/scikit-image/pull/4850))
- Documentation has been added to the contributing notes about how to submit a
  gallery example 
- automatic formatting of docstrings for improved consistency ([#4849](https://github.com/scikit-image/scikit-image/pull/4849))
- improved docstring for `rgb2lab` ([#4839](https://github.com/scikit-image/scikit-image/pull/4839)) and `marching_cubes` [#4846](https://github.com/scikit-image/scikit-image/pull/4846)
- Improved docstring for `measure.marching_cubes`, mentioning how to decimate a
  mesh using mayavi [4846](https://github.com/scikit-image/scikit-image/pull/4846)


Improvements
------------

- In ``skimage.restoration.richardson_lucy``, computations are now be done in
  single-precision when the input image is single-precision. This can give a
  substantial performance improvement when working with single precision data.
- ``pyproject.toml`` has been added to the sdist.

- The performance of the SLIC superpixels algorithm
  (``skimage.segmentation.slice``) was improved for the case where a mask
  is supplied by the user (#4903). The specific superpixels produced by
  masked SLIC will not be identical to those produced by prior releases.

API Changes
-----------

- A default value has been added to `measure.find_contours`, corresponding to
  the half distance between the min and max values of the image 
  [#4862](https://github.com/scikit-image/scikit-image/pull/4862)
- ``skimage.restoration.richardson_lucy`` returns a single-precision output
  when the input is single-precision. Prior to this release, double-precision
  was always used.
- The default value of ``threshold_rel`` in ``skimage.feature.corner`` has
  changed from 0.1 to None, which corresponds to letting 
  ``skimage.feature.peak_local_max`` decide on the default. This is currently
  equivalent to ``threshold_rel=0``.


Bugfixes
--------

- For the ransac algorithm, improved the case where all data points are 
  outliers, which was previously raising an error 
  ([4844](https://github.com/scikit-image/scikit-image/pull/4844))
- An error-causing bug has been corrected for the `bg_color` parameter in `label2rgb` 
  when its value is a string 
  ([#4840](https://github.com/scikit-image/scikit-image/pull/4840))
- A normalization bug was fixed in `metrics.variation_of_information` 
  ([#4875](https://github.com/scikit-image/scikit-image/pull/4875/))
- Fixed the behaviour of Richardson-Lucy deconvolution for images with 3
  dimensions or more ([#4823](https://github.com/scikit-image/scikit-image/pull/4823))
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


Contributors to this release
----------------------------

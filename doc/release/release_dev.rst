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



API Changes
-----------

- A default value has been added to `measure.find_contours`, corresponding to
  the half distance between the min and max values of the image 
  [#4862](https://github.com/scikit-image/scikit-image/pull/4862)


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

Deprecations
------------

- In ``skimage.feature.structure_tensor``, an ``order`` argument has been
  introduced which will default to 'rc' starting in version 0.20.
- ``skimage.feature.structure_tensor_eigvals`` has been deprecated and will be
  removed in version 0.20. Use ``skimage.feature.structure_tensor_eigenvalues``
  instead.


Contributors to this release
----------------------------

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

- A new function ``segmentation.expand_labels`` has been added in order to dilate
  labels while preventing overlap (#4795)
- It is now possible to pass extra measurement functions to
  ``measure.regionprops`` and ``regionprops_table`` (#4810)
- Added a new perimeter function - ``measure.perimeter_crofton``.
- Added 3D support for many filters in skimage.filters.rank.
- New images have been added in the ``data`` subpackage: ``data.eagle``
  (#4922), TODO for other images
  Also note that the image for ``data.camera`` has been changed due to
  copyright issues (#4913).


Documentation
-------------

- A new doc tutorial presenting a cell biology example has been added to the
  gallery (#4648). The scientific content benefited from a much appreciated
  review by Pierre Poulain and Fred Bernard, both assistant professors at
  Université de Paris and Institut Jacques Monod.
- New tutorial on `visualizing 3D data <https://scikit-image.org/docs/dev/auto_examples/applications/plot_3d_image_processing.html>`_ (#4850)
- Documentation has been added to the contributing notes about how to submit a
  gallery example 
- Automatic formatting of docstrings for improved consistency (#4849)
- Improved docstring for ``rgb2lab`` (#4839) and ``marching_cubes`` (#4846)
- Improved docstring for ``measure.marching_cubes``, mentioning how to decimate a
  mesh using mayavi (#4846)
- Improved docstring for ``util.random_noise`` (#5001)
- Improved docstrings for ``morphology.h_maxima`` and ``morphology.h_minima``
  (#4929).
- Improved docstring for ``util.img_as_int`` (#4888).
- An example showing how to explore interactively the properties of labelled
  regions `has been added <https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_regionprops.html>`_
  (#5010).
- Documentation has been added to explain
  `how to download example datasets <https://scikit-image.org/docs/dev/install.html#downloading-all-demo-datasets>`_
  which are not installed with scikit-image (#4984). Similarly, the contributor
  guide has been updated to mention how to host new datasets in a gitlab
  repository (#4892).
- The `benchmarking section of the developer documentation <https://scikit-image.org/docs/dev/contribute.html#benchmarks>`_
  has been expanded (#4905).


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
- ``exposure.adjust_gamma`` has been accelerated for ``uint8`` images thanks to a
  LUT (#4966).  
- ``measure.label`` has been accelerated for boolean input images, by using
  ``scipy.ndimage``'s implementation for this case (#4945).
- ``util.apply_parallel`` now works with multichannel data (#4927).
- ``skimage.feature.peak_local_max`` supports now any Minkowski distance.


API Changes
-----------

- A default value has been added to ``measure.find_contours``, corresponding to
  the half distance between the min and max values of the image 
  #4862
- ``skimage.restoration.richardson_lucy`` returns a single-precision output
  when the input is single-precision. Prior to this release, double-precision
  was always used.
- The default value of ``threshold_rel`` in ``skimage.feature.corner`` has
  changed from 0.1 to None, which corresponds to letting
  ``skimage.feature.peak_local_max`` decide on the default. This is currently
  equivalent to ``threshold_rel=0``.
- ``data.cat`` has been introduced as an alias of ``data.chelsea`` for a more
  descriptive name.
- The ``level`` parameter of ``measure.find_contours`` is now a keyword
  argument, with a default value set to (max(image) - min(image)) / 2.
- ``p_norm`` argument was added to ``skimage.feature.peak_local_max``
  to add support for Minkowski distances.


Bugfixes
--------

- For the RANSAC algorithm, improved the case where all data points are
  outliers, which was previously raising an error
  (#4844)
- An error-causing bug has been corrected for the ``bg_color`` parameter in
  ``label2rgb`` when its value is a string (#4840)
- A normalization bug was fixed in ``metrics.variation_of_information``
  (#4875)
- Fixed the behaviour of Richardson-Lucy deconvolution for images with 3
  dimensions or more (#4823)
- Euler characteristic property of ``skimage.measure.regionprops`` was erroneous
  for 3D objects, since it did not take tunnels into account. A new implementation
  based on integral geometry fixes this bug.
- In ``skimage.morphology.selem.rectangle`` the ``height`` argument
  controlled the width and the ``width`` argument controlled the height.
  They have been replaced with ``nrow`` and ``ncol``.
- ``skimage.segmentation.flood_fill`` and ``skimage.segmentation.flood``
  now consistently handle negative values for ``seed_point``.
- Segmentation faults in ``segmentation.flood`` have been fixed in #4948 and #4972
- A segfault in ``draw.polygon`` for the case of 0-d input has been fixed
  (#4943).
- In ``registration.phase_cross_correlation``, a ``ValueError`` is raised when
  NaNs are found in the computation (as a result of NaNs in input images).
  Before this fix, an incorrect value could be returned where the input images
  had NaNs (#4886).
- ``min_distance`` is now enforced for ``skimage.feature.peak_local_max``
  (#2592).
- Peak detection in labels is fixed in ``skimage.feature.peak_local_max``
  (#4756).
- Input ``labels`` argument renumbering in ``skimage.feature.peak_local_max``
  is avoided (#5047).
- Work with pooch 1.5.0 for fetching data (#5529).


Deprecations
------------

- In ``skimage.feature.structure_tensor``, an ``order`` argument has been
  introduced which will default to 'rc' starting in version 0.20.
- ``skimage.feature.structure_tensor_eigvals`` has been deprecated and will be
  removed in version 0.20. Use ``skimage.feature.structure_tensor_eigenvalues``
  instead.
- The ``skimage.viewer`` subpackage and the ``skivi`` script have been
  deprecated and will be removed in version 0.20. For interactive visualization
  we recommend using dedicated tools such as napari or plotly. In a similar
  vein, the ``qt`` and ``skivi`` plugins of ``skimage.io`` have been deprecated
  and will be removed in version 0.20.
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
- In ``measure.label``, the deprecated ``neighbors`` parameter has been
  removed.


Development process
-------------------

- Benchmarks can now run on older scikit-image commits (#4891)
- Website analytics are tracked using plausible.io and can be visualized on
  https://plausible.io/scikit-image.org (#4893)
- Artifacts for the documentation build are now found in each pull request
  (#4881).
- Documentation source files can now be written in Markdown in addition to
  ResT, thanks to ``myst`` (#4863).

Contributors to this release
----------------------------

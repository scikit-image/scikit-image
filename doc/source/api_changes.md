# Version 0.16

- The following functions are deprecated and will be removed in 0.18:
  ``skimage.measure.compare_mse``,
  ``skimage.measure.compare_nrmse``,
  ``skimage.measure.compare_pnsr``,
  ``skimage.measure.compare_ssim``
  Their functionality still exists, but under the new ``skimage.metrics``
  submodule under different names.
- Additionally, three new functions have been added to ``skimage.metrics``:
  ``skimage.metrics.variation_of_information``
  ``skimage.metrics.adapted_rand_error``
  ``skimage.metrics.contingency_table``
- A new example of plotting these evaluation metrics has been added to the docs.

# Version 0.15

- ``skimage.feature.canny`` now uses a more accurate Gaussian filter
  internally; output values will be different from 0.14.
- ``skimage.filters.threshold_niblack`` and
  ``skimage.filters.threshold_sauvola``
  now accept a tuple as ``window_size`` besides integers.

# Version 0.14

- ``skimage.filters.gaussian_filter`` has been removed. Use
  ``skimage.filters.gaussian`` instead.
- ``skimage.filters.gabor_filter`` has been removed. Use
  ``skimage.filters.gabor`` instead.
- The old syntax support for ``skimage.transform.integrate`` has been removed.
- The ``normalise`` parameter of ``skimage.feature.hog`` was removed due to
  incorrect behavior: it only applied a square root instead of a true
  normalization. If you wish to duplicate the old behavior, set
  ``transform_sqrt=True``.
- ``skimage.measure.structural_similarity`` has been removed. Use
  ``skimage.measure.compare_ssim`` instead.
- In ``skimage.measure.compare_ssim``, the `dynamic_range` has been removed in
  favor of '`data_range`.
- In ``skimage.restoration.denoise_bilateral``, the `sigma_range` kwarg has
  been removed in favor of `sigma_color`.
- ``skimage.measure.marching_cubes`` has been removed in favor of
  ``skimage.measure.marching_cubes_lewiner``.
- ``ntiles_*`` parameters have been removed from
  ``skimage.exposure.equalize_adapthist``. Use ``kernel_size`` instead.
- ``skimage.restoration.nl_means_denoising`` has been removed in
  favor of ``skimage.restoration.denoise_nl_means``.
- ``skimage.measure.LineModel`` has been removed in favor of
  ``skimage.measure.LineModelND``.
- In ``skimage.feature.hog`` visualise has been changed to visualize.
- `freeimage` plugin of ``skimage.io`` has been removed.

# Version 0.13

- `skimage.filter` has been removed. Use `skimage.filters` instead.
- `skimage.filters.canny` has been removed.
  `canny` is available only from `skimage.feature` now.
- Deprecated filters `hsobel`, `vsobel`, `hscharr`, `vscharr`, `hprewitt`,
  `vprewitt`, `roberts_positive_diagonal`, `roberts_negative_diagonal` have
  been removed from `skimage.filters.edges`.
- The `sigma` parameter of `skimage.filters.gaussian` and the `selem` parameter
  of `skimage.filters.median` have been made optional, with default
  values.
- The `clip_negative` parameter of `skimage.util.dtype_limits` is now set
  to `None` by default, equivalent to `True`, the former value. In version
  0.15, will be set to `False`.
- The `circle` parameter of `skimage.transform.radon` and `skimage.transform.iradon`
  are now set to `None` by default, equivalent to `False`, the former value. In version
  0.15, will be set to `True`.
- Parameters ``ntiles_x``, ``ntiles_y`` have been removed from
  ``skimage.exposure.equalize_adapthist``.
- The ``freeimage`` io plugin is no longer supported, and will use ``imageio``
  instead.  We will completely remove the ``freeimage`` plugin in Version 0.14.

# Version 0.12

- ``equalize_adapthist`` now takes a ``kernel_size`` keyword argument, replacing
  the ``ntiles_*`` arguments.
- The functions ``blob_dog``, ``blob_log`` and ``blob_doh`` now return float
  arrays instead of integer arrays.
- ``transform.integrate`` now takes lists of tuples instead of integers
  to define the window over which to integrate.
- `reverse_map` parameter in `skimage.transform.warp` has been removed.
- `enforce_connectivity` in `skimage.segmentation.slic` defaults to ``True``.
- `skimage.measure.fit.BaseModel._params`,
  `skimage.transform.ProjectiveTransform._matrix`,
  `skimage.transform.PolynomialTransform._params`,
  `skimage.transform.PiecewiseAffineTransform.affines_*` attributes
  have been removed.
- `skimage.filters.denoise_*` have moved to `skimage.restoration.denoise_*`.
- `skimage.data.lena` has been removed.

# Version 0.11

- The ``skimage.filter`` subpackage has been renamed to ``skimage.filters``.
- Some edge detectors returned values greater than 1--their results are now
  appropriately scaled with a factor of ``sqrt(2)``.

# Version 0.10

- Removed ``skimage.io.video`` functionality due to broken gstreamer bindings

# Version 0.9

- No longer wrap ``imread`` output in an ``Image`` class
- Change default value of `sigma` parameter in ``skimage.segmentation.slic``
  to 0
- ``hough_circle`` now returns a stack of arrays that are the same size as the
  input image. Set the ``full_output`` flag to True for the old behavior.
- The following functions were deprecated over two releases:
  `skimage.filter.denoise_tv_chambolle`,
  `skimage.morphology.is_local_maximum`, `skimage.transform.hough`,
  `skimage.transform.probabilistic_hough`,`skimage.transform.hough_peaks`.
  Their functionality still exists, but under different names.

Version 0.4
-----------
- Switch mask and radius arguments for ``median_filter``

Version 0.3
-----------
- Remove ``as_grey``, ``dtype`` keyword from ImageCollection
- Remove ``dtype`` from imread
- Generalise ImageCollection to accept a load_func

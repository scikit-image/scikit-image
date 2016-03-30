Version 0.13
------------
- `skimage.filter` has been removed. Use `skimage.filters` instead.
- `skimage.filters.canny` has been removed.
  `canny` is available only from `skimage.feature` now.

Version 0.12
------------
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

Version 0.11
------------
- The ``skimage.filter`` subpackage has been renamed to ``skimage.filters``.
- Some edge detectors returned values greater than 1--their results are now
  appropriately scaled with a factor of ``sqrt(2)``.

Version 0.10
------------
- Removed ``skimage.io.video`` functionality due to broken gstreamer bindings

Version 0.9
-----------
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

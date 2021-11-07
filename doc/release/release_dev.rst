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

- Added support for processing images with channels located along any array
  axis. This is in contrast to previous releases where channels were required
  to be the last axis of an image. See more info on the new ``channel_axis``
  argument under the API section of the release notes.
- A no-reference measure of perceptual blur was added
  (``skimage.measure.blur_effect``).
- Non-local means (``skimage.restoration.denoise_nl_means``) now supports
  3D multichannel, 4D and 4D multichannel data when ``fast_mode=True``.
- An n-dimensional Fourier-domain Butterworth filter
  (``skimage.filters.butterworth``) was added.
- Color conversion functions now have a new ``channel_axis`` keyword argument
  that allows specification of which axis of an array corresponds to channels.
  For backwards compatibility, this parameter defaults to ``channel_axis=-1``,
  indicating that channels are along the last axis.
- Added a new keyword only parameter ``random_state`` to
  ``morphology.medial_axis`` and ``restoration.unsupervised_wiener``.
- Seeding random number generators will not give the same results as the
  underlying generator was updated to use ``numpy.random.Generator``.
- Added ``saturation`` parameter to ``skimage.color.label2rgb``
- Added normalized mutual information metric
  ``skimage.metrics.normalized_mutual_information``
- threshold_local now supports n-dimensional inputs and anisotropic block_size
- New ``skimage.util.label_points`` function for assigning labels to points.
- Added nD support to several geometric transform classes
- Added ``skimage.metrics.hausdorff_pair`` to find points separated by the
  Hausdorff distance.
- Additional colorspace ``illuminants`` and ``observers`` parameter options
  were added to ``skimage.color.lab2rgb``, ``skimage.color.rgb2lab``,
  ``skimage.color.xyz2lab``, ``skimage.color.lab2xyz``,
  ``skimage.color.xyz2luv`` and ``skimage.color.luv2xyz``.
- ``skimage.filters.threshold_multiotsu`` has a new ``hist`` keyword argument
  to allow use with a user-supplied histogram. (#5543)
- ``skimage.restoration.denoise_bilateral`` added support for images containing
  negative values. (#5527)
- The ``skimage.feature`` functions ``blob_dog``, ``blob_doh`` and ``blob_log``
  now support a ``threshold_rel`` keyword argument that can be used to specify
  a relative threshold (in range [0, 1]) rather than an absolute one. (#5517)
- Implement lazy submodule importing (#5101)
- Functions to create pixel graphs and find central pixels (#5602)


Documentation
-------------

- A new doc tutorial presenting a 3D biomedical imaging example has been added
  to the gallery (#4946). The technical content benefited from conversations
  with Genevieve Buckley, Kevin Mader, and Volker Hilsenstein.
- New gallery example for 3D structure tensor.
- New gallery example displaying a 3D dataset.
- Extended rolling ball example with ECG data (1D).
- The stain unmixing gallery example was fixed and now displays proper
  separation of the stains.
- Documentation has been added to the contributing notes about how to submit a
  gallery example.
- Autoformat docstrings in morphology.
- Display plotly figures from gallery example even when running script at CLI.
- Single out docs-only PRs in review process.
- Use matplotlib's infinite axline to demonstrate hough transform.
- Clarify disk documentation inconsistency regarding 'shape'.
- docs: fix simple typo, convertions -> conversions.
- Fixes to linspace in example.
- Minor fixes to Hough line transform code and examples.
- Added 1/2 pixel bounds to extent of displayed images in several examples.
- Add release step on github to RELEASE.txt.
- Remove reference to opencv in threshold_local documentation.
- Update structure_tensor docstring to include per-axis sigma.
- Fix typo in _shared/utils.py docs.
- Proofread and crosslink examples with immunohistochemistry image.
- Spelling correction: witch -> which.
- Mention possible filters in radon_transform -> filtered-back-projection
- Fix dtype info in documentation for watershed.
- Proofread gallery example for Radon transform.
- Use internal function for noise + clarify code in Canny example.
- Make more comprehensive 'see also' sections in filters.
- Specify the release note version instead of the misleading `latest`.
- Remove misleading comment in ``plot_thresholding.py`` example.
- Fix sphinx layout to make the search engine work with recent sphinx versions.
- Draw node IDs in RAG example.
- Update sigma_color description in denoise_bilateral.
- Update intersphinx fallback inventories + add matplotlib fallback inventory.
- Fix numpy deprecation in ``plot_local_equalize.py``.
- Rename ``label`` variable in ``plot_regionprops.py`` to circumvent link issue
  in docs.
- Avoid duplicate API documentation for ImageViewer, CollectionViewer.
- Fix 'blog_dog' typo in ``gaussian`` docs.
- Update reference link documentation in the ``adjust_sigmoid`` function.
- Fix reference to multiscale_basic_features in TrainableSegmenter.
- Slight ``shape_index`` docstring modification to specify 2D array.
- Add stitching gallery example (#5365)
- Add draft SKIP3: transition to scikit-image 1.0 (#5475)
- Mention commit messages in the contribution guidelines. (#5504)
- Fix and standardize docstrings for blob detection functions (#5547)
- Updated the User Guide to reflect usage of ``channel_axis`` rather than
  ``multichannel``. (#5554)
- Update the user guide to use channel_axis rather than multichannel (#5556)
- Add hyperlinks to referenced documentation places. (#5560)
- Update branching instructions to change the location of the pooch repo (#5565)
- add Notes and References section to the Cascade class docstring (#5568)
- Clarify 2D vs nD in skimage.feature.corner docstrings (#5569)
- Fixed math formulas in plot_swirl.py example (#5574)
- Update references in texture feature detectors docstrings (#5578)
- Update mailing list location to discuss.scientific-python.org forum (#5951)
- DOC: Fix docstring in rescale_intensity() (#5964)
- Fix slic documentation (#5975)
- Update docstring for dilation, which is now nD. (#5978)
- Change stitching gallery example thumbnail (#5985)
- Add circle and disk to glossary.md (#5590)
- Update pixel graphs example (#5991)
- Separate entries that have the same description in glossary.md (#5592)
- Do not use space before colon in directive name (#6002)


Improvements
------------

- Many more functions throughout the library now have single precision
  (float32) support.
- Biharmonic  inpainting (``skimage.restoration.inpaint_biharmonic``) was
  refactored and is orders of magnitude faster than before.
- Salt-and-pepper noise generation with ``skimage.util.random_noise`` is now
  faster.
- The performance of the SLIC superpixels algorithm
  (``skimage.segmentation.slice``) was improved for the case where a mask
  is supplied by the user (#4903). The specific superpixels produced by
  masked SLIC will not be identical to those produced by prior releases.
- ``exposure.adjust_gamma`` has been accelerated for ``uint8`` images thanks to
  a LUT (#4966).
- ``measure.label`` has been accelerated for boolean input images, by using
  ``scipy.ndimage``'s implementation for this case (#4945).
- ``util.apply_parallel`` now works with multichannel data (#4927).
- ``skimage.feature.peak_local_max`` supports now any Minkowski distance.
- Fast, non-Cython implementation for ``skimage.filters.correlate_sparse``.
- For efficiency, the histogram is now precomputed within
  ``skimage.filters.try_all_threshold``.
- Faster ``skimage.filters.find_local_max`` when given a finite ``num_peaks``.
- All filters in the ``skimage.filters.rank`` module now release the GIL,
  enabling multithreaded use.
- ``skimage.restoration.denoise_tv_bregman`` and
  ``skimage.restoration.denoise_bilateral`` now release the GIL, enabling
  multithreaded use.
- A ``skimage.color.label2rgb`` performance regression was addressed.
- Improved numerical precision in ``CircleModel.estimate``. (#5190)
- Added default keyword argument values to
  ``skimage.restoration.denoise_tv_bregman``, ``skimage.measure.block_reduce``,
  and ``skimage.filters.threshold_local``. (#5454)
- Make matplotlib an optional dependency (#5990)
- single precision support in skimage.filters (#5354)
- Support nD images and labels in label2rgb (#5550)
- Regionprops table performance refactor (#5576)
- add regionprops benchmark script (#5579)
- remove use of apply_along_axes from greycomatrix & greycoprops (#5580)
- refactor gabor_kernel for efficiency (#5582)
- remove need for channel_as_last_axis decorator in skimage.filters (#5584)
- replace use of scipy.ndimage.gaussian_filter with skimage.filters.gaussian (#5872)
- add channel_axis argument to quickshift (#5987)


API Changes
-----------

- The ``multichannel`` boolean argument has been deprecated. All functions with
  multichannel support now use an integer ``channel_axis`` to specify which
  axis corresponds to channels. Setting ``channel_axis`` to None is used to
  indicate that the image is grayscale. Specifically, existing code with
  ``multichannel=True`` should be updated to use ``channel_axis=-1`` and code
  with ``multichannel=False`` should now specify ``channel_axis=None``.
- Most functions now return float32 images when the input has float32 dtype.
- A default value has been added to ``measure.find_contours``, corresponding to
  the half distance between the min and max values of the image
  (#4862).
- ``data.cat`` has been introduced as an alias of ``data.chelsea`` for a more
  descriptive name.
- The ``level`` parameter of ``measure.find_contours`` is now a keyword
  argument, with a default value set to ``(max(image) - min(image)) / 2``.
- ``p_norm`` argument was added to ``skimage.feature.peak_local_max``
  to add support for Minkowski distances.
- ``skimage.transforms.integral_image`` now promotes floating point inputs to
  double precision by default (for accuracy). A new ``dtype`` keyword argument
  can be used to override this behavior when desired.
- Color conversion functions now have a new ``channel_axis`` keyword argument
  (see **New Features** section).
- SLIC superpixel segmentation outputs may differ from previous versions for
  data that was not already scaled to [0, 1] range. There is now an automatic
  internal rescaling of the input to [0, 1] so that the ``compactness``
  parameter has an effect that is independent of the input image's scaling.
- A bug fix to the phase nomalization applied within
  ``skimage.register.phase_cross_correlation`` may result in a different result
  as compared to prior releases. The prior behavior of "unnormalized" cross
  correlation is still available by explicitly setting ``normalization=None``.
  There is no change to the masked cross-correlation case, which uses a
  different algorithm.


Bugfixes
--------

- Input ``labels`` argument renumbering in ``skimage.feature.peak_local_max``
  is avoided (#5047).
- Nonzero values at the image edge are no longer incorrectly marked as a
  boundary when using ``find_bounaries`` with mode='subpixel' (#5447).
- Fix return dtype of ``_label2rgb_avg`` function.
- Ensure ``skimage.color.separate_stains`` does not return negative values.
- Prevent integer overflow in ``EllipseModel``.
- Fixed off-by one error in pixel bins in Hough line transform,
  ``skimage.transform.hough_line``.
- Handle 1D arrays properly in ``skimage.filters.gaussian``.
- Fix Laplacian matrix size bug in ``skimage.segmentation.random_walker``.
- Regionprops table (``skimage.measure.regionprops_table``) dtype bugfix.
- Fix ``skimage.transform.rescale`` when using a small scale factor.
- Fix ``skimage.measure.label`` segfault.
- Watershed (``skimage.segmentation.watershed``): consider connectivity when
  calculating markers.
- Fix ``skimage.transform.warp`` output dtype when order=0.
- Fix multichannel ``intensity_image`` extra_properties in regionprops.
- Fix error message for ``skimage.metric.structural_similarity`` when image is
  too small.
- Do not mark image edges in 'subpixel' mode of
  ``skimage.segmentation.find_boundaries``.
- Fix behavior of ``skimage.exposure.is_low_contrast`` for boolean inputs.
- Fix wrong syntax for the string argument of ValueError in
  ``skimage.metric.structural_similarity`` .
- Fixed NaN issue in ``skimage.filters.threshold_otsu``.
- Fix ``skimage.feature.blob_dog`` docstring example and normalization.
- Fix uint8 overflow in ``skimage.exposure.adjust_gamma``.
- Work with pooch 1.5.0 for fetching data (#5529).
- The ``offsets`` attribute of ``skimage.graph.MCP`` is now public. (#5547)
- Fix io.imread behavior with pathlib.Path inputs (#5543)
- Make scikit-image imports from Pooch, compatible with pooch >= 1.5.0. (#5529)
- Fix several broken doctests and restore doctesting on GitHub Actions. (#5505)
- Fix broken doctests in ``skimage.exposure.histogram`` and
  ``skimage.measure.regionprops_table``. (#5522)
- Rescale image consistently during SLIC superpixel segmentation. (#5518)
- Corrected phase correlation in ``skimage.register.phase_cross_correlation``.
  (#5461)
- Fix hidden attribute 'offsets' in skimage.graph.MCP (#5551)
- fix phase_cross_correlation for 3D with reference masks (#5559)
- fix return shape of blob_log and blob_dog when no peaks are found (#5567)
- Fix find contours key error (#5577)
- Refactor measure.ransac and add warning when the estimated model is not valid (#5583)
- trainable_segmentation: re-raise in error case (#5600)
- allow regionprops_table to be called with deprecated property names (#5908)
- Fix weight calculation in fast mode of non-local means (#5923)
- fix for #5948: lower boundary 1 for kernel_size in equalize_adapthist (#5949)
- convert pathlib.Path to str in imsave (#5971)
- Fix slic spacing (#5974)
- Add small regularization to avoid zero-division in richardson_lucy (#5976)
- Fix benchmark suite (watershed function was moved) (#5982)
- Fix the estimation of ellipsoid axis lengths in the 3D case (#6013)


Deprecations
------------

Completed deprecations from prior releases
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- In ``measure.label``, the deprecated ``neighbors`` parameter has been
  removed (use ``connectivity`` instead).
- The deprecated ``skimage.color.rgb2grey`` and ``skimage.color.grey2rgb``
  functions have been removed (use ``skimage.color.rgb2gray`` and
  ``skimage.color.gray2rgb`` instead).
- ``skimage.color.rgb2gray`` no longer allows grayscale or RGBA inputs.
- The deprecated ``alpha`` parameter of ``skimage.color.gray2rgb`` has now been
  removed. Use ``skimage.color.gray2rgba`` for conversion to RGBA.
- Attempting to warp a boolean image with ``order > 0`` now raises a ValueError.
- When warping or rescaling boolean images, setting ``anti-aliasing=True`` will
  raise a ValueError.
- The ``bg_label`` parameter of ``skimage.color.label2rgb`` is now 0.
- The deprecated ``filter`` parameter of ``skimage.transform.iradon`` has now
  been removed (use ``filter_name`` instead).
- The deprecated ``skimage.draw.circle`` function has been removed (use
  ``skimage.draw.disk`` instead).
- The deprecated ``skimage.feature.register_translation`` function has
  been removed (use ``skimage.registration.phase_cross_correlation`` instead).
- The deprecated ``skimage.feature.masked_register_translation`` function has
  been removed (use ``skimage.registration.phase_cross_correlation`` instead).
- The deprecated ``skimage.measure.marching_cubes_classic`` function has
  been removed (use ``skimage.measure.marching_cubes`` instead).
- The deprecated ``skimage.measure.marching_cubes_lewiner`` function has
  been removed (use ``skimage.measure.marching_cubes`` instead).
- The deprecated ``skimage.segmentation.circle_level_set`` function has been
  removed (use ``skimage.segmentation.disk_level_set`` instead).
- The deprecated ``inplace`` parameter of ``skimage.morphology.flood_fill``
- The deprecated ``skimage.util.pad`` function has been removed (use
  ``numpy.pad`` instead).
  been removed (use ``in_place`` instead).
- The default ``mode`` in ``skimage.filters.hessian`` is now
  ``'reflect'``.
- The default boundary ``mode`` in ``skimage.filters.sato`` is now
  ``'reflect'``.
- The default boundary ``mode`` in ``skimage.measure.profile_line`` is now
  ``'reflect'``.
- The default value of ``preserve_range`` in
  ``skimage.restoration.denoise_nl_means`` is now False.
- The default value of ``start_label`` in ``skimage.segmentation.slic`` is now
  1.

Newly introduced deprecations:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- The ``multichannel`` argument is now deprecated throughout the library and
  will be removed in 1.0. The new ``channel_axis`` argument should be used
  instead. Existing code with ``multichannel=True`` should be updated to use
  ``channel_axis=-1`` and code with ``multichannel=False`` should now specify
  ``channel_axis=None``.
- ``skimage.feature.greycomatrix`` and ``skimage.feature.greycoprops`` are
  deprecated in favor of ``skimage.feature.graycomatrix`` and
  ``skimage.feature.graycoprops``.
- The ``skimage.morphology.grey`` module has been renamed
  ``skimage.morphology.gray``. The old name is deprecated.
- The ``skimage.morphology.greyreconstruct`` module has been renamed
  ``skimage.morphology.grayreconstruct``. The old name is deprecated.
- see **API Changes** section regarding functions with deprecated argument
  names related to the number of iterations. ``num_iterations`` and
  ``max_num_iter`` are now used throughout the library.
- see **API Changes** section on deprecation of the ``selem`` argument in favor
  of ``footprint`` throughout the library
- Deprecate ``in_place`` in favor of the use of an explicit ``out`` argument
  in ``skimage.morphology.remove_small_objects``,
  ``skimage.morphology.remove_small_holes`` and
  ``skimage.segmentation.clear_border``
- The ``input`` argument of ``skimage.measure.label`` has been renamed
  ``label_image``. The old name is deprecated.
- standardize on ``num_iter`` for paramters describing the number of iterations
  and ``max_num_iter`` for parameters specifying an iteration limit. Functions
  where the old argument names have now been deprecated are::

    skimage.filters.threshold_minimum
    skimage.morphology.thin
    skimage.restoration.denoise_tv_bregman
    skimage.restoration.richardson_lucy
    skimage.segmentation.active_contour
    skimage.segmentation.chan_vese
    skimage.segmentation.morphological_chan_vese
    skimage.segmentation.morphological_geodesic_active_contour
    skimage.segmentation.slic

- The names of several parameters in ``skimage.measure.regionprops`` have been
  updated so that properties are better grouped by the first word(s) of the
  name. The old names will continue to work for backwards compatibility.
  The specific names that were updated are::

    ============================ ============================
    Old Name                     New Name
    ============================ ============================
    max_intensity                intensity_max
    mean_intensity               intensity_mean
    min_intensity                intensity_min

    bbox_area                    area_bbox
    convex_area                  area_convex
    filled_area                  area_filled

    convex_image                 image_convex
    filled_image                 image_filled
    intensity_image              image_intensity

    local_centroid               centroid_local
    weighted_centroid            centroid_weighted
    weighted_local_centroid      centroid_weighted_local

    major_axis_length            axis_major_length
    minor_axis_length            axis_minor_length

    weighted_moments             moments_weighted
    weighted_moments_central     moments_weighted_central
    weighted_moments_hu          moments_weighted_hu
    weighted_moments_normalized  moments_weighted_normalized

    equivalent_diameter          equivalent_diameter_area
    ============================ ============================

- The ``selem`` argument has been renamed to ``footprint`` throughout the
  library. The ``footprint`` argument is now deprecated.


Development process
-------------------

- Test setup and teardown functions added to allow raising an error on any
  uncaught warnings via ``SKIMAGE_TEST_STRICT_WARNINGS_GLOBAL`` environment
  variable.
- Increase automation in release process.
- Release wheels before source
- update minimum supported Matplotlib, NumPy, SciPy and Pillow
- Pin pillow to !=8.3.0
- Rename `master` to `main` throughout
- Ensure that README.txt has write permissions for subsequent imports.
- Run face classification gallery example with a single thread
- Enable pip and skimage.data caching on Azure
- Fix CircleCI and Azure CI caching.
- Address Cython warnings.
- Disable calls to plotly.io.show when running on Azure.
- Remove legacy Travis-CI scripts and update contributor documentation
  accordingly.
- Increase cibuildwheel verbosity.
- Update pip during dev environment installation.
- Add benchmark checks to CI.
- Resolve stochastic rank filter test failures on CI.
- Ensure that README.txt has write permissions for subsequent imports.
- Decorators for helping with the transition between the keyword argument
  multichannel and channel_axis.
- Missing import in lch2lab docstring example (#5998)
- Prefer importing build_py and sdist from setuptools (#6007)
- Reintroduce skimage.test utility (#5909)


Other Updates
-------------
- Refactor np.random.x to use np.random.Generator.
- Avoid warnings about use of deprecated ``scipy.linalg.pinv2``.
- Simplify resize implementation using new SciPy 1.6 zoom option.
- Fix duplicate test function names in ``test_unsharp_mask.py``.
- Benchmarks: ``fix ResizeLocalMeanSuite.time_resize_local_mean`` signature.
- prefer use of new-style NumPy random API in tests (#5450)
- Add fixture enforcing SimpleITK I/O in test_simpleitk.py (#5526)
- MNT: Remove unused stat import from skimage data (#5566)
- MAINT: Remove unused imports (#5595)
- MAINT: Refactor duplicated tests, remove unnecessary assignments and variables (#5596)
- Remove obsolete lazy import (#5992)
- Lazily load data_dir into the top-level namespace (#5996)


Contributors to this release
----------------------------

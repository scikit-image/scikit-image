scikit-image 0.19.3 (2022-06-12)
================================

We're happy to announce the release of scikit-image v0.19.3!

scikit-image is an image processing toolbox for SciPy that includes algorithms
for segmentation, geometric transformations, color space manipulation,
analysis, filtering, morphology, feature detection, and more.

For more information, examples, and documentation, please visit our website:

https://scikit-image.org

Bugs Fixed
----------
- Revert unintentional change to default multichannel behavior introduced in v0.19.0 for ``skimage.restoration.cycle_spin`` (now defaults to single channel again)
- Fix corner case with an optimal angle of 0 degrees in hough_line_peaks
- Fixed the gallery example involving registration with log-polar transformations
- Update test suite for compatibility with the most recent ``tifffile`` release.
- warp/rotate: fixed a bug with clipping when cval is not in the input range
- Fix computation of histogram bins for multichannel integer-valued images

General Maintenance
-------------------
- Update ``skimage.future.manual_polygon_segmentation`` to work with Matplotlib 3.5.
- Update ``skimage.io.imread`` to avoid warnings when using ``imageio``>=2.16.2.
- Now compatible with Pillow >= 9.1 (palette may contain <256 entries)
- Added support for NumPy 1.23

Pull Requests Included
----------------------
- Backport PR #6306 on branch v0.19.x (Fix for error in 'Using Polar and Log-Polar Transformations for Registration') (#6312)
- Backport PR #6271 on branch v0.19.x (hough_line_peaks fix for corner case with optimal angle=0) (#6313)
- Backport PR #6261 on branch v0.19.x (Ignore sparse matrix deprecation warning) (#6316)
- backport PR 6328: Fix issue with newer versions of matplotlib in manual segmentation (#6334)
- Backport PR #6343 on branch v0.19.x (avoid warnings about change to v3 API from imageio) (#6344)
- Backport PR #6355 on branch v0.19.x (remove use of deprecated kwargs from `test_tifffile_kwarg_passthrough`) (#6357)
- Backport PR #6352 on branch v0.19.x (Fix channel_axis default for cycle_spin) (#6358)
- Backport PR #6348 on branch v0.19.x (Fix smoothed image computation when mask is None in canny) (#6359)
- Backport PR #6361 on branch v0.19.x (Document support for Path objects in io functions) (#6363)
- Backport PR #6400 on branch v0.19.x (Add support for NumPy 1.23) (#6403)
- Backport PR #6335 on branch v0.19.x (warp/rotate: fixed a bug with clipping when cval is not in the input range) (#6411)
- Backport PR #6413 on branch v0.19.x (Fix computation of histogram bins for multichannel integer-valued images) (#6414)

10 authors added to this release [alphabetical by first name or login]
----------------------------------------------------------------------
- Albert Y. Shih
- Bartłomiej Śmietanka
- Dave Mellert
- Gregory Lee
- Graham Inggs
- Jarrod Millman
- John Hagen
- Mark Harfouche
- Riadh Fezzani
- Stefan van der Walt

7 reviewers added to this release [alphabetical by first name or login]
-----------------------------------------------------------------------
- Alexandre de Siqueira
- Gregory Lee
- Jarrod Millman
- Juan Nunez-Iglesias
- Lars Grüter
- Mark Harfouche
- Riadh Fezzani


scikit-image 0.19.2 (2021-02-17)
================================

We're happy to announce the release of scikit-image v0.19.2! This is primarily
a bug fix release, although there is one new gallery example related to
detection of fluorescence at the nuclear envelope of mammalian cells.

scikit-image is an image processing toolbox for SciPy that includes algorithms
for segmentation, geometric transformations, color space manipulation,
analysis, filtering, morphology, feature detection, and more.

For more information, examples, and documentation, please visit our website:

https://scikit-image.org

Pull Requests Included
----------------------
- fix mistake in tests.yml made during backport (gh-6129)
- Backport PR #6145 on branch v0.19.x (Fix channel_axis handling in pyramid_gaussian and pyramid_laplace) (gh-6155)
- Backport PR #6130 on branch v0.19.x (bump deprecated Azure windows environment) (gh-6131)
- Backport PR #6148 on branch v0.19.x (deprecate n_iter_max (should be max_num_iter)) (gh-6156)
- Backport PR #6152 on branch v0.19.x (specify python version used by mybinder.org for gallery demos) (gh-6157)
- Backport PR #6139 on branch v0.19.x (fix phase_cross_correlation typo) (gh-6158)
- Backport PR #6133 on branch v0.19.x (Update user warning message for viewer module.) (gh-6159)
- Backport PR #6169 on branch v0.19.x (Fix unintended change to output dtype of match_histograms) (gh-6172)
- Backport PR #6184 on branch v0.19.x (Fix SIFT wrong octave indices + typo) (gh-6186)
- Backport PR #6191 on branch v0.19.x (Fix issue6190 - inconsistent default parameters in pyramids.py) (gh-6193)
- Backport PR #6207 on branch v0.19.x (Always set params to nan when ProjectiveTransform.estimate fails) (gh-6210)
- Backport PR #5262 on branch v0.19.x (Add textbook-like tutorial on measuring fluorescence at nuclear envelope.) (gh-6213)
- Backport PR #6087 on branch v0.19.x (Add two datasets for use in upcoming scientific tutorials.) (gh-6215)
- Backport PR #6214 on branch v0.19.x (EuclideanTransform.estimate should return False when NaNs are present) (gh-6221)
- Backport PR #6219 on branch v0.19.x (Allow the output_shape argument to be any iterable for resize and resize_local_mean) (gh-6222)
- Backport PR #6223 on branch v0.19.x (Update filename in testing instructions.) (gh-6225)
- Backport PR #6231 on branch v0.19.x (Update imports/refs from deprecated scipy.ndimage.filters namespace) (gh-6233)
- Backport PR #6229 on branch v0.19.x (Remove redundant testing on Appveyor) (gh-6234)
- Backport PR #6183 on branch v0.19.x (Fix decorators warnings stacklevel) (gh-6238)
- Backport PR #6239 on branch v0.19.x (DOC: fix SciPy intersphinx) (gh-6241)
- Backport PR #6232 on branch v0.19.x (Include Cython sources via package_data) (gh-6244)
- Backport PR #6227 on branch v0.19.x (Fix calculation of Z normal in marching cubes) (gh-6245)
- Backport PR #6242 on branch v0.19.x (Fix bug in SLIC superpixels with `enforce_connectivity=True` and `start_label > 0`) (gh-6246)
- Backport PR #6211 on branch v0.19.x (PiecewiseAffineTransform.estimate return should reflect underlying transforms) gh-6247
- update MacOS libomp installation in wheel building script (gh-6249)


9 authors added to this release [alphabetical by first name or login]
---------------------------------------------------------------------
- Chris Roat
- Fabian Schneider
- Gregory Lee
- Hande Gözükan
- Larry Bradley
- Marianne Corvellec
- Mark Harfouche
- Miles Lucas
- Riadh Fezzani

8 reviewers added to this release [alphabetical by first name or login]
-----------------------------------------------------------------------
- Alexandre de Siqueira
- Gregory Lee
- Juan Nunez-Iglesias
- Marianne Corvellec
- Mark Harfouche
- Riadh Fezzani
- Robert Haase
- Stefan van der Walt


scikit-image 0.19.1 (2021-12-03)
================================

We're happy to announce the release of scikit-image v0.19.1!

This is a small bug fix release that resolves a couple of backwards compatibility issues and a couple of issues with the wheels on PyPI. Specifically, MacOs wheels for Apple M1 (arm64) on PyPI were broken in 0.19.0, but should now be repaired. The arm64 wheels are for MacOs >= 12 only. Wheel sizes are also greatly reduced relative to 0.19.0 by stripping debug symbols from the binaries and making sure that Cython-generated source files are not bundled in the wheels.


Pull Requests Included
----------------------
- Backport PR #6089 on branch v0.19.x (Skip tests requiring fetched data) (gh-6115)
- Backport PR #6097 on branch v0.19.x (restore non-underscore functions in skimage.data) (gh-6099)
- Backport PR #6095 on branch v0.19.x (Preserve backwards compatibility for `channel_axis` parameter in transform functions) (gh-6100)
- Backport PR #6103 on branch v0.19.x (make rank filter test comparisons robust across architectures) (gh-6106)
- Backport PR #6105 on branch v0.19.x (pass a specific random_state into ransac in test_ransac_geometric) (gh-6107)
- Fix two equality comparison bugs in the wheel build script (gh-6098)
- Backport of gh-6109 (Add linker flags to strip debug symbols during wheel building) (gh-6110)
- Pin setuptools maximum in v0.19.x to avoid breaking on planned distutils API changes (gh-6112)
- Avoid potential circular import of rgb2gray (gh-6113)
- Backport PR #6089 on branch v0.19.x (Skip tests requiring fetched data) (gh-6115)
- Backport PR #6118 on branch v0.19.x (Fixes to tests.yml and fixes for expected warnings) (gh-6127)
- Backport PR #6114 on branch v0.19.x (relax test condition to make it more robust to variable CI load) (gh-6128)


3 authors added to this release [alphabetical by first name or login]
---------------------------------------------------------------------
- Gregory R. Lee
- Joshua Newton
- Mark Harfouche

5 reviewers added to this release [alphabetical by first name or login]
-----------------------------------------------------------------------
- Gregory R. Lee
- Juan Nunez-Iglesias
- Marianne Corvellec
- Mark Harfouche
- Stefan van der Walt


scikit-image 0.19.0 (2021-12-03)
================================

We're happy to announce the release of scikit-image v0.19.0!

scikit-image is an image processing toolbox for SciPy that includes algorithms
for segmentation, geometric transformations, color space manipulation,
analysis, filtering, morphology, feature detection, and more.

For more information, examples, and documentation, please visit our website:

https://scikit-image.org

A highlight of this release is the addition of the popular scale-invariant
feature transform (SIFT) feature detector and descriptor. This release also
introduces a perceptual blur metric, new pixel graph algorithms, and most
functions now operate in single-precision when single-precision inputs are
provided. Many other bug fixes, enhancements and performance improvements are
detailed below.

A significant change in this release is in the treatment of multichannel
images. The existing ``multichannel`` argument to functions has been deprecated
in favor of a new ``channel_axis`` argument. ``channel_axis`` can be used to
specify which axis of an array contains channel information (with
``channel_axis=None`` indicating a grayscale image).

scikit-image now uses "lazy loading", which enables users to access the
functions from all ``skimage`` submodules without the overhead of eagerly
importing all submodules. As a concrete example, after calling "import skimage"
a user can directly call a function such as ``skimage.transform.warp`` whereas
previously it would have been required to first "import skimage.transform".

An exciting change on the development side is the introduction of support for
Pythran as an alternative to Cython for generating compiled code. We plan to
keep Cython support as well going forward, so developers are free to use either
one as appropriate. For those curious about Pythran, a good overview was given
in the SciPy 2021 presentation, "Building SciPy Kernels with Pythran" (https://www.youtube.com/watch?v=6a9D9WL6ZjQ).

This release now supports Python 3.7-3.10. Apple M1 architecture (arm64)
support is new to this release. MacOS 12 wheels are provided for
Python 3.8-3.10.


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
  to allow use with a user-supplied histogram. (gh-5543)
- ``skimage.restoration.denoise_bilateral`` added support for images containing
  negative values. (gh-5527)
- The ``skimage.feature`` functions ``blob_dog``, ``blob_doh`` and ``blob_log``
  now support a ``threshold_rel`` keyword argument that can be used to specify
  a relative threshold (in range [0, 1]) rather than an absolute one. (gh-5517)
- Implement lazy submodule importing (gh-5101)
- Implement weighted estimation of geometric transform matrices (gh-5601)
- Added new pixel graph algorithms in ``skimage.graph``:
  ``pixel_graph`` generates a graph (network) of pixels
  according to their adjacency, and ``central_pixel`` finds
  the geodesic center of the pixels. (gh-5602)
- scikit-image now supports use of Pythran in contributed code. (gh-3226)


Documentation
-------------

- A new doc tutorial presenting a 3D biomedical imaging example has been added
  to the gallery (gh-4946). The technical content benefited from conversations
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
- docs: fix simple typo, conversions -> conversions.
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
- Add stitching gallery example (gh-5365)
- Add draft SKIP3: transition to scikit-image 1.0 (gh-5475)
- Mention commit messages in the contribution guidelines. (gh-5504)
- Fix and standardize docstrings for blob detection functions. (gh-5547)
- Update the User Guide to reflect usage of ``channel_axis`` rather than
  ``multichannel``. (gh-5554)
- Update the user guide to use channel_axis rather than multichannel (gh-5556)
- Add hyperlinks to referenced documentation places. (gh-5560)
- Update branching instructions to change the location of the pooch repo.
  (gh-5565)
- Add Notes and References section to the Cascade class docstring. (gh-5568)
- Clarify 2D vs nD in skimage.feature.corner docstrings (gh-5569)
- Fix math formulas in plot_swirl.py example. (gh-5574)
- Update references in texture feature detectors docstrings (gh-5578)
- Update mailing list location to discuss.scientific-python.org forum (gh-5951)
- DOC: Fix docstring in rescale_intensity() (gh-5964)
- Fix slic documentation (gh-5975)
- Update docstring for dilation, which is now nD. (gh-5978)
- Change stitching gallery example thumbnail (gh-5985)
- Add circle and disk to glossary.md (gh-5590)
- Update pixel graphs example (gh-5991)
- Separate entries that have the same description in glossary.md (gh-5592)
- Do not use space before colon in directive name (gh-6002)


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
  is supplied by the user (gh-4903). The specific superpixels produced by
  masked SLIC will not be identical to those produced by prior releases.
- ``exposure.adjust_gamma`` has been accelerated for ``uint8`` images thanks to
  a LUT (gh-4966).
- ``measure.label`` has been accelerated for boolean input images, by using
  ``scipy.ndimage``'s implementation for this case (gh-4945).
- ``util.apply_parallel`` now works with multichannel data (gh-4927).
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
- Improve numerical precision in ``CircleModel.estimate``. (gh-5190)
- Add default keyword argument values to
  ``skimage.restoration.denoise_tv_bregman``, ``skimage.measure.block_reduce``,
  and ``skimage.filters.threshold_local``. (gh-5454)
- Make matplotlib an optional dependency (gh-5990)
- single precision support in skimage.filters (gh-5354)
- Support nD images and labels in label2rgb (gh-5550)
- Regionprops table performance refactor (gh-5576)
- add regionprops benchmark script (gh-5579)
- remove use of apply_along_axes from greycomatrix & greycoprops (gh-5580)
- refactor gabor_kernel for efficiency (gh-5582)
- remove need for channel_as_last_axis decorator in skimage.filters (gh-5584)
- replace use of scipy.ndimage.gaussian_filter with skimage.filters.gaussian
  (gh-5872)
- add channel_axis argument to quickshift (gh-5987)
- add MacOS arm64 wheels (gh-6068)


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
  (gh-4862).
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
- A bug fix to the phase normalization applied within
  ``skimage.register.phase_cross_correlation`` may result in a different result
  as compared to prior releases. The prior behavior of "unnormalized" cross
  correlation is still available by explicitly setting ``normalization=None``.
  There is no change to the masked cross-correlation case, which uses a
  different algorithm.


Bugfixes
--------

- Input ``labels`` argument renumbering in ``skimage.feature.peak_local_max``
  is avoided (gh-5047).
- fix clip bug in resize when anti_aliasing is applied (gh-5202)
- Nonzero values at the image edge are no longer incorrectly marked as a
  boundary when using ``find_bounaries`` with mode='subpixel' (gh-5447).
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
- Work with pooch 1.5.0 for fetching data (gh-5529).
- The ``offsets`` attribute of ``skimage.graph.MCP`` is now public. (gh-5547)
- Fix io.imread behavior with pathlib.Path inputs (gh-5543)
- Make scikit-image imports from Pooch, compatible with pooch >= 1.5.0.
  (gh-5529)
- Fix several broken doctests and restore doctesting on GitHub Actions.
  (gh-5505)
- Fix broken doctests in ``skimage.exposure.histogram`` and
  ``skimage.measure.regionprops_table``. (gh-5522)
- Rescale image consistently during SLIC superpixel segmentation. (gh-5518)
- Correct phase correlation in ``skimage.register.phase_cross_correlation``.
  (gh-5461)
- Fix hidden attribute 'offsets' in skimage.graph.MCP (gh-5551)
- fix phase_cross_correlation for 3D with reference masks (gh-5559)
- fix return shape of blob_log and blob_dog when no peaks are found (gh-5567)
- Fix find contours key error (gh-5577)
- Refactor measure.ransac and add warning when the estimated model is not valid
  (gh-5583)
- Restore integer image rescaling for edge filters (gh-5589)
- trainable_segmentation: re-raise in error case (gh-5600)
- allow regionprops_table to be called with deprecated property names (gh-5908)
- Fix weight calculation in fast mode of non-local means (gh-5923)
- fix for #5948: lower boundary 1 for kernel_size in equalize_adapthist
  (gh-5949)
- convert pathlib.Path to str in imsave (gh-5971)
- Fix slic spacing (gh-5974)
- Add small regularization to avoid zero-division in richardson_lucy (gh-5976)
- Fix benchmark suite (watershed function was moved) (gh-5982)
- catch QhullError and return empty array (``convex_hull``) (gh-6008)
- add property getters for all newly deprecated regionprops names (gh-6000)
- Fix the estimation of ellipsoid axis lengths in the 3D case (gh-6013)
- Fix peak local max segfault (gh-6035)
- Avoid circular import errors when EAGER_IMPORT=1 (gh-6042)
- remove all use of the deprecated distutils package (gh-6044)


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
- Attempting to warp a boolean image with ``order > 0`` now raises a
  ValueError.
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
- standardize on ``num_iter`` for parameters describing the number of iterations
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
  library. The ``selem`` argument is now deprecated.


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
- Add missing import in lch2lab docstring example (gh-5998)
- Prefer importing build_py and sdist from setuptools (gh-6007)
- Reintroduce skimage.test utility (gh-5909)


Other Updates
-------------
- Refactor np.random.x to use np.random.Generator.
- Avoid warnings about use of deprecated ``scipy.linalg.pinv2``.
- Simplify resize implementation using new SciPy 1.6 zoom option.
- Fix duplicate test function names in ``test_unsharp_mask.py``.
- Benchmarks: ``fix ResizeLocalMeanSuite.time_resize_local_mean`` signature.
- Prefer use of new-style NumPy random API in tests (gh-5450)
- Add fixture enforcing SimpleITK I/O in test_simpleitk.py (gh-5526)
- MNT: Remove unused stat import from skimage data (gh-5566)
- MAINT: Remove unused imports (gh-5595)
- MAINT: Refactor duplicated tests, remove unnecessary assignments and
  variables (gh-5596)
- Remove obsolete lazy import (gh-5992)
- Lazily load data_dir into the top-level namespace (gh-5996)
- Update scipy requirement to 1.4.1 and use scipy.fft instead of scipy.fftpack
  (gh-5999)
- Remove lines generating Requires metadata (gh-6017)
- Update wheel builds to include Python 3.10 (gh-6021)
- Update pyproject.toml to handle Python 3.10 and Apple arm64 (gh-6022)
- Add python 3.10 test runs on GitHub Actions and Appveyor (gh-6027)
- Pin sphinx to <4.3 until new sphinx-gallery release is available (gh-6029)
- Relax a couple of equality tests causing i686 test failures on cibuildwheel
  (gh-6031)
- Avoid matplotlib import overhead during 'import skimage' (gh-6032)
- Update sphinx gallery pin (gh-6034)


Contributors to this release
----------------------------


80 authors added to this release [alphabetical by first name or login]
----------------------------------------------------------------------
- Abhinavmishra8960 (Abhinavmishra8960)
- abouysso
- Alessia Marcolini
- Alex Brooks
- Alexandre de Siqueira
- Andres Fernandez
- Andrew Hurlbatt
- andrewnags (andrewnags)
- Antoine Bierret
- BMaster123 (BMaster123)
- Boaz Mohar
- Bozhidar Karaargirov
- Carlos Andrés Álvarez Restrepo
- Christoph Gohlke
- Christoph Sommer
- Clement Ng
- cmarasinou
- Cris Luengo
- David Manthey
- Devanshu Shah
- Dhiraj Kumar Sah
- divyank agarwal
- Egor Panfilov
- Emmanuelle Gouillart
- Erik Reed
- erykoff (erykoff)
- Fabian Schneider
- Felipe Gutierrez-Barragan
- François Boulogne
- Fred Bunt
- Fukai Yohsuke
- Gregory R. Lee
- Hari Prasad
- Harish Venkataraman
- Harshit Dixit
- Ian Hunt-Isaak
- Jaime Rodríguez-Guerra
- Jan-Hendrik Müller
- Janakarajan Natarajan
- Jenny Vo
- john lee
- Jonathan Striebel
- Joseph Fox-Rabinovitz
- Juan Antonio Barragan Noguera
- Juan Nunez-Iglesias
- Julien Jerphanion
- Jurneo
- klaussfreire (klaussfreire)
- Larkinnjm1 (Larkinnjm1)
- Lars Grüter
- Mads Dyrmann
- Marianne Corvellec
- Marios Achilias
- Mark Boer
- Mark Harfouche
- Matthias Bussonnier
- Mauro Silberberg
- Max Frei
- michalkrawczyk (michalkrawczyk)
- Niels Cautaerts
- Pamphile ROY
- Pradyumna Rahul
- R
- Raphael
- Riadh Fezzani
- Robert Haase
- Sebastian Gonzalez Tirado
- Sebastián Vanrell
- serge-sans-paille (serge-sans-paille)
- Stefan van der Walt
- t.ae
- that1solodev (Xyno18)
- Thomas Walter
- Tim Gates
- Tom Flux
- Vinicius D. Cerutti
- Volker Hilsenstein
- WeiChungChang
- yacth
- Yash-10 (Yash-10)

63 reviewers added to this release [alphabetical by first name or login]
------------------------------------------------------------------------
- Abhinavmishra8960
- Alessia Marcolini
- Alex Brooks
- Alexandre de Siqueira
- Andres Fernandez
- Andrew Hurlbatt
- andrewnags
- BMaster123
- Boaz Mohar
- Carlos Andrés Álvarez Restrepo
- Clement Ng
- Cris Luengo
- Dan Schult
- David Manthey
- Egor Panfilov
- Emmanuelle Gouillart
- erykoff
- Fabian Schneider
- Felipe Gutierrez-Barragan
- François Boulogne
- Fukai Yohsuke
- Genevieve Buckley
- Gregory R. Lee
- Jan Eglinger
- Jan-Hendrik Müller
- Janakarajan Natarajan
- Jarrod Millman
- Jirka Borovec
- Joan Massich
- Johannes Schönberger
- john lee
- Jon Crall
- Joseph Fox-Rabinovitz
- Josh Warner
- Juan Nunez-Iglesias
- Julien Jerphanion
- Kenneth Hoste
- klaussfreire
- Larkinnjm1
- Lars Grüter
- Marianne Corvellec
- Mark Boer
- Mark Harfouche
- Matthias Bussonnier
- Max Frei
- michalkrawczyk
- Niels Cautaerts
- Pamphile ROY
- Pomax
- R
- Raphael
- Riadh Fezzani
- Robert Kern
- Ross Barnowski
- Sebastian Berg
- Sebastian Gonzalez Tirado
- Sebastian Wallkötter
- serge-sans-paille
- Stefan van der Walt
- t.ae
- Vinicius D. Cerutti
- Volker Hilsenstein
- Yash-10

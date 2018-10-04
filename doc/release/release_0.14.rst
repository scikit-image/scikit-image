Announcement: scikit-image 0.14.1
=================================

We're happy to announce the release of scikit-image v0.14.1!

scikit-image is an image processing toolbox for SciPy that includes algorithms
for segmentation, geometric transformations, color space manipulation,
analysis, filtering, morphology, feature detection, and more.

This is our first release under our Long Term Support for 0.14 policy. As a
reminder, 0.14 is the last release to support Python 2.7, but it will be
updated with bug fixes and popular features until January 1st, 2020.

This release contains the following changes from 0.14.0:


Bug fixes
---------
- ``skimage.color.adapt_rgb`` was applying input functions to the wrong axis
  (#3097)
- ``CollectionViewer`` now indexes correctly (it had been broken by an update
  to NumPy indexing) (#3288)
- Handle deprecated indexing-by-list and NumPy ``matrix`` from NumPy 1.15
  (#3238, #3242, #3292)
- Fix incorrect inertia tensor calculation (#3303) (Special thanks to JP Cornil
  for reporting this bug and for their patient help with this fix)
- Fix missing comma in ``__all__`` listing of ``moments_coord_central``, so it
  and ``moments_normalized`` can now be correctly imported from the ``measure``
  namespace (#3374)
- Fix background color in ``label2rgb(..., kind='avg')`` (#3280)

Enhancements
------------
- "Reflect" mode in transforms now works fine when an image dimension has size
  1 (#3174)
- ``img_as_float`` now allows single-precision (32-bit) float arrays to pass
  through unmodified, rather than being up-converted to 64-bit (#3110, #3052,
  #3391)
- Speed up rgb2gray computation (#3187)
- The scikit-image viewer now works with different PyQt versions (#3157)
- The ``cycle_spin`` function for enhanced denoising works single-threaded
  when dask is not installed now (#3218)
- scikit-image's ``io`` module will no longer inadvertently set the matplotlib
  backend when imported (#3243)
- Fix deprecated ``get`` keyword from dask in favor of ``scheduler`` (#3366)
- Add missing ``cval`` parameter to threshold_local (#3370)


API changes
-----------
- Remove deprecated ``dynamic_range`` in ``measure.compare_psnr`` (#3313)

Documentation
-------------
- Improve the documentation on data locality (#3127)
- Improve the documentation on dealing with video (#3176)
- Update broken link for Canny filter documentation (#3276)
- Fix incorrect documentation for the ``center`` parameter of
  ``skimage.transform.rotate`` (#3341)
- Fix incorrect formatting of docstring in ``measure.profile_line`` (#3236)

Build process / development
---------------------------
- Ensure Cython is 0.23.4 or newer (#3171)
- Suppress warnings during testing (#3143)
- Fix skimage.test (#3152)
- Don't upload artifacts to AppVeyor (there is no way to delete them) (#3315)
- Remove ``import *`` from the scikit-image package root (#3265)
- Allow named non-core contributors to issue MeeseeksDev commands (#3357,
  #3358)
- Add testing in Python 3.7 (#3359)
- Add license file to the binary distribution (#3322)
- ``lookfor`` is no longer defined in ``__init__.py`` but rather imported to it
  (#3162)
- Add ``pyproject.toml`` to ensure Cython is present before building (#3295)
- Add explicit Python version Trove classifiers for PyPI (#3417)
- Ignore known test failures in 32-bit releases, allowing 32-bit wheel builds
  (#3434)
- Ignore failure to raise floating point warnings on certain ARM platforms
  (#3337)
- Fix tests to be compatible with PyWavelets 1.0 (#3406)

Credits
-------
Made with commits from (alphabetical by last name):

- François Boulogne
- Genevieve Buckley
- Sean Budd
- Matthias Bussonnier
- Sarkis Dallakian
- Christoph Deil
- François-Michel De Rainville
- Emmanuelle Gouillart
- Yaroslav Halchenko
- Mark Harfouche
- Jonathan Helmus
- Gregory Lee
- @Legodev
- Matt McCormick
- Juan Nunez-Iglesias
- Egor Panfilov
- Jesse Pangburn
- Johannes Schönberger
- Stefan van der Walt

Reviewed by (alphabetical by last name):

- François Boulogne
- Emmanuelle Gouillart
- Mark Harfouche
- Juan Nunez-Iglesias
- Egor Panfilov
- Stéfan van der Walt
- Josh Warner

And with the special support of [MeeseeksDev](https://github.com/MeeseeksBox),
created by Matthias Bussonnier


Announcement: scikit-image 0.14.0
=================================

We're happy to announce the release of scikit-image v0.14.0!

scikit-image is an image processing toolbox for SciPy that includes algorithms
for segmentation, geometric transformations, color space manipulation,
analysis, filtering, morphology, feature detection, and more.

This is the last major release with official support for Python 2.7. Future
releases will be developed using Python 3-only syntax.

However, 0.14 is a long-term support (LTS) release and will receive bug fixes
and backported features deemed important (by community demand) until January
1st 2020 (end of maintenance for Python 2.7; see PEP 373 for details).

For more information, examples, and documentation, please visit our website:

http://scikit-image.org


New Features
------------
- Lookfor function to search across the library: ``skimage.lookfor``. (#2713)
- nD support for ``skimage.transform.rescale``, ``skimage.transform.resize``,
  and ``skimage.transform.pyramid_*`` transforms. (#1522)
- Chan-Vese segmentation algorithm. (#1957)
- Manual segmentation with matplotlib for fast data annotation:
  ``skimage.future.manual_polygon_segmentation``,
  ``skimage.future.manual_lasso_segmentation``. (#2584)
- Hysteresis thresholding:
  ``skimage.filters.apply_hysteresis_threshold``. (#2665)
- Segmentation with morphological snakes:
  ``skimage.segmentation.morphological_chan_vese`` (2D),
  ``skimage.segmentation.morphological_geodesic_active_contour`` (2D and 3D). (#2791)
- nD support for image moments: ``skimage.measure.moments_central``,
  ``skimage.measure.moments_central``, ``skimage.measure.moments_normalized``,
  ``skimage.measure.moments_hu``. This change leads to 3D/nD compatibility for
  many regionprops. (#2603)
- Image moments from coordinate input: ``skimage.measure.moments_coords``,
  ``skimage.measure.moments_coords_central``. (#2859)
- Added 3D support to ``blob_dog`` and ``blob_log``. (#2854)
- Inertia tensor and its eigenvalues can now be computed outside of
  regionprops; available in ``skimage.measure.inertia_tensor``. (#2603)
- Cycle-spinning function for approximating shift-invariance by averaging
  results from a series of spatial shifts:
  ``skimage.restoration.cycle_spin``. (#2647)
- Haar-like feature: ``skimage.feature.haar_like_feature``,
  ``skimage.feature.haar_like_feature_coord``,
  ``skimage.feature.draw_haar_like_feature``. (#2848)
- Data generation with random_shapes function:
  ``skimage.draw.random_shapes``. (#2773)
- Subset of LFW (Labeled Faces in the Wild) database:
  ``skimage.data.cbcl_face_database``. (#2905)
- Fully reworked montage function (now with a better padding behavior):
  ``skimage.util.montage``. (#2626)
- YDbDr colorspace conversion routines: ``skimage.color.rgb2ydbdr``,
  ``skimage.color.ydbdr2rgb``. (#3018)


Improvements
------------
- ``VisuShrink`` method for ``skimage.restoration.denoise_wavelet``. (#2470)
- New ``max_ratio`` parameter for ``skimage.feature.match_descriptors``. (#2472)
- ``skimage.transform.resize`` and ``skimage.transform.rescale`` have a new
  ``anti_aliasing`` option to avoid aliasing artifacts when down-sampling
  images. (#2802)
- Support for multichannel images for ``skimage.feature.hog``. (#2870)
- Non-local means denoising (``skimage.restoration.denoise_nl_means``) has
  a new optional parameter, ``sigma``, that can be used to specify the noise
  standard deviation. This enables noise-robust patch distance estimation. (#2890)
- Mixed dtypes support for ``skimage.measure.compare_ssim``,
  ``skimage.measure.compare_psnr``, etc. (#2893)
- New ``alignment`` parameter in ``skimage.feature.plot_matches``. (#2955)
- New ``seed`` parameter in ``skimage.transform.probabilistic_hough_line``. (#2960)
- Various performance improvements. (#2821, #2878, #2967, #3035, #3056, #3100)


Bugfixes
--------
- Fixed ``skimage.measure.regionprops.bbox_area`` returning incorrect value. (#2837)
- Changed gradient and L2-Hys norm computation in ``skimage.feature.hog``
  to closely follow the paper. (#2864)
- Fixed ``skimage.color.convert_colorspace`` not working for YCbCr, YPbPr. (#2780)
- Fixed incorrect composition of projective tranformation with inverse transformation. (#2826)
- Fixed bug in random walker appearing when seed pixels are isolated inside pruned zones. (#2946)
- Fixed ``rescale`` not working properly with different rescale factors in multichannel case. (#2959)
- Fixed float and integer dtype support in ``skimage.util.invert``. (#3030)
- Fixed ``skimage.measure.find_contours`` raising StopIteration on Python 3.7. (#3038)
- Fixed platform-specific issues appearing in Windows and/or 32-bit environments. (#2867, #3033)


API Changes
-----------
- ``skimage.util.montage.`` namespace has been removed, and
  ``skimage.util.montage.montage2d`` function is now available as
  ``skimage.util.montage2d``.
- ``skimage.morphology.binary_erosion`` now uses ``True`` as border
  value, and is now consistent with ``skimage.morphology.erosion``.


Deprecations
------------
- ``freeimage`` plugin has been removed from ``skimage.io``.
- ``skimage.util.montage2d`` is deprecated and will be removed in 0.15.
  Use ``skimage.util.montage`` function instead.
- ``skimage.novice`` is deprecated and will be removed in 0.16.
- ``skimage.transform.resize`` and ``skimage.transform.rescale`` have a new
  ``anti_aliasing`` option that avoids aliasing artifacts when down-sampling
  images. This option will be enabled by default in 0.15.
- ``regionprops`` will use row-column coordinates in 0.16. You can start
  using them now with ``regionprops(..., coordinates='rc')``. You can silence
  warning messages, and retain the old behavior, with
  ``regionprops(..., coordinates='xy')``. However, that option will go away
  in 0.16 and result in an error. This change has a number of consequences.
  Specifically, the "orientation" region property will measure the
  anticlockwise angle from a *vertical* line, i.e. from the vector (1, 0) in
  row-column coordinates.
- ``skimage.morphology.remove_small_holes`` ``min_size`` argument is deprecated
  and will be removed in 0.16. Use ``area_threshold`` instead.


Contributors to this release
----------------------------

- Alvin
- Norman Barker
- Brad Bazemore
- Leonid Bloch
- Benedikt Boecking
- Jirka Borovec
- François Boulogne
- Larry Bradley
- Robert Bradshaw
- Matthew Brett
- Floris van Breugel
- Alex Chum
- Yannick Copin
- Nethanel Elzas
- Kira Evans
- Christoph Gohlke
- GGoussar
- Jens Glaser
- Peter Goldsborough
- Emmanuelle Gouillart
- Ben Hadfield
- Mark Harfouche
- Scott Heatwole
- Gregory R. Lee
- Guillaume Lemaitre
- Theodore Lindsay
- Kevin Mader
- Jarrod Millman
- Vinicius Monego
- Pradyumna Narayana
- Juan Nunez-Iglesias
- Kesavan PS
- Egor Panfilov
- Oleksandr Pavlyk
- Justin Pinkney
- Robert Pollak
- Jonathan Reich
- Émile Robitaille
- Rose Zhao
- Alex Rothberg
- Arka Sadhu
- Max Schambach
- Johannes Schönberger
- Sourav Singh
- Kesavan Subburam
- Matt Swain
- Saurav R. Tuladhar
- Nelle Varoquaux
- Viraj
- David Volgyes
- Stefan van der Walt
- Thomas Walter
- Scott Warchal
- Josh Warner
- Nicholas Weir
- Sera Yang
- Chiang, Yi-Yo
- corrado9999
- ed1d1a8d
- eepaillard
- leaprovenzano
- mikigom
- mrastgoo
- mutterer
- pmneila
- timhok
- zhongzyd


We'd also like to thank all the people who contributed their time to perform the reviews:

- Leonid Bloch
- Jirka Borovec
- François Boulogne
- Matthew Brett
- Thomas A Caswell
- Kira Evans
- Peter Goldsborough
- Emmanuelle Gouillart
- Almar Klein
- Gregory R. Lee
- Joan Massich
- Juan Nunez-Iglesias
- Faraz Oloumi
- Daniil Pakhomov
- Egor Panfilov
- Dan Schult
- Johannes Schönberger
- Steven Silvester
- Alexandre de Siqueira
- Nelle Varoquaux
- Stefan van der Walt
- Josh Warner
- Eric Wieser


Full list of changes
--------------------
This release is the result of 14 months of work.
It contains the following 186 merged pull requests by 67 committers:

- n-dimensional rescale, resize, and pyramid transforms (#1522)
- Segmentation: Implemention of a simple Chan-Vese Algorithm (#1957)
- JPEG quality argument in imsave (#2063)
- improve geometric models fitting (line, circle) using LSM (#2433)
- Improve input parameter handling in `_sift_read` (#2452)
- Remove broken test in `_shared/tests/test_interpolation.py` (#2454)
- [MRG] Pytest migration (#2468)
- Add VisuShrink method for `denoise_wavelet` (#2470)
- Ratio test for descriptor matching (#2472)
- Make HOG visualization use midpoints of orientation bins (#2525)
- DOC: Add example for rescaling/resizing/downscaling (#2560)
- Gallery random walker: Rescale image range to -1, 1 (#2575)
- Update conditional requirement for PySide (#2578)
- Add configuration file for `pep8_speaks` (#2579)
- Manual segmentation tool with matplotlib (#2584)
- Website updates (documentation build) (#2585)
- Update the release process notes (#2593)
- Defer matplotlib imports (#2596)
- Spelling: replaces colour by color (#2598)
- Add nD support to image moments computation (#2603)
- Set xlim and ylim in rescale gallery example (#2606)
- Reduce runtime of local_maxima gallery example (#2608)
- MAINT _shared.testing now contains pytest's useful functions (#2614)
- error message misspelled, integral to integer (#2615)
- Respect standard notations for images in functions arguments (#2617)
- MAINT: remove unused argument in private inpainting function (#2618)
- MAINT: some minor edits on Chan Vese segmentation (#2619)
- Fix UserWarning: Unknown section Example (#2620)
- Eliminate some TODOs for 0.14 (#2621)
- Clean up and fix bug in ssim tests (#2622)
- Add padding_width to montage2d and add montage_rgb (#2626)
- Add tests covering erroneous input to morphology.watershed (#2631)
- Fix name of code coverage tool (#2638)
- MAINT: Remove undefined attributes in skimage.filters (#2643)
- Improve the support for 1D images in `color.gray2rgb`  (#2645)
- ENH: add cycle spinning routine (#2647)
- as_gray replaces as_grey in imread() and load() (#2652)
- Fix AppVeyor pytest execution (#2658)
- More TODOs for 0.14 (#2659)
- pin sphinx to <1.6 (#2662)
- MAINT: use relative imports instead of absolute ones (#2664)
- Add hysteresis thresholding function (#2665)
- Improve hysteresis docstring (#2669)
- Add helper functions img_as_float32 and img_as_float64 (#2673)
- Remove unnecessary assignment in pxd file. (#2683)
- Unused var and function call in documentation example (#2684)
- Make `imshow_collection` to plot images on a grid of convenient aspect ratio (#2689)
- Fix typo in Chan-Vese docstrings (#2692)
- Fix data type error with marching_cubes_lewiner(allow_degenerate=False) (#2694)
- Add handling for uniform arrays when finding local extrema. (#2699)
- Avoid uneccesary copies in skimage.morphology.label (#2701)
- Deprecate `visualise` in favor of `visualize` in `skimage.feature.hog` (#2705)
- Remove alpha channel when saving to jpg format (#2706)
- Tweak in-place installation instructions (#2712)
- Add `skimage.lookfor` function (#2713)
- Speedup image dtype conversion by switching to `asarray` (#2715)
- MAINT reorganizing CI-related scripts (#2718)
- added rect function to draw module (#2719)
- Remove duplicate parameter in `skimage.io.imread` docstring (#2725)
- Add support for 1D arrays for grey erosion (#2727)
- Build with Xcode 9 beta 3, MacOS 10.12 (#2730)
- Travis docs one platform (#2732)
- Install documentation build requirements on Travis-CI (#2737)
- Add reference papers for `restoration.inpaint_biharmonic` (#2738)
- Completely remove `freeimage` plugin from `skimage.io` (#2744)
- Implementation and test fix for shannon_entropy calculation. (#2749)
- Minor cleanup (#2750)
- Add notes on testing to CONTRIBUTING (#2751)
- Update OSX install script (#2752)
- fix bug in horizontal seam_carve and seam_carve test. issue :#2545 (#2754)
- Recommend merging instead of rebasing, to lower contribution barrier (#2757)
- updated second link, first link still has paywall (#2768)
- DOC: set_color docstring, in-place said explicitly (#2771)
- Add module for generating random, labeled shapes (#2773)
- Ignore known failures (#2774)
- Update testdoc (#2775)
- Remove bento support (#2776)
- AppVeyor supports dot-file-style (#2779)
- Fix bug in `color.convert_colorspace` for YCbCr, YPbPr (#2780)
- Reorganizing requirements (#2781)
- WIP: Deal with long running command on travis (#2782)
- Deprecate the novice module (#2742) (#2784)
- Document mentioning deprecations in the release notes (#2785)
- [WIP] FIX Swirl center coordinates are reversed (#2790)
- Implementation of the Morphological Snakes (#2791)
- Merge TASKS.txt with CONTRIBUTING.txt (#2800)
- Add Gaussian filter-based antialiasing to resize (#2802)
- Add morphological snakes to release notes (#2803)
- Return empty array if hough_line_peaks detects nothing (#2805)
- Add W503 to pep8speaks ignore. (#2816)
- Slice PIL palette correctly using extreme image value. (#2818)
- Move INSTALL to top-level (#2819)
- Make simple watershed fast again (#2821)
- The gallery now points to the stable docs (#2822)
- Adapt AppVeyor to use Python.org dist, and remove install script (#2823)
- Remove pytest yield (#2824)
- Bug fix in projective tranformation composition with inverse transformation (#2826)
- FIX: add estimate_sigma to __all__ in restoration module (#2829)
- Switch from LaTeX to MathJax in doc build (#2832)
- Docstring fixes for better formula formatting (#2834)
- Fix regionprops.bbox_area bug (#2837)
- MAINT: add Python 3.6 to appveyor, small edits (#2840)
- Allow convex area calculation in 3D for regionprops (#2847)
- [MRG] DOC fix documentation build (#2851)
- Change default args from list to tuple in `feature.draw_multiblock_lbp` (#2852)
- Add 3D support to `blob_dog` and `blob_log` (#2854)
- Update compare_nrmse docstring (#2855)
- Fix link order in example (#2858)
- Add Computation of Image Moments to Coordinates (#2859)
- Revert gradient formula, modify the deprecation warning, and fix L2-Hys norm in `skimage.feature.hog` (#2864)
- OverflowError: Python int too large to convert to C long on win-amd64-py2.7 (#2867)
- Fix `skimage.measure.centroid` and add test coverage (#2869)
- Add multichannel support to `feature.hog` (#2870)
- Remove scipy version check in `active_contour` (#2871)
- Update DOI reference in `measure.compare_ssim` (#2872)
- Fix randomness and expected ranges for RGB in `test_random_shapes`. (#2877)
- Nl means fixes for large datasets (#2878)
- Make `test_random_shapes` use internally shipped testing tools (#2879)
- DOC: Update docstring for is_low_constrast to match function signature (#2883)
- Update URL in RAG docstring (#2885)
- Fix spelling typo in NL means docstring (#2887)
- noise-robust patch distance estimation for non-local means (#2890)
- Allow mixed dtypes in compare_ssim, compare_psnr, etc. (#2893)
- EHN add Haar-like feature (#2896)
- Add CBCL face database subset to `skimage.data` (#2897)
- EXA example for haar like features (#2898)
- Install documentation dependencies on all builds (#2900)
- Improve LineModelND doc strings (#2903)
- Add a subset of LFW dataset to `skimage.data` (#2905)
- Update default parameter values in the docstring of `skimage.restoration.unsupervised_wiener` (#2906)
- Revert "Add CBCL face database subset to `skimage.data`" (#2907)
- remove unused parameter 'n_segments' in `_enforce_label_connectivity_cython()` (#2908)
- Update six version to make pytest_cov work (#2909)
- Fix typos in `draw._random_shapes._generate_triangle_mask` docstring (#2914)
- do not assume 3 channels during non-local means denoising (#2922)
- add missing cdef in _integral_image_3d (non-local means) (#2923)
- Replace `morphology.remove_small_holes` argument `min_size` with `area_threshold` (#2924)
- Ensure warning to provide bool array is warranted (#2930)
- Remove copyright notice with permission of the author (Thomas Lewiner) (#2932)
- Fix link to Windows binaries in README. (#2934)
- Handle NumPy 1.14 API changes (#2935)
- Specify `gradient` parameter docstring in `compare_ssim` (#2937)
- Fixed broken link on LBP documentation (#2941)
- Corrected bug related to border value of morphology.binary_erosion (#2945)
- Correct bug in random walker when seed pixels are isolated inside pruned zones (#2946)
- Fix Cython compilation warnings in NL Means and Watershed (#2947)
- Add `alignment` parameter to `feature.plot_matches` (#2955)
- Raise warning when attempting to save boolean image (#2957)
- Allow different rescale factors in multichannel warp (#2959)
- Add seed parameter to probabilistic_hough_line (#2960)
- Minor style fixes for #2946 (#2961)
- Build on fewer AppVeyor platforms to avoid timeout (#2962)
- Watershed segmentation: make usable for large arrays (#2967)
- Mark data_range as being a float (#2971)
- Use correct NumPy version comparison in pytest configuration (#2975)
- Handle matplotlib 2.2 pre-release deprecations (#2977)
- Bugfix LineModelND.residuals does not use the optional parameter `params` (#2979)
- Return empty list on flat images with hough_ellipse #2820 (#2996)
- Add release notes for 0.13.1 (#2999)
- MAINT: PIL removed saving RGBA images as jpeg files (#3004)
- Ensure stdev is always nonnegative in _mean_std (#3008)
- Add citation information to README (#3013)
- Add YDbDr colorspace conversion routines (#3018)
- Minor style and documentation updates for #2859 (#3023)
- `draw.random_shapes` API improvements (#3029)
- Type dependent inversion (#3030)
- Fix ValueError: Buffer dtype mismatch, expected 'int64_t' but got 'int' on win_amd64 (#3033)
- Replace pow function calls in Cython modules to fix performance issues on Windows (#3035)
- Add __pycache__ and .cache to .gitignore. (#3037)
- Fix RuntimeError: generator raised StopIteration on Python 3.7 (#3038)
- Fix invert tests (#3039)
- Fix examples not displaying figures (#3040)
- Correct reference for the coins sample image (#3042)
- Switch to basis numpy int dtypes in dtype_range (#3050)
- speedup img_as_float by making division multiplication and avoiding unecessary allocation (#3056)
- For sparse CG solver, provide atol=0 keyword for SciPy >= 1.1 (#3063)
- Update dependencies and deprecations to fix Travis builds (#3072)
- Sanitizing marching_cubes_lewiner spacing input argument (#3074)
- Allow convex_hull_image on empty images (#3076)
- v0.13.x: Backport NumPy 1.14 compatibility (#3085)
- Force Appveyor to fail on failed tests (#3093)
- Add `threshold_local` to `filters` module namespace (#3096)
- Replace grey by gray where no deprecation is needed (#3098)
- Optimize _probabilistic_hough_line function (#3100)
- Rebuild docs upon deploy to ensure Javascript is generated (#3104)
- Fix random gallery script generation (#3106)

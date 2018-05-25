Announcement: scikit-image 0.14.0
=================================

We're happy to announce the release of scikit-image v0.14.0!

scikit-image is an image processing toolbox for SciPy that includes algorithms
for segmentation, geometric transformations, color space manipulation,
analysis, filtering, morphology, feature detection, and more.

This is the last major release with official support for Python 2.7. Future
releases will be developed using Python 3 only-syntax.

However, 0.14 is a long-time support (LTS) release and will receive bug fixes
and backported features deemed important (by community demand) for several years.

For more information, examples, and documentation, please visit our website:

http://scikit-image.org


New Features
------------
- Lookfor function to search across the library: ``skimage.lookfor``. (#2713)
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
  ``skimage.measure.moments_hu``. This change leads to nD compatibility for many
  regionprops. (#2603)
- Image moments from coordinate input: ``skimage.measure.moments_coords``,
  ``skimage.measure.moments_coords_central``. (#2859)
- Inertia tensor and its eigenvalues can now be computed outside of
  regionprops; available in ``skimage.measure.inertia_tensor``. (#2603)
- Cycle-spinning function for approximating shift-invariance by averaging
  results from a series of spatial shifts:
  ``skimage.restoration.cycle_spin``. (#2647)
- Haar-like feature: ``skimage.feature.haar_like_feature``,
  ``skimage.feature.haar_like_feature_coord``,
  ``skimage.feature.draw_haar_like_feature``. (#2848)
- Data generation with random_shapes function:
  ``skimage.draw.random_shapes``. (#277)
- Subset of LFW (Labeled Faces in the Wild) database:
  ``skimage.data.cbcl_face_database``. (#2905)
- Fully reworked montage function (now with a better padding behavior):
  ``skimage.util.montage``. (#2626)


Improvements
------------
- ``VisuShrink`` method for ``skimage.restoration.denoise_wavelet``. (#2470)
- ``skimage.transform.resize`` and ``skimage.transform.rescale`` have a new
  ``anti_aliasing`` option to avoid aliasing artifacts when down-sampling
  images. (#2802)
- Support for multichannel images for ``skimage.feature.hog``. (#2870)
- Non-local means denoising (``skimage.restoration.denoise_nl_means``) has
  a new optional parameter, ``sigma``, that can be used to specify the noise
  standard deviation. This enables noise-robust patch distance estimation. (#2890)
- New ``alignment`` parameter in ``skimage.feature.plot_matches``. (#2955)


API Changes
-----------
- ``skimage.util.montage.`` namespace has been removed, and
  ``skimage.util.montage.montage2d`` function is now available as
  ``skimage.util.montage2d``.
- ``skimage.morphology.binary_erosion`` now uses ``True`` as border
  value, and is now consistent with ``skimage.morphology.erosion``.


Deprecations
------------
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


Highlights
----------
This release is the result of 8 months of work.
It contains the following 105 merged pull requests by 42 authors:

- Optimize _probabilistic_hough_line function (#3100)
- Replace grey by gray where no deprecation is needed (#3098)
- Add `threshold_local` to `filters` module namespace (#3096)
- Force Appveyor to fail on failed tests (#3093)
- v0.13.x: Backport NumPy 1.14 compatibility (#3085)
- Allow convex_hull_image on empty images (#3076)
- Sanitizing marching_cubes_lewiner spacing input argument (#3074)
- Update dependencies and deprecations to fix Travis builds (#3072)
- For sparse CG solver, provide atol=0 keyword for SciPy >= 1.1 (#3063)
- speedup img_as_float by making division multiplication and avoiding unecessary allocation (#3056)
- Switch to basis numpy int dtypes in dtype_range (#3050)
- Correct reference for the coins sample image (#3042)
- Fix examples not displaying figures (#3040)
- Fix invert tests (#3039)
- Fix RuntimeError: generator raised StopIteration on Python 3.7 (#3038)
- Add __pycache__ and .cache to .gitignore. (#3037)
- Replace pow function calls in Cython modules to fix performance issues on Windows (#3035)
- Fix ValueError: Buffer dtype mismatch, expected 'int64_t' but got 'int' on win_amd64 (#3033)
- Type dependent inversion (#3030)
- `draw.random_shapes` API improvements (#3029)
- Minor style and documentation updates for #2859 (#3023)
- Add YDbDr colorspace conversion routines (#3018)
- Add citation information to README (#3013)
- Ensure stdev is always nonnegative in _mean_std (#3008)
- MAINT: PIL removed saving RGBA images as jpeg files (#3004)
- Add release notes for 0.13.1 (#2999)
- Return empty list on flat images with hough_ellipse #2820 (#2996)
- Bugfix LineModelND.residuals does not use the optional parameter `params` (#2979)
- Handle matplotlib 2.2 pre-release deprecations (#2977)
- Use correct NumPy version comparison in pytest configuration (#2975)
- Mark data_range as being a float (#2971)
- Watershed segmentation: make usable for large arrays (#2967)
- Build on fewer AppVeyor platforms to avoid timeout (#2962)
- Minor style fixes for #2946 (#2961)
- Add seed parameter to probabilistic_hough_line (#2960)
- Allow different rescale factors in multichannel warp (#2959)
- Raise warning when attempting to save boolean image (#2957)
- Add `alignment` parameter to `feature.plot_matches` (#2955)
- Fix Cython compilation warnings in NL Means and Watershed (#2947)
- Correct bug in random walker when seed pixels are isolated inside pruned zones (#2946)
- Corrected bug related to border value of morphology.binary_erosion (#2945)
- Fixed broken link on LBP documentation (#2941)
- Specify `gradient` parameter docstring in `compare_ssim` (#2937)
- Handle NumPy 1.14 API changes (#2935)
- Fix link to Windows binaries in README. (#2934)
- Remove copyright notice with permission of the author (Thomas Lewiner) (#2932)
- Ensure warning to provide bool array is warranted (#2930)
- Replace `morphology.remove_small_holes` argument `min_size` with `area_threshold` (#2924)
- add missing cdef in _integral_image_3d (non-local means) (#2923)
- do not assume 3 channels during non-local means denoising (#2922)
- Fix typos in `draw._random_shapes._generate_triangle_mask` docstring (#2914)
- Update six version to make pytest_cov work (#2909)
- remove unused parameter 'n_segments' in `_enforce_label_connectivity_cython()` (#2908)
- Revert "Add CBCL face database subset to `skimage.data`" (#2907)
- Update default parameter values in the docstring of `skimage.restoration.unsupervised_wiener` (#2906)
- Add a subset of LFW dataset to `skimage.data` (#2905)
- Improve LineModelND doc strings (#2903)
- Install documentation dependencies on all builds (#2900)
- EXA example for haar like features (#2898)
- Add CBCL face database subset to `skimage.data` (#2897)
- EHN add Haar-like feature (#2896)
- Allow mixed dtypes in compare_ssim, compare_psnr, etc. (#2893)
- noise-robust patch distance estimation for non-local means (#2890)
- Fix spelling typo in NL means docstring (#2887)
- Update URL in RAG docstring (#2885)
- DOC: Update docstring for is_low_constrast to match function signature (#2883)
- Make `test_random_shapes` use internally shipped testing tools (#2879)
- Nl means fixes for large datasets (#2878)
- Fix randomness and expected ranges for RGB in `test_random_shapes`. (#2877)
- Update DOI reference in `measure.compare_ssim` (#2872)
- Remove scipy version check in `active_contour` (#2871)
- Add multichannel support to `feature.hog` (#2870)
- Fix `skimage.measure.centroid` and add test coverage (#2869)
- OverflowError: Python int too large to convert to C long on win-amd64-py2.7 (#2867)
- Revert gradient formula, modify the deprecation warning, and fix L2-Hys norm in `skimage.feature.hog` (#2864)
- Add Computation of Image Moments to Coordinates (#2859)
- Fix link order in example (#2858)
- Update compare_nrmse docstring (#2855)
- Add 3D support to `blob_dog` and `blob_log` (#2854)
- Change default args from list to tuple in `feature.draw_multiblock_lbp` (#2852)
- [MRG] DOC fix documentation build (#2851)
- Allow convex area calculation in 3D for regionprops (#2847)
- MAINT: add Python 3.6 to appveyor, small edits (#2840)
- Fix regionprops.bbox_area bug (#2837)
- Docstring fixes for better formula formatting (#2834)
- Switch from LaTeX to MathJax in doc build (#2832)
- FIX: add estimate_sigma to __all__ in restoration module (#2829)
- Bug fix in projective tranformation composition with inverse transformation (#2826)
- Remove pytest yield (#2824)
- Adapt AppVeyor to use Python.org dist, and remove install script (#2823)
- The gallery now points to the stable docs (#2822)
- Make simple watershed fast again (#2821)
- Move INSTALL to top-level (#2819)
- Slice PIL palette correctly using extreme image value. (#2818)
- Add W503 to pep8speaks ignore. (#2816)
- Also document submodules, and ignore private modules `_*` (#2810)
- Return empty array if hough_line_peaks detects nothing (#2805)
- Fix bug in `color.convert_colorspace` for YCbCr, YPbPr (#2780)
- Add module for generating random, labeled shapes (#2773)
- as_gray replaces as_grey in imread() and load() (#2652)
- ENH: add cycle spinning routine (#2647)
- MAINT _shared.testing now contains pytest's useful functions (#2614)
- Add nD support to image moments computation (#2603)
- Update conditional requirement for PySide (#2578)
- Make HOG visualization use midpoints of orientation bins (#2525)


Contributors to this release
----------------------------

- Alvin
- Norman Barker
- Leonid Bloch
- Benedikt Boecking
- François Boulogne
- Larry Bradley
- Matthew Brett
- Alex Chum
- Yannick Copin
- Nethanel Elzas
- Kira Evans
- Christoph Gohlke
- Peter Goldsborough
- Emmanuelle Gouillart
- Ben Hadfield
- Mark Harfouche
- Scott Heatwole
- Gregory R. Lee
- Guillaume Lemaitre
- Kevin Mader
- Jarrod Millman
- Pradyumna Narayana
- Juan Nunez-Iglesias
- Egor Panfilov
- Oleksandr Pavlyk
- Alex Rothberg
- Max Schambach
- Johannes Schönberger
- Matt Swain
- Saurav R. Tuladhar
- Nelle Varoquaux
- Viraj
- David Volgyes
- Stefan van der Walt
- Thomas Walter
- Scott Warchal
- Nicholas Weir
- corrado9999
- ed1d1a8d
- eepaillard
- mikigom
- mutterer


We'd also like to thank all the people who contributed their time to perform the reviews:

- Leonid Bloch
- Jirka Borovec
- François Boulogne
- Kira Evans
- Peter Goldsborough
- Emmanuelle Gouillart
- Almar Klein
- Gregory R. Lee
- Joan Massich
- Juan Nunez-Iglesias
- Daniil Pakhomov
- Egor Panfilov
- Johannes Schönberger
- Steven Silvester
- Stefan van der Walt
- Josh Warner
- Eric Wieser

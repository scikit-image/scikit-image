Announcement: scikit-image 0.14.0
=================================

We're happy to announce the release of scikit-image v0.14.0!

scikit-image is an image processing toolbox for SciPy that includes algorithms
for segmentation, geometric transformations, color space manipulation,
analysis, filtering, morphology, feature detection, and more.

This is the last major release with official support for Python 2.7. Future
releases will be developed using Python 3-only syntax.

However, 0.14 is a long-time support (LTS) release and will receive bug fixes
and backported features deemed important (by community demand) for two years
(till the end of maintenance of Python 2.7; see PEP 373 for the details).

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


Full list of changes
--------------------
This release is the result of 8 months of work.
It contains the following 105 merged pull requests by 42 committers:

- Make HOG visualization use midpoints of orientation bins (#2525)
- Update conditional requirement for PySide (#2578)
- Add nD support to image moments computation (#2603)
- MAINT _shared.testing now contains pytest's useful functions (#2614)
- ENH: add cycle spinning routine (#2647)
- as_gray replaces as_grey in imread() and load() (#2652)
- Add module for generating random, labeled shapes (#2773)
- Fix bug in `color.convert_colorspace` for YCbCr, YPbPr (#2780)
- Return empty array if hough_line_peaks detects nothing (#2805)
- Also document submodules, and ignore private modules `_*` (#2810)
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

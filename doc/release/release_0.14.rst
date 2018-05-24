Announcement: scikit-image 0.14.0
=================================

We're happy to announce the release of scikit-image v0.14.0!

This is the last major release with an official support of Python 2, future
releases will be developed broadly using Python 3 capabilities, and will be
tested against Python 3 (or greater) environments only.

This release is also a long-time support (LTS) release, which means that it will
receive important backports and bugfixes for several years.

scikit-image is an image processing toolbox for SciPy that includes algorithms
for segmentation, geometric transformations, color space manipulation,
analysis, filtering, morphology, feature detection, and more.

For more information, examples, and documentation, please visit our website:

http://scikit-image.org


New Features
------------
- Manual segmentation with matplotlib:
  ``skimage.future.manual_polygon_segmentation``,
  ``skimage.future.manual_lasso_segmentation``. (#2584)
- Hysteresis thresholding:
  ``skimage.filters.apply_hysteresis_threshold``. (#2665)
- Lookfor function: ``skimage.lookfor``. (#2713)
- Reworked montage function: ``skimage.util.montage``. (#2626)
- 2D and 3D segmentation with morphological snakes:
  ``skimage.segmentation.morphological_chan_vese``,
  ``skimage.segmentation.morphological_geodesic_active_contour``. (#2791)
- nD support for image moments:
  ``skimage.measure.moments_central``, ``skimage.measure.moments_central``,
  ``skimage.measure.moments_normalized``, ``skimage.measure.moments_hu``. (#2603)
- Inertia tensor and its eigenvalues can now be computed outside of
  regionprops; available in ``skimage.measure.inertia_tensor``. (#2603)
- Cycle-spinning function for approximating shift-invariance by averaging
  results from a series of spatial shifts:
  ``skimage.restoration.cycle_spin``. (#2647)
- Data generation with random_shapes function:
  ``skimage.draw.random_shapes``. (#2773)
- Haar-like feature: ``skimage.feature.haar_like_feature``,
  ``skimage.feature.haar_like_feature_coord``,
  ``skimage.feature.draw_haar_like_feature``. (#2848)
- Subset of LFW database: ``skimage.data.cbcl_face_database``. (#2905)
- Image moments from coordinate input: ``skimage.measure.moments_coords``,
  ``skimage.measure.moments_coords_central``. (#2859)


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
- ``skimage.util.montage.montage2d`` is now available as
  ``skimage.util.montage2d``.
- ``skimage.morphology.binary_erosion`` now uses ``True`` as border
  value, and is now consistent with ``skimage.morphology.erosion``.


Deprecations
------------
- ``skimage.util.montage2d`` is deprecated and will be removed in 0.15.
  Use ``skimage.util.montage`` instead.
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
It contains the following 60 merges by 42 contributors:

- Make simple watershed fast again (#2821)
- Slice PIL palette correctly using extreme image value. (#2818)
- Also document submodules, and ignore private modules `_*` (#2810)
- Adapt AppVeyor to use Python.org dist, and remove install script (#2823)
- Fix bug in `color.convert_colorspace` for YCbCr, YPbPr (#2780)
- Bug fix in projective tranformation composition with inverse transformation (#2826)
- The gallery now points to the stable docs (#2822)
- Docstring fixes for better formula formatting (#2834)
- Fix regionprops.bbox_area bug (#2837)
- Remove yield tests which are deprecated in pytest (#2824)
- [MRG] DOC fix documentation build (#2851)
- Change default args from list to tuple in `feature.draw_multiblock_lbp` (#2852)
- Return empty array if hough_line_peaks detects nothing (#2805)
- Make HOG visualization use midpoints of orientation bins (#2525)
- Add module for generating random, labeled shapes (#2773)
- Fix `skimage.measure.centroid` and add test coverage. (#2869)
- ENH: add cycle spinning routine (#2647)
- Make `test_random_shapes` use internally shipped testing tools (#2879)
- Update conditional requirement for PySide (#2578)
- Fix spelling typo in NL means docstring (#2887)
- Update URL in RAG docstring (#2885)
- Add multichannel support to `feature.hog` (#2870)
- Install documentation dependencies on all builds (#2900)
- Add CBCL face database subset to `skimage.data` (#2897)
- Improve LineModelND docstrings (#2903)
- `skimage.restoration.unsupervised_wiener` Update default  doc to match code (#2906)
- remove unused parameter 'n_segments' in `_enforce_label_connectivity_cython()` (#2908)
- Nl means fixes for large datasets (#2878)
- add missing cdef in _integral_image_3d (non-local means) (#2923)
- Replace `morphology.remove_small_holes` argument `min_size` with `area_threshold` (#2924)
- do not assume 3 channels during non-local means denoising (#2922)
- Update DOI reference in `measure.compare_ssim` (#2872)
- Fix link to Windows binaries in README. (#2934)
- Corrected bug related to border value of morphology.binary_erosion (#2945)
- Add seed parameter to probabilistic_hough_line (#2960)
- Allow different rescale factors in multichannel warp (#2959)
- Correct bug in random walker when seed pixels are isolated inside pruned zones (#2946)
- Minor style fixes for #2946 (#2961)
- Build on fewer AppVeyor platforms to avoid timeout (#2962)
- Raise warning when attempting to save boolean image (#2957)
- Specify gradient parameter docstring in compare_ssim (#2937)
- Use correct NumPy version comparison in pytest configuration (#2975)
- Fix Cython compilation warnings in NL Means and Watershed (#2947)
- Ensure stdev is always nonnegative in _mean_std (#3008)
- Add citation information to README (#3013)
- Bugfix: LineModelND.residuals does not use the optional parameter `params` (#2979)
- Add Computation of Image Moments to Coordinates (#2859)
- Type dependent inversion (#3030)
- Minor style and documentation updates for #2859 (#3023)
- Fix RuntimeError: generator raised StopIteration on Python 3.7 (#3038)
- Correct reference for the coins sample image (#3042)
- Switch to basis numpy int dtypes in dtype_range (#3050)
- as_gray replaces as_grey in imread() and load() (#2652)
- For sparse CG solver, provide atol=0 keyword for SciPy >= 1.1 (#3063)
- Example for haar-like features (#2898)
- Force Appveyor to fail on failed tests (#3093)
- speedup img_as_float by making division multiplication and avoiding unecessary allocation (#3056)
- Add `threshold_local` to `filters` module namespace (#3096)
- Optimize _probabilistic_hough_line function (#3100)
- Replace grey by gray where no deprecation is needed (#3098)


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
- Thomas Walter
- Saurav R. Tuladhar
- Nelle Varoquaux
- Viraj
- David Volgyes
- Stefan van der Walt
- Scott Warchal
- Nicholas Weir
- corrado9999
- ed1d1a8d
- eepaillard
- mikigom
- mutterer


We'd like also to thank all the people who contributed their time to perform the reviews:

- Leonid Bloch
- Jirka Borovec
- François Boulogne
- Kira Evans
- Christoph Gohlke
- Peter Goldsborough
- Emmanuelle Gouillart
- Mark Harfouche
- Almar Klein
- Gregory R. Lee
- Guillaume Lemaitre
- Kevin Mader
- Joan Massich
- Viraj Navkal
- Juan Nunez-Iglesias
- Daniil Pakhomov
- Egor Panfilov
- Oleksandr Pavlyk
- Alex Rothberg
- Johannes Schönberger
- Steven Silvester
- Saurav R. Tuladhar
- Nelle Varoquaux
- Stefan van der Walt
- Thomas Walter
- Josh Warner
- Eric Wieser
- eepaillard
- nelzas

Announcement: scikit-image 0.16.2
=================================

We're happy to announce the release of scikit-image v0.16.2!

scikit-image is an image processing toolbox for SciPy that includes algorithms
for segmentation, geometric transformations, color space manipulation,
analysis, filtering, morphology, feature detection, and more.

This is a bug fix release that addresses several critical issues from 0.16.1.

Bug fixes
---------
- Migrate to networkx 2.x (#4236, #4237)
- Sync required numpy and dask to runtime versions (#4233, #4239)
- Fix wrong argument parsing in structural_similarity (#4246, #4247)
- Fix active contour gallery example after change to rc coordinates (#4257, #4262)

4 authors added to this release [alphabetical by first name or login]
---------------------------------------------------------------------
- François Boulogne
- Jarrod Millman
- Mark Harfouche
- Ondrej Pesek

6 reviewers added to this release [alphabetical by first name or login]
-----------------------------------------------------------------------
- Alexandre de Siqueira
- Egor Panfilov
- François Boulogne
- Juan Nunez-Iglesias
- Mark Harfouche
- Nelle Varoquaux


Announcement: scikit-image 0.16.1
=================================

We're happy to announce the release of scikit-image v0.16.1!

scikit-image is an image processing toolbox for SciPy that includes algorithms
for segmentation, geometric transformations, color space manipulation,
analysis, filtering, morphology, feature detection, and more.

For more information, examples, and documentation, please visit our website:

https://scikit-image.org

Starting from this release, scikit-image will follow the recently
introduced NumPy deprecation policy, `NEP 29
<https://github.com/numpy/numpy/blob/master/doc/neps/nep-0029-deprecation_policy.rst>__`.
Accordingly, scikit-image 0.16 drops support for Python 3.5.
This release of scikit-image officially supports Python 3.6 and 3.7.

Special thanks to Matthias Bussonnier for `Frappuccino
<https://github.com/Carreau/frappuccino>`__, which helped us catch all API
changes and nail down the APIs for new features.

New Features
------------
- New `skimage.evaluate` module containing simple metrics (mse,
  nrme, psd) and segmentation metrics (adapted rand error, variation of
  information) (#4025)
- n-dimensional TV-L1 optical flow algorithm for registration --
  `skimage.registration.optical_flow_tvl1` (#3983)
- Draw a line in an n-dimensional array -- `skimage.draw.line_nd`
  (#2043)
- 2D Farid & Simoncelli edge filters - `skimage.filters.farid`,
  `skimage.filters.farid_h`, and `skimage.filters.farid_v` (#3775)
- 2D majority voting filter assigning to each pixel the most commonly
  occurring value within its neighborhood -- `skimage.filters.majority`
  (#3836, #3839)
- Multi-level threshold "multi-Otsu" method, a thresholding algorithm
  used to separate the pixels of an input image into several classes by
  maximizing the variances between classes --
  `skimage.filters.threshold_multiotsu` (#3872, #4174)
- New example data -- `skimage.data.shepp_logan_phantom`, `skimage.data.colorwheel`,
  `skimage.data.brick`, `skimage.data.grass`, `skimage.data.roughwall`, `skimage.data.cell`
  (#3958, #3966)
- Compute and format image region properties as a table --
  `skimage.measure.regionprops_table` (#3959)
- Convert a polygon into a mask -- `skimage.draw.poly2mask`  (#3971, #3977)
- Visual image comparison helper `skimage.util.compare_images`,
  that returns an image showing the difference between two input images (#4089)
- `skimage.transform.warp_polar` to remap image into
  polar or log-polar coordinates. (#4097)

Improvements
------------

- RANSAC: new option to set initial samples selected for initialization (#2992)
- Better repr and str for `skimage.transform.ProjectiveTransform` (#3525,
  #3967)
- Better error messages and data type stability to
  `skimage.segmentation.relabel_sequential` (#3740)
- Improved compatibility with dask arrays in some image thresholding methods (#3823)
- `skimage.io.ImageCollection` can now receive lists of patterns (#3928)
- Speed up `skimage.feature.peak_local_max` (#3984)
- Better error message when incorrect value for keyword argument `kind` in
  `skimage.color.label2rgb` (#4055)
- All functions from `skimage.drawing` now supports multi-channel 2D images (#4134)

API Changes
-----------
- Deprecated subpackage ``skimage.novice`` has been removed.
- Default value of ``multichannel`` parameters has been set to False in
  `skimage.transform.rescale`, `skimage.transform.pyramid_reduce`,
  `skimage.transform.pyramid_laplacian`,
  `skimage.transform.pyramid_gaussian`, and
  `skimage.transform.pyramid_expand`. Guessing is no longer performed for 3D
  arrays.
- Deprecated argument ``visualise`` has been removed from
  `skimage.feature.hog`. Use ``visualize`` instead.¨
- `skimage.transform.seam_carve` has been completely removed from the
  library due to licensing restrictions.
- Parameter ``as_grey`` has been removed from `skimage.data.load` and
  `skimage.io.imread`. Use ``as_gray`` instead.
- Parameter ``min_size`` has been removed from
  `skimage.morphology.remove_small_holes`. Use ``area_threshold`` instead.
- Deprecated ``correct_mesh_orientation`` in `skimage.measure` has been
  removed.
- `skimage.measure._regionprops` has been completely switched to using
  row-column coordinates. Old x-y interface is not longer available.
- Default value of ``behavior`` parameter has been set to ``ndimage`` in
  `skimage.filters.median`.
- Parameter ``flatten`` in `skimage.io.imread` has been removed in
  favor of ``as_gray``.
- Parameters ``Hxx, Hxy, Hyy`` have been removed from
  `skimage.feature.corner.hessian_matrix_eigvals` in favor of ``H_elems``.
- Default value of ``order`` parameter has been set to ``rc`` in
  `skimage.feature.hessian_matrix`.
- ``skimage.util.img_as_*`` functions no longer raise precision and/or loss warnings.

Bugfixes
--------

- Corrected error with scales attribute in ORB.detect_and_extract (#2835)
  The scales attribute wasn't taking into account the mask, and thus was using
  an incorrect array size.
- Correct for bias in Inverse Randon Transform (`skimage.transform.irandon`) (#3067)
  Fixed by using the Ramp filter equation in the spatial domain as described
  in the reference
- Fix a rounding issue that caused  a rotated image to have a
  different size than the input (`skimage.transform.rotate`)  (#3173)
- RANSAC uses random subsets of the original data and not bootstraps. (#3901,
  #3915)
- Canny now produces the same output regardless of dtype (#3919)
- Geometry Transforms: avoid division by zero & some degenerate cases (#3926)
- Fixed float32 support in denoise_bilateral and denoise_tv_bregman (#3936)
- Fixed computation of Meijering filter and avoid ZeroDivisionError (#3957)
- Fixed `skimage.filters.threshold_li` to prevent being stuck on stationnary
  points, and thus at local minima or maxima (#3966)
- Edited `skimage.exposure.rescale_intensity` to return input image instead of
  nans when all 0 (#4015)
- Fixed `skimage.morphology.medial_axis`. A wrong indentation in Cython
  caused the function to not behave as intended. (#4060)
- Fixed `skimage.restoration.denoise_bilateral` by correcting the padding in
  the gaussian filter(#4080)
- Fixed `skimage.measure.find_contours` when input image contains NaN.
  Contours interesting NaN will be left open (#4150)
- Fixed `skimage.feature.blob_log` and `skimage.feature.blob_dog` for 3D
  images and anisotropic data (#4162)
- Fixed `skimage.exposure.adjust_gamma`, `skimage.exposure.adjust_log`,
  and `skimage.exposure.adjust_sigmoid` such that when provided with a 1 by
  1 ndarray, it returns 1 by 1 ndarrays and not single number floats (#4169)

Deprecations
------------
- Parameter ``neighbors`` in `skimage.measure.convex_hull_object` has been
  deprecated in favor of ``connectivity`` and will be removed in version 0.18.0.
- The following functions are deprecated in favor of the `skimage.metrics`
  module (#4025):

    - `skimage.measure.compare_mse`
    - `skimage.measure.compare_nrmse`
    - `skimage.measure.compare_psnr`
    - `skimage.measure.compare_ssim`

- The function `skimage.color.guess_spatial_dimensions` is deprecated and
  will be removed in 0.18 (#4031)
- The argument ``bc`` in `skimage.segmentation.active_contour` is
  deprecated.
- The function `skimage.data.load` is deprecated and will be removed in 0.18
  (#4061)
- The function `skimage.transform.match_histogram` is deprecated in favor of
  `skimage.exposure.match_histogram` (#4107)
- The parameter ``neighbors`` of `skimage.morphology.convex_hull_object` is
  deprecated. 
- The `skimage.transform.randon_tranform` function will convert input image
  of integer type to float by default in 0.18. To preserve current behaviour,
  set the new argument ``preserve_range`` to True. (#4131)


Documentation improvements
--------------------------

- DOC: Improve the documentation of transform.resize with respect to the anti_aliasing_sigma parameter (#3911)
- Fix URL for stain deconvolution reference (#3862)
- Fix doc for denoise guassian (#3869)
- DOC: various enhancements (cross links, gallery, ref...), mainly for corner detection (#3996)
- [DOC] clarify that the inertia_tensor may be nD in documentation (#4013)
- [DOC] How to test and write benchmarks (#4016)
- Spellcheck @CONTRIBUTING.txt (#4008)
- Spellcheck @doc/examples/segmentation/plot_watershed.py (#4009)
- Spellcheck @doc/examples/segmentation/plot_thresholding.py (#4010)
- Spellcheck @skimage/morphology/binary.py (#4011)
- Spellcheck @skimage/morphology/extrema.py (#4012)
- docs update for downscale_local_mean and N-dimensional images (#4079)
- Remove fancy language from 0.15 release notes (#3827)
- Documentation formatting / compilation fixes (#3838)
- Remove duplicated section in INSTALL.txt. (#3876)
- ENH: doc of ridge functions (#3933)
- Fix docstring for Threshold Niblack (#3917)
- adding docs to circle_perimeter_aa (#4155)
- Update link to NumPy docstring standard in Contribution Guide (replaces #4191) (#4192)
- DOC: Improve downscale_local_mean() docstring (#4180)
- DOC: enhance the result display in ransac gallery example (#4109)
- Gallery: use fstrings for better readability (#4110)
- MNT: Document stacklevel parameter in contribution guide (#4066)
- Fix minor typo (#3988)
- MIN: docstring improvements in canny functions (#3920)
- Minor docstring fixes for #4150 (#4184)
- Fix `full` parameter description in compare_ssim (#3860)
- State Bradley threshold equivalence in Niblack docstring (#3891)
- Add plt.show() to example-code for consistency. (#3908)
- CC0 is not equivalent to public domain. Fix the note of the horse image (#3931)
- Update the joblib link in tutorial_parallelization.rst (#3943)
- Fix plot_edge_filter.py references (#3946)
- Add missing argument to docstring of PaintTool (#3970)
- Improving documentation and tests for directional filters (#3956)
- Added new thorough examples on the inner working of
  ``skimage.filters.threshold_li`` (#3966)
- matplotlib: remove interpolation=nearest, none in our examples (#4002)
- fix URL encoding for wikipedia references in filters.rank.entropy and filters.rank.shannon_entropy docstring (#4007)
- Fixup integer division in examples (#4032)
- Update the links the installation guide (#4118)
- Gallery hough line transform (#4124)
- Cross-linking between function documentation should now be much improved! (#4188)
- Better documentation of the ``num_peaks`` of `skimage.feature.corner_peaks` (#4195)


Other Pull Requests
-------------------
- Add benchmark suite for exposure module (#3312)
- Remove precision and sign loss warnings from ``skimage.util.img_as_`` (#3575)
- Propose SKIPs and add mission/vision/values, governance (#3585)
- Use user-installed tifffile if available (#3650)
- Simplify benchmarks pinnings (#3711)
- Add project_urls to setup for PyPI and other services (#3834)
- Address deprecations for 0.16 release (#3841)
- Followup deprecations for 0.16 (#3851)
- Build and test the docs in Azure (#3873)
- Pin numpydoc to pre-0.8 to fix dev docs formatting (#3893)
- Change all HTTP links to HTTPS (#3896)
- Skip extra deps on OSX (#3898)
- Add location for Sphinx 2.0.1 search results; clean up templates (#3899)
- Fix CSS styling of Sphinx 2.0.1 + numpydoc 0.9 rendered docs (#3900)
- Travis CI: The sudo: tag is deprcated in Travis (#4164)
- MNT Preparing the 0.16 release (#4204)
- FIX generate_release_note when contributor_set contains None (#4205)
- Specify that travis should use Ubuntu xenial (14.04) not trusty (16.04) (#4082)
- MNT: set stack level accordingly in lab2xyz (#4067)
- MNT: fixup stack level for filters ridges (#4068)
- MNT: remove unused import `deprecated` from filters.thresholding (#4069)
- MNT: Set stacklevel correctly in io matplotlib plugin (#4070)
- MNT: set stacklevel accordingly in felzenszwalb_cython (#4071)
- MNT: Set stacklevel accordingly in img_as_* (convert) (#4072)
- MNT: set stacklevel accordingly in util.shape (#4073)
- MNT: remove extreneous matplotlib warning (#4074)
- Suppress warnings in tests for viewer (#4017)
- Suppress warnings in test suite regarding measure.label (#4018)
- Suppress warnings in test_rank due to type conversion (#4019)
- Add todo item for imread plugin testing (#3907)
- Remove matplotlib agg warning when using the sphinx gallery. (#3897)
- Forward-port release notes for 0.14.4 (#4137)
- Add tests for pathological arrays in threshold_li (#4143)
- setup.py: Fail gracefully when NumPy is not installed (#4181)
- Drop Python 3.5 support (#4102)
- Force imageio reader to return NumPy arrays (#3837)
- Fixing connecting to GitHub with SSH info. (#3875)
- Small fix to an error message of `skimage.measure.regionprops` (#3884)
- Unify skeletonize and skeletonize 3D APIs (#3904)
- Add location for Sphinx 2.0.1 search results; clean up templates (#3910)
- Pin numpy version forward (#3925)
- Replacing pyfits with Astropy to read FITS (#3930)
- Add warning for future dtype kwarg removal (#3932)
- MAINT: cleanup regionprop add PYTHONOPTIMIZE=2 to travis array (#3934)
- Adding complexity and new tests for filters.threshold_multiotsu (#3935)
- Fixup dtype kwarg warning in certain image plugins (#3948)
- don't cast integer to float before using it as integer in numpy logspace (#3949)
- avoid low contrast image save in a doctest. (#3953)
- MAINT: Remove unused _convert_input from filters._gaussian (#4001)
- Set minimum version for imread so that it compiles from source on linux in test builds (#3960)
- Cleanup plugin utilization in data.load and testsuite (#3961)
- Select minimum imageio such that it is compatible with pathlib (#3969)
- Remove pytest-faulthandler from test dependencies (#3987)
- Fix tifffile and __array_function__ failures in our CI (#3992)
- MAINT: Do not use assert in code, raise an exception instead. (#4006)
- Enable packagers to disable failures on warnings. (#4021)
- Fix numpy 117 rc and dask in thresholding filters (#4022)
- silence r,c  warnings when property does not depend on r,c (#4027)
- remove warning filter, fix doc wrt r,c (#4028)
- Import Iterable from collections.abc (#4033)
- Import Iterable from collections.abc in vendored tifffile code (#4034)
- Correction of typos after #4025 (#4036)
- Rename internal function called assert_* -> check_* (#4037)
- Improve import time (#4039)
- Remove .meeseeksdev.yml (#4045)
- Fix mpl deprecation on grid() (#4049)
- Fix gallery after deprecation from #4025 (#4050)
- fix mpl future deprecation normed -> density (#4053)
- Add shape= to circle perimeter in hough_circle example (#4047)
- Critical: address internal warnings in test suite related to metrics 4025 (#4063)
- Use functools instead of a real function for the internal warn function (#4062)
- Test rank capture warnings in threadsafe manner (#4064)
- Make use of FFTs more consistent across the library (#4084)
- Fixup region props test (#4099)
- Turn single backquotes to double backquotes in filters (#4127)
- Refactor radon transform module (#4136)
- Fix broken import of rgb2gray in benchmark suite (#4176)
- Fix doc building issues with SKIPs (#4182)
- Remove several __future__ imports (#4198)
- Restore deprecated coordinates arg to regionprops (#4144)
- Refactor/optimize threshold_multiotsu (#4167)
- Remove Python2-specific code (#4170)
- `view_as_windows` incorrectly assumes that a contiguous array is needed  (#4171)
- Handle case in which NamedTemporaryFile fails (#4172)
- Fix incorrect resolution date on SKIP1 (#4183)
- API updates before 0.16 (#4187)
- Fix conversion to float32 dtype (#4193)


Contributors to this release
----------------------------

- Abhishek Arya
- Alexandre de Siqueira
- Alexis Mignon
- Anthony Carapetis
- Bastian Eichenberger
- Bharat Raghunathan
- Christian Clauss
- Clement Ng
- David Breuer
- David Haberthür
- Dominik Kutra
- Dominik Straub
- Egor Panfilov
- Emmanuelle Gouillart
- Etienne Landuré
- François Boulogne
- Genevieve Buckley
- Gregory R. Lee
- Hadrien Mary
- Hamdi Sahloul
- Holly Gibbs
- Huang-Wei Chang
- i3v (i3v)
- Jarrod Millman
- Jirka Borovec
- Johan Jeppsson
- Johannes Schönberger
- Jon Crall
- Josh Warner
- Juan Nunez-Iglesias
- Kaligule (Kaligule)
- kczimm (kczimm)
- Lars Grueter
- Shachar Ben Harim
- Luis F. de Figueiredo
- Mark Harfouche
- Mars Huang
- Dave Mellert
- Nelle Varoquaux
- Ollin Boer Bohan
- Patrick J Zager
- Riadh Fezzani
- Ryan Avery
- Srinath Kailasa
- Stefan van der Walt
- Stuart Berg
- Uwe Schmidt


Reviewers for this release
--------------------------

- Alexandre de Siqueira
- Anthony Carapetis
- Bastian Eichenberger
- Clement Ng
- David Breuer
- Egor Panfilov
- Emmanuelle Gouillart
- Etienne Landuré
- François Boulogne
- Genevieve Buckley
- Gregory R. Lee
- Hadrien Mary
- Hamdi Sahloul
- Holly Gibbs
- Jarrod Millman
- Jirka Borovec
- Johan Jeppsson
- Johannes Schönberger
- Jon Crall
- Josh Warner
- jrmarsha
- Juan Nunez-Iglesias
- kczimm
- Lars Grueter
- leGIT-bot
- Mark Harfouche
- Mars Huang
- Dave Mellert
- Paul Müller
- Phil Starkey
- Ralf Gommers
- Riadh Fezzani
- Ryan Avery
- Sebastian Berg
- Stefan van der Walt
- Uwe Schmidt


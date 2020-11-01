Announcement: scikit-image 0.17.2
=================================

We're happy to announce the release of scikit-image v0.17.2, which is a bug-fix
release.

Bug fixes
---------

- We made pooch an optional dependency, since it has been added as required
  dependency by mistake (#4666), and we fixed a bug about the path used for pooch
  to download data (#4662)
- The support of float 32 images was corrected for slic segmentation,
  ORB and BRIEF feature detectors (#4683, #4684, #4685, #4696, #4697)
- We removed deprecated arguments (#4691)
   * ``mask``, ``shift_x``, and ``shift_y`` from ``skimage.filters.median``
   * ``beta1`` and ``beta2`` from ``skimage.filters.frangi``
   * ``beta1`` and ``beta2`` from ``skimage.filters.hessian``
   * ``dtype`` from ``skimage.io.imread``
   * ``img`` from skimage.morphology.skeletonize_3d.
- Gallery examples were updated to suppress warnings and take into account new
  default values in some functions (#4692 and #4676)



6 authors added to this release [alphabetical by first name or login]
---------------------------------------------------------------------
- Alexandre de Siqueira
- Emmanuelle Gouillart
- François Boulogne
- Juan Nunez-Iglesias
- Mark Harfouche
- Riadh Fezzani



Announcement: scikit-image 0.17.1
=================================

We're happy to announce the release of scikit-image v0.17.1!


scikit-image is an image processing toolbox for SciPy that includes algorithms
for segmentation, geometric transformations, color space manipulation,
analysis, filtering, morphology, feature detection, and more.


For more information, examples, and documentation, please visit our website:

https://scikit-image.org

Many thanks to the 54 authors who contributed the amazing number of 213 merged
pull requests! scikit-image is a community-based project and we are happy that
this number includes first-time contributors to scikit-image.

Special thanks for the release to the Cython team, who helped us make our code
compatible with their coming Cython 3.0 release. 

New Features
------------

- Hyperparameter calibration of denoising algorithms with
  `restoration.calibrate_denoiser` (#3824), with corresponding
  gallery example and tutorial.
- `measure.profile_line` has a new `reduce_func` parameter to accept a
  reduction operation to be computed on pixel values along the profile (#4206)
- nD windows for reducing spectral leakage when computing the FFT of
  n-dimensional images, with `filters.window` (#4252) (with new gallery example)
- Add Minkowski distance metric support to corner_peak (#4218)
- `util.map_array` was introduced to map a set of pixel values to another one
  (for example to map region labels to the size of regions in an image of
  labels) #4612 and #4646
- Masked marching cubes (#3829)
- The SLIC superpixel algorithm now accepts a mask to exclude some parts of the
  image and force the superpixel boundaries to follow the boundary of the mask
  (#3850)
- Pooch -- on the fly download of datasets from github: we introduced the
  possibility to include larger datasets in the `data` submodule, thanks to the
  `pooch` library. `data.download_all` fetches all datasets. (#3945)
- Starting with this version, our gallery examples now have links to run the
  example notebook on a binder instance. (#4543)

New doc tutorials and gallery examples have been added to the use of regionprops_table (#4348)
geometrical transformations (#4385), and the registration of rotation and
scaling with no shared center (#4515). A new section on registration has been
added to the gallery (#4575).

Improvements
------------

- scikit-image aims at being fully compatible with 3D arrays, and when possible
  with nD arrays. nD support has been added to color conversion functions
  (#4418), to the CLAHE `exposure.equalize_adapthist` algorithm (#4598) 
  and to the Sobel, Scharr, and Prewitt filters (#4347).
- Multichannel support for denoise_tv_bregman (#4446)
- The memory footprint of `segmentation.relabel_sequential` has been reduced in
  the case of labels much larger than the number of labels (#4612)
- Random ellipses are now possible in `draw.random_shapes` (#4493)
- Add border conditions to ridge filters (#4396)
- `segmentation.random_walker` new Jacobi preconditioned conjugate gradient mode
  (#4359) and minor corrections #4630
- Warn when rescaling with NaN in exposure.intensity_range (#4265)

We have also improved the consistency of several functions regarding the way
they handle data types

- Make dtype consistent in filters.rank functions (#4289)
- Fix colorconv float32 to double cast (#4296)
- Prevent radon from upcasting float32 arrays to double (#4297)
- Manage iradon_sart input and output data type (#4300)

API Changes
-----------

- When used with floating point inputs, ``denoise_wavelet`` no longer rescales
  the range of the data or clips the output to the range [0, 1] or [-1, 1].
  For non-float inputs, rescaling and clipping still occurs as in prior
  releases (although with a bugfix related to the scaling of ``sigma``).
- For 2D input, edge filters (Sobel, Scharr, Prewitt, Roberts, and Farid)
  no longer set the boundary pixels to 0 when a mask is not supplied. This was
  changed because the boundary mode for `scipy.ndimage.convolve` is now
  ``'reflect'``, which allows meaningful values at the borders for these
  filters. To retain the old behavior, pass
  ``mask=np.ones(image.shape, dtype=bool)`` (#4347)
- When ``out_range`` is a range of numbers and not a dtype in
  :func:`skimage.exposure.rescale_intensity`, the output data type will always
  be float (#4585)
- The values returned by :func:`skimage.exposure.equalize_adapthist` will be
  slightly different from previous versions due to different rounding behavior
  (#4585)
- Move masked_register_translation from feature to registration (#4503)
- Move register_translation from skimage.feature to skimage.registration (#4502)
- Move watershed from morphology to segmentation (#4443)
- Rename draw.circle() to draw.disk() (#4428)
- The forward and backward maps returned by :func:`skimage.segmentation.relabel_sequential`
  are no longer NumPy arrays, but more memory-efficient `ArrayMap` objects that behave
  the same way for mapping. See the ``relabel_sequential`` documentation for more details.
  To get NumPy arrays back, cast it as a NumPy array: ``np.asarray(forward_map)`` (#4612)


Bugfixes
--------

- ``denoise_wavelet``: For user-supplied `sigma`, if the input image gets
  rescaled via ``img_as_float``, the same scaling will be applied to `sigma` to
  preserve the relative scale of the noise estimate. To restore the old,
  behaviour, the user can manually specify ``rescale_sigma=False``.
- Fix Frangi artefacts around the image (#4343)
- Fix Negative eigenvalue in inertia_tensor_eigvals due to floating point precision (#4589)
- Fix morphology.flood for F-ordered images (#4556)
- Fix h_maxima/minima strange behaviors on floating point image input (#4496)
- Fix peak_local_max coordinates ordering (#4501)
- Sort naturally peaks coordinates of same amplitude in peak_local_max (#4582)
- Fix denoise_nl_means data type management (#4322)
- Update rescale_intensity to prevent under/overflow and produce proper output dtype (#4585)

(other small bug fixes are part of the list of other pull requests at the end)

Deprecations
------------
The minimal supported Python version by this release is 3.6.

- Parameter ``inplace`` in skimage.morphology.flood_fill has been deprecated
  in favor of ``in_place`` and will be removed in version scikit-image 0.19.0
  (#4250).
- ``skimage.segmentation.circle_level_set`` has been deprecated and will be
  removed in 0.19. Use ``skimage.segmentation.disk_level_set`` instead.
- ``skimage.draw.circle`` has been deprecated and will be removed in 0.19.
  Use ``skimage.draw.disk`` instead.
- Deprecate filter argument in iradon due to clash with python keyword (#4158)
- Deprecate marching_cubes_classic (#4287)
- Change label2rgb default background value from -1 to 0 (#4614)
- Deprecate rgb2grey and grey2rgb (#4420)
- Complete deprecation of circle in morphsnakes (#4467)
- Deprecate non RGB image conversion in rgb2gray (#4838, #4439), and deprecate
  non gray scale image conversion in gray2rgb (#4440)

The list of other pull requests is given at the end of this document, after the
list of authors and reviewers.

54 authors added to this release [alphabetical by first name or login]
----------------------------------------------------------------------

- aadideshpande (aadideshpande)
- Alexandre de Siqueira
- Asaf Kali
- Cedric
- D-Bhatta (D-Bhatta)
- Danielle
- Davis Bennett
- Dhiren Serai
- Dylan Cutler
- Egor Panfilov
- Emmanuelle Gouillart
- Eoghan O'Connell
- Eric Jelli
- Eric Perlman
- erjel (erjel)
- Evan Widloski
- François Boulogne
- Gregory R. Lee
- Hazen Babcock
- Jan Eglinger
- Joshua Batson
- Juan Nunez-Iglesias
- Justin Terry
- kalvdans (kalvdans)
- Karthikeyan Singaravelan
- Lars Grüter
- Leengit (Leengit)
- leGIT-bot (leGIT-bot)
- LGiki
- Marianne Corvellec
- Mark Harfouche
- Marvin Albert
- mellertd (Dave Mellert)
- Miguel de la Varga
- Mostafa Alaa
- Mojdeh Rastgoo (mrastgoo)
- notmatthancock (matt)
- Ole Streicher
- Riadh Fezzani
- robroooh (robroooh)
- SamirNasibli
- schneefux (schneefux)
- Scott Sievert
- Stefan van der Walt
- Talley Lambert
- Tim Head (betatim)
- Thomas A Caswell
- Timothy Sweetser
- Tony Tung
- Uwe Schmidt
- VolkerH (VolkerH)
- Xiaoyu Wu
- Yuanqin Lu
- Zaccharie Ramzi
- Zhōu Bówēi 周伯威


35 reviewers added to this release [alphabetical by first name or login]
------------------------------------------------------------------------
- Alexandre de Siqueira
- Asaf Kali
- D-Bhatta
- Egor Panfilov
- Emmanuelle Gouillart
- Eoghan O'Connell
- erjel
- François Boulogne
- Gregory R. Lee
- Hazen Babcock
- Jacob Quinn Shenker
- Jirka Borovec
- Josh Warner
- Joshua Batson
- Juan Nunez-Iglesias
- Justin Terry
- Lars Grüter
- Leengit
- leGIT-bot
- Marianne Corvellec
- Mark Harfouche
- Marvin Albert
- mellertd
- Miguel de la Varga
- Riadh Fezzani
- robroooh
- SamirNasibli
- Stefan van der Walt
- Timothy Sweetser
- Tony Tung
- Uwe Schmidt
- VolkerH
- Xiaoyu Wu
- Zhōu Bówēi 周伯威


Other Pull Requests
*******************
- [WIP] DOC changing the doc in plot_glcm (#2789)
- Document tophat in the gallery (#3609)
- More informative error message on boolean images for regionprops  (#4156)
- Refactor/fix threshold_multiotsu (#4178)
- Sort the generated API documentation alphabetically (#4208)
- Fix the random Linux build fails in travis CI (#4227)
- Initialize starting vector for `scipy.sparse.linalg.eigsh` to ensure reproducibility in graph_cut (#4251)
- Add histogram matching test (#4254)
- MAINT: use SciPy's implementation of convolution method (#4267)
- Improve CSS for SKIP rendering (#4271)
- Add toggle for prompts in docstring examples next to copybutton (#4273)
- Tight layout for glcm example in gallery (#4285)
- Forward port 0.16.2 release notes (#4290)
- Fix typo in `hog` docstring (#4302)
- pyramid functions take preserve_range kwarg (#4310)
- Create test and fix types (#4311)
- Deprecate numpy.pad wrapping (#4313)
- Clarify merge policy in core contributor guide (#4315)
- Regionprops is empty bug (#4316)
- Add check to avoid import craching (#4319)
- Fix typo in `simple_metrics` docstring (#4323)
- Make peak_local_max exclude_border independent and anisotropic (#4325)
- Fix blob_log/blob_dog and their corresponding tests (#4327)
- Add section on closing issues to core dev guide (#4328)
- Use gaussian filter output array if provided (#4329)
- Move cython pinning forward (#4330)
- Add python 3.8 to the build matrix (#4331)
- Avoid importing mathematical functions from scipy as told ;) (#4332)
- Add dtype keyword argument to block reduce and small documentation changes (#4334)
- Add explicit use of 32-bit int in fast_exp (#4338)
- Fix single precision cast to double in slic (#4339)
- Change `measure.block_reduce` to accept explicit `func_kwargs` kwd (#4341)
- Fix equalize_adapthist border artifacts (#4349)
- Make hough_circle_peaks respect min_xdistance, min_ydistance (#4350)
- Deprecate CONTRIBUTORS.txt and replace by git shortlog command (#4351)
- Add warning on pillow version if reading a MPO image (#4354)
- Minor documentation improvement in `measure.block_reduce` (#4355)
- Add example to highlight regionprops_table (#4356)
- Remove code that tries to avoid upgrading large dependencies from setup.py (#4362)
- Fix float32 promotion in cubic interpolation (#4363)
- Update to the new way of generating Sphinx search box (#4367)
- clarify register_translation example description (#4368)
- Bump scipy minimum version to 1.0.1 (#4372)
- Fixup OSX Builds by skipping building with numpy 1.18.0 (#4376)
- Bump pywavelets to 0.5.2 (#4377)
- mini-galleries for classes as well in API doc (#4381)
- gallery: Fix typo + reduce the angle to a reasonable value (#4386)
- setup: read long description from README (#4392)
- Do not depend on test execution order for success (#4393)
- _adapthist module refactoring and memory use reduction (#4395)
- Documentation fixes for transform (rescale, warp_polar) (#4401)
- DOC: specify the meaning of m in ransac formula (#4404)
- Updating link to values in core developer guide (#4405)
- Fix subtract_mean underflow correction (#4409)
- Fix hanging documentation build in Azure (#4411)
- Fix warnings regarding invalid escape sequences. (#4414)
- Fix the URLs in skimage.transform.pyramids (#4415)
- Fix profile_line interpolation errors (#4416)
- MAINT: replace circle_level_set by disk_level_set (#4421)
- Add stacklevel=2 to deprecation warnings in skimage.measure.marching_cubes (#4422)
- Deprecate rank.tophat and rank.bottomhat (#4423)
- Add gray2rgba and deprecate RGBA support in gray2rgb (#4424)
- ISSUE_TEMPLATE: add note about image.sc forum (#4429)
- Fix the link in skips.1-governance (#4432)
- Fix the dead link in skimage.feature.canny (#4433)
- Fix use_quantiles behavior in canny (#4437)
- Remove redundant checks for threshold values in Canny (#4441)
- Difference of Gaussians function (#4445)
- Fix test for denoise_tv_bregman accepting float32 and float64 as inputs (#4448)
- Standardize colon usage in docstrings (#4449)
- Bump numpy version to 1.15.1 (#4452)
- Set minimum tifffile version to fix numpy incompatibility (#4453)
- Cleanup warnings regarding denoise_wavelet (#4456)
- Address FutureWarning from numpy in subdtype check in reginoprops (#4457)
- Skip warnings in doctests for warning module (#4458)
- Skip doctests for deprecated functions rank.tophat rank.bottomhat since they emit warnings (#4459)
- Skip morphology.watershed doctest since it was moved and emits a warning (#4460)
- Use rgba2rgb directly where rgb kind is inferred (#4461)
- Cleanup corner peaks warnings (#4463)
- Fix edgecase bugs in segmentation.relabel_sequential (#4465)
- Fix deltaE cmc close colors bug (#4469)
- Fix bool array warping (#4470)
- Fix bool array profile_line (#4471)
- Fix values link in governance (#4472)
- Improving example on filters (#4479)
- reduce runtime of non-local means tests (#4480)
- Add sponsor button (#4481)
- reduced the duration of the longest tests (#4487)
- tiny improvements to haar feature examples (#4490)
- Add min version to sphinx-gallery >= 0.3.1 to work with py3.8 (#4498)
- Fix KeyError in find_contours (#4505)
- Fix bool array save with imageio plugin (#4512)
- Fixing order of elements in docstrings of skimage/color/colorconv (#4518)
- Fix exposure_adapthist return when clip_limit == 1 (#4519)
- Adding info on venv activation on Windows (#4521)
- Fix similarity transform scale (#4524)
- Added explanation in the example of `segmentation/plot_label.py` to make the background transparent (#4527)
- Add example code for generating structuring elements. (#4528)
- Block imread version 0.7.2 due to build failure (#4529)
- Maint: edits to suppress some warnings (unused imports, blank lines) (#4530)
- MNT: remove duplicate nogil specification (#4546)
- Block pillow 7.1.0, see #4548 (#4551)
- Fix binder requirements (#4555)
- Do not enforce pil plugin in skimage.data (#4560)
- Remove "backport to 0.14" in github template (#4561)
- Fix inconsistency in docstring (filters.median) (#4562)
- Disable key check for texlive in travis-mac as a temporary workaround (#4565)
- Bump Pywavelets min requirement to 1.1.1 (#4568)
- Strip backslash in sphinx 3.0.0 (#4569)
- Remove binary specification from match_descriptors docstring (#4571)
- Remove importing skimage.transform as tf (#4576)
- Add note to remove option in doc config when numpydoc will be patched (#4578)
- update task in TODO.txt (#4579)
- Rename convert to _convert, as it is a private function (#4590)
- Do not overwrite data module in plot_skeleton.py (#4591)
- [CI fix] add import_array in cython files where numpy is cimport-ed (#4592)
- Recommend cnp.import_array in contribution guide (#4593)
- Add example of natsort usage in documentation (#4599)
- Fix broken and permanently moved links (#4600)
- Fix typo in cython import_array (#4602)
- Update min required sphinx version for sphinx-copybutton (#4604)
- Clarify error message when montaging multichannel nD images and multichannel=False (#4607)
- Fix register_translation warning message (#4609)
- Add notes on deprecation warnings in marching_cube_* and gray2rgb (#4610)
- Improve loading speed of our gallery by reducing the thumbnail size (#4613)
- Fixed wrong behaviour of `exposure.rescale_intensity` for constant input. (#4615)
- Change math formatting in the docstrings (#4617)
- Add .mypy_cache to the gitignore (#4620)
- typo fixes for register rotation gallery example (#4623)
- Userguide: add a visualization chapter (#4627)
- Fix deprecation warnings due to invalid escape sequences.  (#4628)
- add docstring examples for moments_hu and centroid (#4632)
- Update pooch registry with new file location (#4635)
- Misleading "ValueError: Input array has to be either 3- or 4-dimensional" in montage (#4638)
- Fix broken link (#4639)
- AffineTransform: Allow a single value for 'scale' to apply to both sx & sy (#4642)
- Fix CI - cython 3.0a4 (#4643)
- Fix sphinx (#4644)
- Fix ArrayMap test (#4645)
- Remove copy of tifffile; install from pip (#4235)
- Refactor/move neighborhood utility functions in morphology (#4209)


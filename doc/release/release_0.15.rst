Announcement: scikit-image 0.15.0
=================================

We're happy to announce the release of scikit-image v0.15.0!

scikit-image is an image processing toolbox for SciPy that includes algorithms
for segmentation, geometric transformations, color space manipulation,
analysis, filtering, morphology, feature detection, and more.

For more information, examples, and documentation, please visit our website:

https://scikit-image.org

0.15 is the first scikit-image release that is only compatible with Python 3.5
and above. Python 2.7 users should strongly consider upgrading to Python 3.5+,
or use the 0.14 long term support releases.


New Features
------------

- N-dimensional flood fill, with tolerance (#3245)
- Attribute operators (#2680)
- Extension of register_translation to enable subpixel precision in 3D and
  optionally disable error calculation (#2880)
- unsharp mask filtering (#2772)
- New options ``connectivity``, ``indices`` and ``allow_borders`` for
  ``skimage.morphology.local_maxima`` and ``local_minima``. (#3022)
- Image translation registration for masked data
  (``skimage.feature.masked_register_translation``) (#3334)
- Frangi (vesselness), Meijering (neuriteness), and Sato (tubeness) filters
  (#3515)
- Allow float->float conversion of any range (#3052)
- Let lower precision float arrays pass through ``img_as_float`` (#3110)
- Lazy apply_parallel (allows optimization of dask array operations) (#3121)
- Add range option for histogram. (#2479)
- Add histogram matching (#3568)


Improvements
------------

- Replace ``morphology.local_maxima`` with faster flood-fill based Cython
  version (#3022)
- ``skivi`` is now using ``qtpy`` for Qt4/Qt5/PySide/PySide2 compatibility (a
  new optional dependency).
- Performance is now monitored by
  `Airspeed Velocity <https://asv.readthedocs.io/en/stable/>`_. Benchmark
  results will appear at https://pandas.pydata.org/speed/ (#3137)
- Speed up inner loop of GLCM (#3378)
- Allow tuple to define kernel in threshold_niblack and threshold_sauvola (#3596)
- Add support for anisotropic blob detection in blob_log and blob_dog (#3690)


API Changes
-----------

- ``skimage.transform.seam_carve`` has been removed because the algorithm is
  patented. (#3751)
- Parameter ``dynamic_range`` in ``skimage.measure.compare_psnr`` has been
  removed. Use parameter ``data_range`` instead. (#3313)
- imageio is now the preferred plugin for reading and writing images. (#3126)
- imageio is now a dependency of scikit-image. (#3126)
- ``regular_grid`` now returns a tuple instead of a list for compatibility
  with numpy 1.15 (#3238)
- ``colorconv.separate_stains`` and ``colorconv.combine_stains`` now uses
  base10 instead of the natural logarithm as discussed in issue #2995. (#3146)
- Default value of ``clip_negative`` parameter in ``skimage.util.dtype_limits``
  has been set to ``False``.
- Default value of ``circle`` parameter in ``skimage.transform.radon``
  has been set to ``True``.
- Default value of ``circle`` parameter in ``skimage.transform.iradon``
  has been set to ``True``.
- Default value of ``mode`` parameter in ``skimage.transform.swirl``
  has been set to ``reflect``.
- Deprecated ``skimage.filters.threshold_adaptive`` has been removed.
  Use ``skimage.filters.threshold_local`` instead.
- Default value of ``multichannel`` parameter in
  ``skimage.restoration.denoise_bilateral`` has been set to ``False``.
- Default value of ``multichannel`` parameter in
  ``skimage.restoration.denoise_nl_means`` has been set to ``False``.
- Default value of ``mode`` parameter in ``skimage.transform.resize``
  and ``skimage.transform.rescale`` has been set to ``reflect``.
- Default value of ``anti_aliasing`` parameter in ``skimage.transform.resize``
  and ``skimage.transform.rescale`` has been set to ``True``.
- Removed the ``skimage.test`` function. This functionality can be achieved
  by calling ``pytest`` directly.
- ``morphology.local_maxima`` now returns a boolean array (#3749)


Bugfixes
--------

- Correct bright ridge detection for Frangi filter (#2700)
- ``skimage.morphology.local_maxima`` and ``skimage.morphology.local_minima``
  no longer raise an error if any dimension of the image is smaller 3 and
  the keyword ``allow_borders`` was false.
- ``skimage.morphology.local_maxima`` and ``skimage.morphology.local_minima``
  will return a boolean array instead of an array of 0s and 1s if the
  parameter ``indices`` was false.
- When ``compare_ssim`` is used with ``gaussian_weights`` set to True, the
  boundary crop used when computing the mean structural similarity will now
  exactly match the width of the Gaussian used. The Gaussian filter window is
  also now truncated at 3.5 rather than 4.0 standard deviations to exactly match
  the original publication on the SSIM. These changes should produce only a very
  small change in the computed SSIM value. There is no change to the existing
  behavior when ``gaussian_weights`` is False. (#3802)
- erroneous use of cython wrap around (#3481)
- Speed up block reduce by providing the appropriate parameters to numpy (#3522)
- Add urllib.request again (#3766)
- Repeat pixels in reflect mode when image has dimension 1 (#3174)
- Improve Li thresholding (#3402, 3622)


Deprecations
------------

- Python 2 support has been dropped. Users should have Python >= 3.5. (#3000)
- ``skimage.util.montage2d`` has been removed. Use ``skimage.util.montage`` instead.
- ``skimage.novice`` is deprecated and will be removed in 0.16.
- ``skimage.transform.resize`` and ``skimage.transform.rescale`` option
  ``anti_aliasing`` has been enabled by default.
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
- ``skimage.filters.median`` will change behavior in the future to have an
  identical behavior as ``scipy.ndimage.median_filter``. This behavior can be
  set already using ``behavior='ndimage'``. In 0.16, it will be the default
  behavior and removed in 0.17 as well as the parameter of the previous
  behavior (i.e., ``mask``, ``shift_x``, ``shift_y``) will be removed.


Documentation improvements
--------------------------

- Correct rotate method's center parameter doc (#3341)
- Add Sphinx copybutton (#3530)
- Add glossary to the documentation (#3626)
- Add image of retina to our data (#3748)
- Add microaneurysms() to gallery (#3765)
- Better document remove_small_objects behaviour: int vs bool (#2830)
- Linking preserve_range parameter calls to docs (#3109)
- Update the documentation regarding datalocality (#3127)
- Specify conda-forge channel for scikit-image conda install (#3189)
- Turn DOIs into web links in docstrings (#3367)
- Update documentation for regionprops (#3602)
- DOC: Improve the RANSAC gallery example (#3554)
- DOC: "feature.peak_local_max" : explanation of multiple same-intensity peaks returned by the function; added details on ``exclude_border`` parameter  (#3600)


Improvements
------------

- MNT: handle a deprecation warning for np.linspace and floats for the num parameter (#3453)
- TST: numpy empty arrays are not inherently Falsy (#3455)
-  handle warning in scipy cdist for unused parameters (#3456)
- MNT: don't use filter_warnings in test suite. (#3459)
- Add doc notes on setting up the build environment (#3472)
- Release the GIL in numerous cython functions (#3490)
- Cython touchups to use float32 and float64 (#3493)
- rank_filters: Change how the bitdepth and max_bin are computed to ensure exact warnings. (#3501)
- Rank: Optimize OTSU filter (#3504)
- Rank - Fix rank entropy and OTSU tests (#3506)
- delay importing pyplot in manual segmentation (#3533)
- Get rid of the requirements-parser dependency (#3534)
- filter warning from ``correct_mesh_orientation`` in tests (#3549)
- cloudpickle is really a doc dependency, not a core one (#3634)
- optional dependencies on pip (#3645)
- Fewer test warnings in 3.7 (#3687)
- collections.abc nit (#3692)
- Streamlined issue template (#3697)
- Tighten the PR Template (#3701)
- Use language level to 3 in cython for future compatibility (#3707)
- Update ISSUE_TEMPLATE.md with info about numpy and skimage versions (#3730)
- Use relative imports for many cython modules (#3759)
- Pass tests that don't raise floating point exceptions on arm with soft-fp (#3337)


Other improvements
------------------

- BUG: Fix greycoprops correlation always returning 1 (#2532)
- Add section on API discovery via ``skimage.lookfor`` (#2539)
- Speedup 2D warping for affine transformations (#2902)
- Credit Reviewers in Release Notes (#2927)
- Added small galleries in the API (#2940)
- Use skimage gaussian filter to avoid integer rounding artifacts (#2983)
- Remove Python 2 compatibility (#3000)
- Add ``rectangle_perimeter`` feature to ``skimage.draw`` (#3069)
- Update installation instructions to reference existing requirements specification (#3113)
- Updated release notes with pre 0.13.1 phase (#3114)
- Release guidelines update (#3115)
- Ensure we are installing with / running on Python 3 (#3119)
- Hide warnings in test_unsharp_mask (#3130)
- Process 0.15 deprecations (#3132)
- Documentation: always use dev branch javascript (#3136)
- Add initial airspeed velocity (asv) framework (#3137)
- Supress warnings for flatten during io testing (#3143)
- Recover from exceptions in filters.try_all_threshold() (#3149)
- Fix skimage.test() to run the unittests (#3152)
- skivi: Use qtpy to handle different Qt versions (#3157)
- Refactor python version checking. (#3160)
- Move data_dir to within ``data/__init__.py`` (#3161)
- Move the definition of lookfor out of __init__.py (#3162)
- Normalize the package number to PEP440 (#3163)
- Remove skimage.test as it was never used. (#3164)
- Added a message about qtpy to the INSTALL.rst (#3168)
- Regression fix: Travis should fail if tests fail (#3170)
- Set minimum cython version to ``0.23.4`` (#3171)
- Add rgba2rgb to API docs (#3175)
- Minor doc formatting fixes in video.rst (#3176)
- Decrease the verbosity of the testing (#3182)
- Speedup rgb2gray using matrix multiply (#3187)
- Add instructions for meeseeksdev to PR template (#3194)
- Remove installation instructions for video packages (#3197)
- Big image labeling fix (#3202)
- Handle dask deprecation in cycle_spin (#3205)
- Fix Qt viewer painttool indexing (#3210)
- build_versions.py is no longer hard coded. (#3211)
- Remove dtype constructor call in exposure.rescale_intensity (#3213)
- Various updates to the ASV benchmarks (#3215)
- Add a link to stack overflow on github README (#3217)
- MAINT: remove encoding information in file headers (python 3) (#3219)
- Build tools: Dedicate a --pre build in appveyor and ensure other builds don't download --pre (#3222)
- Fix the human readable error message on a bad build. (#3223)
- Respect input array type in apply_parallel by default (#3225)
- Travis cleanup pip commands (#3227)
- Add benchmarks for morphology.watershed (#3234)
- Correcte docstring formatting so that code block is displayed as code (#3236)
- Defer skimage.io import of matplotlib.pyplot until needed (#3243)
- Add benchmark for Sobel filters (#3249)
- Remove cython md5 hashing since it breaks the build process (#3254)
- Fix typo in documentation. (#3262)
- Issue 3156: skimage/__init__.py Update docstring and fix import *  (#3265)
- Object detector module (#3267)
- Do not import submodules while building (#3270)
- Add benchmark suite for canny (#3271)
- improve segmentation.felzenszwalb document #3264 (#3272)
- Update _canny.py (#3276)
- Add benchmark suite for histogram equalization (#3285)
- fix link to equalist_hist blog reference (#3287)
- .gitignore: novice: Ignore save-demo.jpg (#3289)
- Guide the user of denoise_wavelet to choose an orthogonal wavelet. (#3290)
- Remove unused lib in skimage/__init__.py (#3291)
- BUILD: Add pyproject.toml to ensure cython is present (#3295)
- Handle intersphinx and mpl deprecation warnings in docs (#3300)
- Minor PEP8 fixes (#3305)
- cython: check for presence of cpp files during install from sdist (#3311)
- appveyor: don't upload any artifacts (#3315)
- Add benchmark suite for hough_line() (#3319)
- Novice skip url test (#3320)
- Remove benchmarks from wheel (#3321)
- Add license file to the wheel (binary) distribution (#3322)
- codecov: ignore build scripts in coverage and don't comment on PRs (#3326)
- Matplotlib 2.2.3 +  PyQt5.11 (#3345)
- Allow @hmaarrfk to mention MeeseeksDev to backport. (#3357)
- Add Python 3.7 to the test matrix (#3359)
- Fix deprecated keyword from dask (#3366)
- Incompatible modes with anti-aliasing in skimage.transform.resize (#3368)
- Missing cval parameter in threshold_local (#3370)
- Avoid Sphinx 1.7.8 (#3381)
- Show our data in the gallery (#3388)
- Minor updates to grammar in numpy images page (#3389)
- assert_all_close doesn't exist, make it ``assert_array_equal`` (#3391)
- Better behavior of Gaussian filter for arrays with a large number of dimensions (#3394)
- Allow import/execution with -OO (#3398)
- Mark tests known to fail on 32bit architectures with xfail (#3399)
- Hardcode the inputs to test_ssim_grad (#3403)
- TST: make test_wavelet_denoising_levels compatible with PyWavelets 1.0 (#3406)
- Allow tifffile.py to handle I/O. (#3409)
- Add explicit Trove classifier for Python 3 (#3415)
- Fix error in contribs.py (#3418)
- MAINT: remove pyside restriction since we don't support Python 3.4 anymore (#3421)
- Build tools: simplify how MPL_DIR is obtained. (#3422)
- Build tools: Don't run tests twice in travis. (#3423)
- Build tools: Add an OSX build with optional dependencies. (#3424)
- MAINT: Reverted the changes in #3300 that broke the MINIMIUM_REQUIREMENTS tests (#3427)
- MNT: Convert links using http to https (#3428)
- MAINT: Use upstream colormaps now that matplotlib has been upgraded (#3429)
- Build tools: Make pyamg an optional dependency and remove custom logic (#3431)
- Build tools: Fix PyQt installed in minimum requirements build (#3432)
- MNT: multiprocessing should always be available since we depend on python >=2.7 (#3434)
- MAINT Use np.full instead of cst*np.ones (#3440)
- DOC: Fix LaTeX build via ``make latexpdf``  (#3441)
- Update instructions et al for releases after 0.14.1 (#3442)
- Remove code specific to python 2 (#3443)
- Fix default value of ``methods`` in ``_try_all`` to avoid exception (#3444)
- Fix morphology.local_maxima for input with any dimension < 3 (#3447)
- Use raw strings to avoid unknown escape symbol warnings (#3450)
- Speed up xyz2rgb by clipping output in place (#3451)
- MNT; handle deprecation warnings in tifffile (#3452)
- Build tools: TST: filter away novice deprecation warnings during testing (#3454)
- Build tools: don't use the pytest.fixtures decorator anymore in class fixtures  (#3458)
- Preserving the fill_value of a masked array (#3461)
- Fix VisibleDeprecationWarning from np.histogram, normed=True (#3463)
- Build Tools: DOC: Document that now PYTHONOPTMIZE build is blocked by SciPy (#3470)
- DOC: Replace broken links by webarchive equivalent links (#3471)
- FIX: making the plot_marching_cubes example visible. (#3474)
- Avoid Travis failure regarding ``skimage.lookfor`` (#3477)
- Fix Python executable for sphinx-build in docs Makefile (#3478)
- Build Tools: Block specific Cython versions (#3479)
- Fix typos (#3480)
- Add "optional" indications to docstrings (#3495)
- Rename 'mnxc' (masked normalize cross-correlation) to something more descriptive (#3497)
- Random walker bug fix: no error should be raised when there is nothing to do (#3500)
- Various minor edits for active contour (#3508)
- Fix range for uint32 dtype in user guide (#3512)
- Raise meaningful exception in warping when image is empty (#3518)
- DOC: Development installation instructions for Ubuntu are missing tkinter (#3520)
- Better gallery examples and tests for masked translation registration (#3528)
- DOC: make more docstrings compliant with our standards (#3529)
- Build tools: Remove restriction on simpleitk for python 3.7 (#3535)
- Speedup and add benchmark for ``skeletonize_3d`` (#3536)
- Update requirements/README.md on justification of matplotlib 3.0.0 in favor of #3476 (#3542)
- Doc enhancements around denoising features. (#3553)
- Use 'getconf _NPROCESSORS_ONLN' as fallback for nproc in Makefile of docs (#3563)
- Fix matplotlib set_*lim API deprecations (#3564)
- Switched from np.power to np.cbrt (#3570)
- Filtered out DeprecationPendingWarning for matrix subclass (#3572)
- Add RGB to grayscale example to gallery (#3574)
- Build tools: Refactor check_sdist so that it takes a filename as a parameter (#3579)
- Turn dask to an optional requirement (#3582)
- _marching_cubes_lewiner_cy: mark char as signed (#3587)
- Hyperlink DOIs to preferred resolver (#3589)
- Missing parameter description in ``morphology.reconstruction`` docstring #3581 (#3591)
- Update chat location (#3598)
- Remove orphan code (skimage/filters/_ctmf.pyx). (#3601)
- More explicit example title, better list rendering in plot_cycle_spinning.py (#3606)
- Add rgb to hsv example in the gallery (#3607)
- Update documentation of ``perimeter`` and add input validation (#3608)
- Additionnal mask option to clear_border (#3610)
- Set up CI with Azure Pipelines (#3612)
- [MRG] EHN: median filters will accept floating image (#3616)
- Update Travis-CI to xcode 10.1 (#3617)
- Minor tweaks to _mean_std code (#3619)
- Add explicit ordering of gallery sections (#3627)
- Delete broken links (#3628)
- Build tools: Fix test_mpl_imshow for matplotlib 2.2.3 and numpy 1.16 (#3635)
- First draft of core dev guide (#3636)
- Add more details about the home page build process (#3639)
- Ensure images resources with long querystrings can be read (#3642)
- Delay matplotlib import in skimage/future/manual_segmentation.py (#3648)
- make the low contrast check optional when saving images (#3653)
- Correctly ignore release notes auto-generated for docs (#3656)
- Remove MANIFEST file when making the 'clean' target (#3657)
- Clarify return values in _overlap docstrings in feature/blob.py (#3660)
- Contribution script: allow specification of GitHub development branch (#3661)
- Update core dev guide: deprecation, contributor guide, required experience (#3662)
- Add release notes for 0.14.2 (#3664)
- FIX gallery: Add multichannel=True to match_histogram (#3672)
- MAINT Minor code style improvements (#3673)
- Pass parameters through tifffile plugin (#3675)
- DOC unusused im3d_t in example (#3677)
- Remove wrong cast of Py_ssize_t to int (#3682)
- Build tools: allow python 3.7 to fail, but travis to continue (#3683)
- Build tools: remove pyproject.toml (#3688)
- Fix ValueError: not enough values to unpack (#3703)
- Several fixes for heap.pyx (#3704)
- Enable the faulthandler module during testing (#3708)
- Build tools: Fix Python 3.7 builds on travis (#3709)
- Replace np.einsum with np.tensordot in _upsampled_dft (#3710)
- Fix potential use of NULL pointers (#3717)
- Fix potential memory leak (#3718)
- Fix potential use of NULL pointers (#3719)
- Fix and improve core_cy.pyx (#3720)
- Build tools: Downgrade Xcode to 9.4 on master (#3723)
- Improve visual_test.py (#3732)
- Updated painttool to work with color images and properly scale labels. (#3733)
- Add image.sc forum badge to README (#3738)
- Block PyQt 5.12.0 on Travis (#3743)
- Build tools: Fix matplotlib + qt 5.12 the same way upstream does it (#3744)
- gallery: remove xx or yy  sorted directory names (#3761)
- Allow for f-contiguous 2D arrays in convex_hull_image (#3762)
- Build tools: Set astropy minimum requirement to 1.2 to help the CIs. (#3767)
- Avoid NumPy warning while stacking arrays. (#3768)
- Set CC0 for microaneurysms (#3778)
- Unify LICENSE files for easier interpretation (#3791)
- Readme: Remove expectation for future fix from matplotlib (#3794)
- Improved documentation/test in ``flood()`` (#3796)
- Use ssize_t in denoise cython (#3800)
- Removed non-existent parameter in docstring (#3803)
- Remove redundant point in draw.polygon docstring example (#3806)
- Ensure watershed auto-markers respect mask (#3809)


75 authors added to this release [alphabetical by first name or login]
----------------------------------------------------------------------

- Abhishek Arya
- Adrian Roth
- alexis-cvetkov (Alexis Cvetkov-Iliev)
- Ambrose J Carr
- Arthur Imbert
- blochl (Leonid Bloch)
- Brian Smith
- Casper da Costa-Luis
- Christian Rauch
- Christoph Deil
- Christoph Gohlke
- Constantin Pape
- David Breuer
- Egor Panfilov
- Emmanuelle Gouillart
- fivemok
- François Boulogne
- François Cokelaer
- François-Michel De Rainville
- Genevieve Buckley
- Gregory R. Lee
- Gregory Starck
- Guillaume Lemaitre
- Hugo
- jakirkham (John Kirkham)
- Jan
- Jan Eglinger
- Jathrone
- Jeremy Metz
- Jesse Pangburn
- Johannes Schönberger
- Jonathan J. Helmus
- Josh Warner
- Jotham Apaloo
- Juan Nunez-Iglesias
- Justin
- Katrin Leinweber
- Kim Newell
- Kira Evans
- Kirill Klimov
- Lars Grueter
- Laurent P. René de Cotret
- Legodev
- mamrehn
- Marcel Beining
- Mark Harfouche
- Matt McCormick
- Matthias Bussonnier
- mrastgoo
- Nehal J Wani
- Nelle Varoquaux
- Onomatopeia
- Oscar Javier Hernandez
- Page-David
- PeterJackNaylor
- PinkFloyded
- R S Nikhil Krishna
- ratijas
- Rob
- robroooh
- Roman Yurchak
- Sarkis Dallakian
- Scott Staniewicz
- Sean Budd
- shcrela
- Stefan van der Walt
- Taylor D. Scott
- Thein Oo
- Thomas Walter
- Tom Augspurger
- Tommy Löfstedt
- Tony Tung
- Vilim Štih
- yangfl
- Zhanwen "Phil" Chen


46 reviewers added to this release [alphabetical by first name or login]
------------------------------------------------------------------------

- Abhishek Arya
- Adrian Roth
- Alexandre de Siqueira
- Ambrose J Carr
- Arthur Imbert
- Brian Smith
- Christian Rauch
- Christoph Gohlke
- David Breuer
- Egor Panfilov
- Emmanuelle Gouillart
- Evan Putra Limanto
- François Boulogne
- François Cokelaer
- Gregory R. Lee
- Grégory Starck
- Guillaume Lemaitre
- Ilya Flyamer
- jakirkham
- Jarrod Millman
- Johannes Schönberger
- Josh Warner
- Jotham Apaloo
- Juan Nunez-Iglesias
- Justin
- Lars Grueter
- Laurent P. René de Cotret
- Marcel Beining
- Mark Harfouche
- Matthew Brett
- Matthew Rocklin
- Matti Picus
- mrastgoo
- Onomatopeia
- PeterJackNaylor
- Rob
- Roman Yurchak
- Scott Staniewicz
- Stefan van der Walt
- Thein Oo
- Thomas A Caswell
- Thomas Walter
- Tom Augspurger
- Tomas Kazmar
- Tommy Löfstedt
- Vilim Štih

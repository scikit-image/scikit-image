scikit-image 0.13.1 is a bug-fix and compatibility update. See below for
the many new features in 0.13.0.

The main contribution in 0.13.1 is Jarrod Millman's valiant work to ensure
scikit-image works with both NetworkX 1.11 and 2.0 (#2766). Additional updates
include:

- Bug fix in similarity transform estimation, by GitHub user @zhongzyd (#2690)
- Bug fixes in ``skimage.util.plot_matches`` and ``denoise_wavelet``,
  by Gregory Lee (#2650, #2640)
- Documentation updates by Egor Panfilov (#2716) and Jirka Borovec (#2524)
- Documentation build fixes by Gregory Lee (#2666, #2731), Nelle
  Varoquaux (#2722), and Stéfan van der Walt (#2723, #2810)


Announcement: scikit-image 0.13.0
=================================

We're happy to (finally) announce the release of scikit-image v0.13.0!

scikit-image is an image processing toolbox for SciPy that includes algorithms
for segmentation, geometric transformations, color space manipulation,
analysis, filtering, morphology, feature detection, and more.

For more information, examples, and documentation, please visit our website:

http://scikit-image.org

and our gallery of examples

http://scikit-image.org/docs/dev/auto_examples/

Highlights
----------

This release is the result of a year of work, with over 200 pull requests by
82 contributors. Highlights include:

- Improved n-dimensional image support. This release adds nD support to:

  * ``regionprops`` computation for centroids (#2083)
  * ``segmentation.clear_border`` (#2087)
  * Hessian matrix (#2194)

- In addition, the following new functions support nD images:

  * new wavelet denoising function, ``restoration.denoise_wavelet``
    (#1833, #2190, #2238, #2240, #2241, #2242, #2462)
  * new thresholding functions, ``filters.threshold_sauvola`` and
    ``filters.threshold_niblack`` (#2266, #2441)
  * new local maximum, local minimum, hmaxima, hminima functions (#2449)

- Grey level co-occurrence matrix (GLCM) now works with uint16 images
- ``filters.try_all_threshold`` to rapidly see output of various thresholding
  methods
- Frangi and Hessian filters (2D only) (#2153)
- New *compact watershed* algorithm in ``segmentation.watershed`` (#2211)
- New *shape index* algorithm in ``feature.shape_index`` (#2312)

New functions and features
--------------------------

- Add threshold minimum algorithm (#2104)
- Implement mean and triangle thresholding (#2126)
- Add Frangi and Hessian filters (#2153)
- add bbox_area to region properties (#2187)
- colorconv: Add rgba2rgb() (#2181)
- Lewiner marching cubes algorithm (#2052)
- image inversion (#2199)
- wavelet denoising (from #1833) (#2190)
- routine to estimate the noise standard deviation from an image (#1837)
- Add compact watershed and clean up existing watershed (#2211)
- Added the missing 'grey2rgb' function. (#2316)
- Shape index (#2312)
- Fundamental and essential matrix 8-point algorithm (#1357)
- Add YUV, YIQ, YPbPr, YCbCr colorspaces
- Detection of local extrema from morphology (#2449)
- shannon entropy (#2416)

Documentation improvements
--------------------------

- add details about github SSH keys in contributing page (#2073)
- Add example for felzenszwalb image segmentation (#2096)
- Sphinx gallery for example gallery (#2078)
- Improved region boundary RAG docs (#2106)
- Add gallery Lucy-Richardson deconvolution algorithm (#2376)
- Gallery: Use Horse to illustrate Convex Hull (#2431)
- Add working with OpenCV in user guide (#2519)

Code improvements
-----------------

- Remove lena image from test suite (#1985)
- Remove duplicate mean calculation in skimage.feature.match_template (#1980)
- Add nD support to clear_border (#2087)
- Add uint16 images support for co-occurrence matrix (#2095)
- Add default parameters for Gaussian and median filters (#2151)
- try_all to choose the best threshold algorithm (#2110)
- Add support for multichannel in Felzenszwalb segmentation (#2134)
- Improved SimilarityTransform, new EuclideanTransform class (#2044)
- ENH: Speed up Hessian matrix computation (#2194)
- add n-dimensional support to denoise_wavelet (#2242)
- Speedup ``inpaint_biharmonic`` (#2234)
- Update hessian matrix code to include order kwarg (#2327)
- Handle cases for label2rgb where input labels are negative and/or
  nonconsecutive (#2370)
- Added watershed_line parameter (#2393)

API Changes
-----------

- Remove deprecated ``filter`` module. Use ``filters`` instead. (#2023)
- Remove ``skimage.filters.canny`` links. Use ``feature.canny`` instead. (#2024)
- Removed Python 2.6 support and related checks (#2033)
- Remove deprecated {h/v}sobel, {h/v}prewitt, {h/v}scharr,
  roberts_{positive/negative} filters (#2159)
- Remove deprecated ``_mode_deprecations`` (#2156)
- Remove deprecated None defaults in ``rescale_intensity`` (#2161)
- Parameters ``ntiles_x`` and ``ntiles_y`` have been removed from
  ``exposure.equalize_adapthist``
- The minimum NumPy version is now 1.11, and the minimum SciPy version is now
  0.17

Deprecations
------------

- clip_negative will be set to false by default in version 0.15
  (func: dtype_limits) (#2228)
- Deprecate "dynamic_range" in favor of "data_range" (#2384)
- The default value of the ``circle`` argument to ``radon`` and ``iradon``
  transforms will be ``True`` in 0.15 (#2235)
- The default value of ``multichannel`` for ``denoise_bilateral`` and
  ``denoise_nl_means`` will be ``False`` in 0.15
- The default value of ``block_norm`` in ``feature.hog`` will be L2-Hysteresis in
  0.15.
- The ``threshold_adaptive`` function is deprecated. Use ``threshold_local``
  instead.
- The default value of ``mode`` in ``transform.swirl``, ``resize``, and ``rescale``
  will be "reflect" in 0.15.

Contributors to this release
----------------------------

- AbdealiJK
- Rodrigo Benenson
- Vighnesh Birodkar
- Jirka Borovec
- François Boulogne
- Matthew Brett
- Sarwat Fatima
- Rachel Finck
- Joe Futrelle
- Jeroen Van Goey
- Christoph Gohlke
- Roman Golovanov
- Emmanuelle Gouillart
- Anshita Gupta
- David Haberthür
- Jeff Hemmelgarn
- Hiyorimi
- Daniel Hyams
- Alex Izvorski
- Kyle Jackson
- Jirka
- JohnnyTeutonic
- Kevin Keraudren
- Almar Klein
- Yu Kobayashi
- Moriyoshi Koizumi
- Lachlan
- LachlanD
- George Laurent
- Gregory R. Lee
- Evan Limanto
- Ben Longo
- Victor MARTIN
- Oliver Mader
- Ken'ichi Matsui
- Jeremy Metz
- Jeyson Molina
- Michael Mueller
- Juan Nunez-Iglesias
- Egor Panfilov
- Paul
- PengchengAi
- Francisco de la Peña
- Pavlin Poličar
- Orion Poplawski
- Zoe Richards
- Todd V. Rovito
- Christian Sachs
- Sanya
- Johannes Schönberger
- Pavel Shevchuk
- Scott Sievert
- Steven Silvester
- Shaun Singh
- Sourav Singh
- Alexandre Fioravante de Siqueira
- Samuel St-Jean
- Noah Stier
- Ole Streicher
- Martin Thoma
- Matěj Týč
- Viraj
- Stefan van der Walt
- Josh Warner
- Olivia Wilson
- Robin Wilson
- Martin Zackrisson
- Yue Zheng
- Nick Zoghb
- alexandrejaguar
- almar
- cespenel
- danielballan
- dmesejo
- eli
- jwittenbach
- lgeorge
- mljli
- rjeli
- skrish13
- tseclaudia
- walter

Pull requests merged in this release
------------------------------------

- Warn if user tries to build with older Cython version (#1986)
- Remove lena image from test suite (#1985)
- Add inpaint to module init (#1987)
- Pre-calculate tempate mean (#1980)
- rgb2grey -> grey2rgb (#1989)
- Also expose rgb2gray as rgb2grey (#1990)
- Remove all .md5 files on clean (#1992)
- avoid deprecation warnings when calling compute_ssim with multichannel=True (#1994)
- DOC: Suggest multichannel=True in compute_ssim error (#1999)
- [DOC] add link to guide (#2001)
- Fix docs-->doc in CONTRIBUTING (#2009)
- Turn ``dask`` into an optional dependency (#2013)
- Correct regexp for catching mpl warnings (#2014)
- BUILD: Use --pre flag for Travis pip installs. (#1938)
- Github templates (#1954)
- added doc to PaintTool (#1934)
- skimage.segmentation.quickshift signature is missing from API docs (#2017)
- MAINT: Upgrade tifffile (#2016)
- Modified .gitignore to properly ignore auto_example files (#1966)
- MAINT: Switch from coveralls -> codecov in CI build (#2015)
- skimage.segmentation.quickshift signature is missing from API docs, third attempt (#2021)
- MAINT: Remove deprecated ``filter`` module (#2023)
- Remove ``skimage.filters.canny`` links (#2024)
- Document regionprops bbox property. (#2030)
- Fix URL to texturematch paper (#2031)
- Improved skimage.segmentation.active_contour input arguments' dtype support (#2032)
- Fix local test function (#2034)
- Removed Python 2.6 support and related checks (#2033)
- Test on OSX (#2038)
- Change coverage badge to codecov (#2055)
- TST: Speed up bilateral filter tests (#2061)
- Speed up colorconv._convert (#2064)
- FIX: Fix import of 'warn' in qt_plugin (#2070)
- Add YUV, YIQ, YPbPr, YCbCr colorspaces
- adding details about github SSH keys in contributing page (#2073)
- ENH: Pass np.random.RandomState to RANSAC (#2072)
- Handle IO objects with tifffile (#2046)
- Updated centroid to use coords - works in 3d (#2083)
- [WIP] Hierarchical Merging of Region Boundary RAGs (#2058)
- Add nD support to clear_border (#2087)
- DOC: update for new API (minor) (#2090)
- Add example for felzenszwalb image segmentation (#2096)
- DOC: add space before column on variable def (minor...) (#2102)
- DOC: Guide new contributors to HTTPS, not SSH (#2082)
- Add François Boulogne to the mailmap (#2117)
- Move skimage.filters.rank description and todos from README into docstring. (#2115)
- Fixing Error and documentation on Otsu Threshold (#2118)
- Add scuinto's second email address to mailmap (#2122)
- MAINT: around label and regionprops functions. (#2100)
- Add threshold minimum algorithm (#2104)
- Sphinx gallery for example gallery (#2078)
- DOC: make a title shorter in gallery (#2128)
- DOC: refactor axes with lists (#2129)
- DOC ENH + API fix on houghline transform (#2089)
- Fix indentation for example script (#2136)
- Implement mean and triangle thresholding (#2126)
- Move ``skimage.measure.label`` references to the docstring (#2143)
- Fix outdated GraphicsGems link (#2149)
- Docstring (#2145)
- Add uint16 images support for co-occurrence matrix (#2095)
- Remove deprecared {h/v}sobel, {h/v}prewitt, {h/v}scharr, roberts_{positive/negative} filters (#2159)
- Remove deprecated ``_mode_deprecations`` (#2156)
- Default parameters (#2151)
- ENH: try_all to choose the best threshold algorithm and DOC refactoring (#2110)
- BUGFIX: inverse_map should not be None (#2160)
- Switched felzenszwalb gray to multichannel version (#2134)
- Writing, style, and PEP8 fixes for greycomatrix (#2157)
- Add Frangi and Hessian filters (#2153)
- Improved SimilarityTransform, new EuclideanTransform class (#2044)
- color.colorconv: Fix documentation of rgb2gray() (#2169)
- fix region merging in ``segmentation.felzenszwalb`` (#2164)
- Remove deprecated None defaults in ``rescale_intensity`` (#2161)
- DOC: add a note to template_match (#2176)
- Added chapter title formatting for numpy_images.rst (#2177)
- Fix threshold_triangle to work with non-integer images. (#2171)
- Improved region boundary RAG docs (#2106)
- ENH add bbox_area to region properties (#2187)
- colorconv: Add rgba2rgb() (#2181)
- DOC: add DOI to references (#2188)
- remove local threshold in try_all_threshold (#2180)
- DOC: add a note on warning treatment (#2198)
- ENH: Speed up Hessian matrix computation (#2194)
- Add missing unittests for data and convert horse to binary (#2196)
- Fix ssim example (#2208)
- [MRG] MAINT: Replaced gaussian_filter with filters.gaussian (#2210)
- [MRG] DOC: corrected mssim docstring to return float (#2218)
- FEAT: Lewiner marching cubes algorithm (#2052)
- Fix bug in salt and pepper noise (#2223)
- TST: Updated AppVeyor to use Conda, added msvc_runtime (#2217)
- Improve docstrings for captions (#2185)
- Add task update version on wikipedia (#2230)
- NEW + DOC: image inversion (#2199)
- ENH: Implements wavelet denoising (from #1833) (#2190)
- TEST: define seed in setup() / Fix random test failure (#2227)
- add n-dimensional support to denoise_wavelet (#2242)
- API: clip_negative will be set to false by default in version 0.15 (func: dtype_limits) (#2228)
- Speedup ``inpaint_biharmonic`` (#2234)
- MAINT dtype.py (PEP8) (#2231)
- Removed unused extend_image (#2251)
- ENH:  routine to estimate the noise standard deviation from an image (#1837)
- Restrict sphinx builds to a single process.  Remove vendored numpydoc. (#2257)
- Added more specific check for image shape in threshold_otsu warning (#2259)
- Allow running ``setup.py egg_info`` without numpy installed. (#2260)
- Add compact watershed and clean up existing watershed (#2211)
- Use numpy.pad directly, removing most shipped code in util.pad (#2265)
- DOC: fix references (#2262)
- DOC: tiny fixes in gallery (#2226)
- DOC: fix typo (#2274)
- Update Manifest.in (#2255)
- Bugfix unbounded correlation -- Dhyams fix for match template (#2263)
- DOC: Refactor example skeletonize in the gallery (#2141)
- [MRG+1] Insert metadata in docstrings of images in skimage.data.* (#2236)
- MAINT: Radon (docstring, API, PEP8) (#2235)
- [MRG+2] MAINT: Fix numpy deprecation (#2283)
- Reduce whitespace around plots (#2144)
- [MRG+1] By default, clear_border is not inplace (#2285)
- Remove unused imports in ``transform.{pyx/pxd}`` (#2288)
- [MRG+1] Add community guidelines to doc navigation (#2287)
- Adding colors to the IHC (#2279)
- FIX: select num_peaks if labels is specified  (#2098)
- [MRG+1] Add felzenszwalb shape validation (#2286)
- [MRG+1] more closesly match the BayesShrink paper in _wavelet_threshold (#2241)
- Remove usages of ``subplots_adjust`` (#2289)
- [MRG+1] Change documentation page favicon (#2291)
- [MRG+1] TST: prefer ``assert_`` from numpy.testing over assert (#2298)
- TSTFIX: Bug fix for development version of scipy (#2302)
- Enhance ``compare_ssim`` docstring (#2314)
- Added the missing 'grey2rgb' function. (#2316)
- PEP8 (#2304)
- Made Python wrappers for public Cython functions (#2303)
- Update mailing list location (#2328)
- Shape Index (#2312)
- Add pywavelets to runtime requirements in DEPENDS.txt (#2238)
- Refactor variable names in ``skimage.draw`` (#2321)
- Fix display problem when printing error messages (#2326)
- Added catch for zero image in threshold_li (#2338)
- FIX: Modified peak_local_max to use relabel_sequential (#2341)
- Update favicon in _static (#2355)
- Remove incorrect input type assumption in doctrings for rgb2hsv and h… (#2354)
- Update the default boundary mode in transform.swirl (#2331)
- Update imread() document (#2358)
- Check for valid mode in random_walker(). (#2362)
- Fix 1 broken test in _shared not executed by nose/travis (#2229)
- Update hessian matrix code to include order kwarg (#2327)
- Clarify purpose of beta1 and beta2 parameters in documentations of sk… (#2382)
- Handle cases for label2rgb where input labels are negative and/or nonconsecutive (#2370)
- Update ``exposure.equalize_adapthist`` args and docstring (#2220)
- Fix (x, y) origin description in user guide (#2385)
- Update docstring for show_rag method (#2375)
- Fix display problem when printing error messages (#2372)
- Added a check for empty array in _shared.utils.py (#2364)
- Fix no peaks blob log (#2349)
- ENH: Extend draw.ellipse with orientation kwarg (#2366)
- Fundamental and essential matrix 8-point algorithm (#1357)
- Fix reference to travis notes (#2403)
- Fix deprecated option in sphinx that causes warning treated as error in travis (#2395)
- Update Travis Script (#2374)
- Remove the freeimage plugin (#1933)
- Fix shape type for histogram (#2417)
- Add illuminant and observer parameters to the rgb2lab and lab2rgb functions. (#2306)
- PEP8 (#2413)
- MAINT: merge lists of dtypes (#2420)
- Made (partially) ``pep8``-compliant (#2392)
- Added titles and text to make plot_brief.py example more clear (#2193)
- DOC: Add reference to standard illuminant (#2418)
- Added titles and text to the subplots to make it easier to new comers for plot_censure.py example (#2191)
- Deprecate "dynamic_range" in favor of "data_range" (#2384)
- Make PR 2266 n-D compatible (#4)
- Add new "thin" method based on Guo and Hall 1989 (#2294)
- local threshold niblack sauvola (from Jeysonmc PR) (#2266)
- stable ellipse fitting (#2394)
- Add gallery Lucy-Richardson deconvolution algorithm (#2376)
- Improve SIFT loader docstring according to comments and StackOverflow (#2404)
- Change to Javascript loading of search index (patch by Julian Taylor) (#2438)
- Fix segfault in connected components (patch by Yaroslav Halchenko) (#2437)
- Refactor ``util/dtype.py`` (#2425)
- ENH: Gallery, various little stylish corrections (DFT example). (#2430)
- Make peak_local_max return indices sorted, always (#2435)
- Correct comment of probabilistic_hough_line(). (#2448)
- Added watershed_line parameter (#2393)
- Solved Gaussian value range #2383 (#2388)
- Gallery: Use Horse to illustrate Convex Hull (#2431)
- MRG: update build matrix for Python 3.6 (#2451)
- Wavelet denoising in YCbCr color space (#2240)
- Gallery: Use gray cmap for coins (#2459)
- Bug fix for Sauvola and Niblack thresholding (#2441)
- MAINT: removes _wavelet_threshold docstring (#2460)
- BUG: fix denoise_wavelet for odd-length input (#2462)
- MAINT: warns for new multichannel default in denoise_{bilateral, nl_means} (#2467)
- Various enhancements in gallery for denoising (#2461)
- Tool for checking completeness of sdist (#2085)
- Add different ``skimage.hog`` blocks normalization methods (#2040)
- DOC: fix typos and add references (#2478)
- update sphinx gallery to 0.1.8 (#2474)
- DOC: Fix typo in gaussian filter docstring (#2487)
- Add threshold_local, deprecate old threshold_adaptive API (#2490)
- Default edge mode change for resize and rescale (#2484)
- Add ``dask[array]`` to optional requirements (#2494)
- DOC:  Adds an instruction to CONTRIBUTING.txt & Updates the git install link for Windows (#2495)
- ENH: generalize hough_peak functions (#2109)
- Fix gallery examples (#2504)
- Bump min scipy version (#2254)
- DOC: img_as_float add note about range if input dtype is float (#2499)
- Update tifffile for 2017.01.12 changes (#2497)
- Replace local_sum by block_reduce in docstrings. (#2498)
- MAINT: pass scipys truncate parameter to gaussian filter API (#2508)
- DOC: gallery: join segmentation: enhancement (#2507)
- Tidy up the deployment of dev docs (#2516)
-  Do not require cython for normal builds (#2509)
- Fix broken ``test_ncut_stable_subgraph`` for Python 3.6, enable Python 3.6 in Travis (#2511)
- Improved background labeling (#2381)
- For imread's load_func, make the img_num argument optional (#2054)
- Make compatible with current networkx master (#2455)
- Miscellaneous tidying in HOG code (#2526)
- BUG: Fix NumPy error when no descriptors are returned by ORB (#2537)
- BUG: ValueError in restoration.denoise_bilateral for zeros image (#2533)
- Fix link to Python XY (#2542)
- TST: fix ValueError with scipy-0.19.0rc2 (#2544)
- DOC: Update URL for data.coins() (#2548)
- Replace GRIN URL with Flickr URL (#2547)
- Have ``threshold_minimum`` return identical results on i686 and x86_64 (#2549)
- Minor Fix (Issue #2554) (#2556)
- Remove ``offset`` parameter from ``filters.threshold_sauvola`` docstring (#2566)
- Practical guide to reading video files (#1012)
- Remove dask from ``requirements.txt`` (#2572)
- Fix ``morphology.watershed`` error message (#2570)
- DOC: Added working with OpenCV in user guide (#2519)
- NEW: add shannon entropy (#2416)
- Fix typo in ylabel of GLCM demo (#2576)
- Detection of local extrema from morphology (#2449)
- Add extrema functions to ``__init__`` (#2588)


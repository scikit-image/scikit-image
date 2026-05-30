scikit-image 0.18.3 (2021-08-24)
================================

We're happy to announce the release of scikit-image v0.18.3!

scikit-image is an image processing toolbox for SciPy that includes algorithms
for segmentation, geometric transformations, color space manipulation,
analysis, filtering, morphology, feature detection, and more.

This is a small bugfix release for compatibility with Pooch 1.5 and SciPy 1.7.

Bug fixes
---------
- Only import from Pooch's public API. This resolves an import failure with
  Pooch 1.5.0. (#5531, backport of #5529)
- Do not use deprecated ``scipy.linalg.pinv2`` in ``random_walker`` when
  using the multigrid solver. (#5531, backport of #5437)

3 authors added to this release [alphabetical by first name or login]
---------------------------------------------------------------------
David Manthey
Gregory Lee
Mark Harfouche

3 reviewers added to this release [alphabetical by first name or login]
-----------------------------------------------------------------------
Gregory Lee
Juan Nunez-Iglesias
Mark Harfouche


scikit-image 0.18.2 (2021-06-10)
================================

We're happy to announce the release of scikit-image v0.18.2!

scikit-image is an image processing toolbox for SciPy that includes algorithms
for segmentation, geometric transformations, color space manipulation,
analysis, filtering, morphology, feature detection, and more.

This release mostly serves to add wheels for the aarch64 architecture; it also
fixes a couple of minor bugs.

For more information, examples, and documentation, please visit our website:

https://scikit-image.org

Bug fixes
---------
- allow either SyntaxError or OSError for truncated JPG (#5315, #5334)
- Fix sphinx: role already being registered (#5319, #5335)

Development process
-------------------
- Update pyproject.toml to ensure pypy compatibility and aarch compatibility (#5326, #5328)
- Build aarch64 wheels (#5197, #5210)
- See if latest Ubuntu image fixes QEMU CPU detection issue (#5227, #5233)
- Rename `master` to `main` throughout (#5243, #5295)
- Fixup test for INSTALL_FROM_SDIST (#5283, #5296)
- Remove unnecessary manual installation of packages from before_install (#5298)
- Use manylinux2010 for python 3.9+ (#5303, #5310)
- add numpy version specification on aarch for cpython 3.8 (#5374, #5375)

7 authors added to this release [alphabetical by first name or login]
---------------------------------------------------------------------
- François Boulogne
- Janakarajan Natarajan
- Juan Nunez-Iglesias
- John Lee
- Mark Harfouche
- MeeseeksMachine
- Stéfan van der Walt


9 reviewers added to this release [alphabetical by first name or login]
-----------------------------------------------------------------------
- Alexandre de Siqueira
- Gregory R. Lee
- Juan Nunez-Iglesias
- Marianne Corvellec
- Mark Harfouche
- Matti Picus
- Matthias Bussonnier
- Riadh Fezzani
- Stéfan van der Walt


scikit-image 0.18.1 (2020-12-23)
================================

This is a bug fix release and contains the following two bug fixes:

- Fix indexing error for labelling in large (>2GB) arrays (#5143, #5151)
- Only use retry_if_failed with recent pooch (#5148)

See below for the new features and API changes in 0.18.0.

scikit-image 0.18.0 (2020-12-15)
================================

We're happy to announce the release of scikit-image v0.18.0!

scikit-image is an image processing toolbox for SciPy that includes algorithms
for segmentation, geometric transformations, color space manipulation,
analysis, filtering, morphology, feature detection, and more.

This release of scikit-image drops support for Python 3.6 in accordance with
the `NEP-29 Python and Numpy version support community standard
<https://numpy.org/neps/nep-0029-deprecation_policy.html>`_: Python 3.7 or
newer is required to run this version.

For more information, examples, and documentation, please visit our website:

https://scikit-image.org


New Features
------------

- Add the iterative Lucas-Kanade (iLK) optical flow method (#4161)
- Add Feret diameter in region properties (#4379, #4820)
- Add functions to compute Euler number and Crofton perimeter estimation (#4380)
- Add a function to compute the Hausdorff distance (#4382)
- Added 3D support for many filters in ``skimage.filters.rank``.
- An experimental implementation of trainable pixel segmentation, aiming for
  compatibility with the scikit-learn API, has been added to
  ``skimage.future``. Try it out! (#4739)
- Add a new function ``segmentation.expand_labels`` to dilate labels while
  preventing overlap (#4795)
- It is now possible to pass extra measurement functions to
  ``measure.regionprops`` and ``regionprops_table`` (#4810)
- Add rolling ball algorithm for background subtraction (#4851)
- New sample images have been added in the ``data`` subpackage: ``data.eagle``
  (#4922), ``data.human_mitosis`` (#4939), ``data.cells3d`` (#4951), and
  ``data.vortex`` (#5041). Also note that the image for ``data.camera`` has
  been changed due to copyright issues (#4913).
- ``skimage.feature.structure_tensor`` now supports 3D (and nD) images as input
  (#5002)
- Many thresholding methods can now receive a precomputed histogram as input,
  resulting in significant speedups if multiple methods are tried on the same
  image, or if a fast histogram method is used. (#5006)
- ``measure.regionprops`` now supports multichannel intensity images (#5037)

Documentation
-------------

- Add an example to the flood fill tutorial (#4619)
- Docstring enhancements for marching cubes and find_contours (#4641)
- A new tutorial presenting a cell biology example has been added to the
  gallery (#4648). Special thanks to Pierre Poulain and Fred Bernard
  (Université de Paris and Institut Jacques Monod) for scientific review of
  this example!
- Improve register rotation example with notes and references (#4723)
- Add versionadded for new scalar type support for "scale" param in
  ``transform.AffineTransform`` (#4733)
- New tutorial on `visualizing 3D data <https://scikit-image.org/docs/dev/auto_examples/applications/plot_3d_image_processing.html>`_ (#4850)
- Add example for 3D adaptive histogram equalization (AHE) (#4658)
- Automatic formatting of docstrings for improved consistency (#4849)
- Improved docstring for ``rgb2lab`` (#4839) and ``marching_cubes`` (#4846)
- Improved docstring for ``measure.marching_cubes``, mentioning how to decimate a
  mesh using mayavi (#4846)
- Document how to contribute a gallery example. (#4857)
- Fix and improve entropy example (#4904)
- expand the benchmarking section of the developer docs (#4905)
- Improved docstring for ``util.random_noise`` (#5001)
- Improved docstrings for ``morphology.h_maxima`` and ``morphology.h_minima``
  (#4929).
- Improved docstring for ``util.img_as_int`` (#4888).
- A new example demonstrates interactive exploration of regionprops results
  using the PyData stack (pandas, seaborn) at
  <https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_regionprops.html>`_
  (#5010).
- Documentation has been added to explain
  `how to download example datasets <https://scikit-image.org/docs/dev/user_guide/install.html#downloading-all-demo-datasets>`_
  which are not installed with scikit-image (#4984). Similarly, the contributor
  guide has been updated to mention how to host new datasets in a gitlab
  repository (#4892).
- The `benchmarking section of the developer documentation <https://scikit-image.org/docs/dev/development/contribute.html#benchmarks>`_
  has been expanded (#4905).
- Added links to the image.sc forum in example pages (#5094, #5096)
- Added missing datasets to gallery examples (#5116, #5118)
- Add farid filters in __all__, to populate the documentation (#5128, #5129)
- Proofread gallery example for rank filters. (#5126, #5136)

Improvements
------------

- float32 support for SLIC (#4683), ORB (#4684, #4697), BRIEF (#4685),
  ``pyramid_gaussian`` (#4696), Richardson-Lucy deconvolution (#4880)
- In ``skimage.restoration.richardson_lucy``, computations are now done in
  single-precision when the input image is single-precision. This can give a
  substantial performance improvement when working with single precision data.
- Richardson-Lucy deconvolution now has a ``filter_epsilon`` keyword argument
  to avoid division by very small numbers (#4823)
- Add default level parameter (max-min) / 2 in ``measure.find_contours`` (#4862)
- The performance of the SLIC superpixels algorithm
  (``skimage.segmentation.slice``) was improved for the case where a mask
  is supplied by the user (#4903). The specific superpixels produced by
  masked SLIC will not be identical to those produced by prior releases.
- ``exposure.adjust_gamma`` has been accelerated for ``uint8`` images by using
  a look-up table (LUT) (#4966).
- ``measure.label`` has been accelerated for boolean input images, by using
  ``scipy.ndimage``'s implementation for this case (#4945).
- ``util.apply_parallel`` now works with multichannel data (#4927).
- ``skimage.feature.peak_local_max`` supports now any Minkowski distance.
- We now use sparse cross-correlation to accelerate local thresholding
  functions (#4912)
- ``morphology.convex_hull_image`` now uses much less memory by checking hull
  inequalities in sequence (#5020)
- Polygon rasterization is more precise and will no longer potentially exclude
  input vertices. (#5029)
- Add data optional requirements to allow pip install scikit-image[data]
  (#5105, #5111)
- OpenMP support in MSVC (#4924, #5111)
- Restandardize handling of Multi-Image files (#2815, #5132)
- Consistent zoom boundary behavior across SciPy versions (#5131, #5133)

API Changes
-----------

- ``skimage.restoration.richardson_lucy`` returns a single-precision output
  when the input is single-precision. Prior to this release, double-precision
  was always used. (#4880)
- The default value of ``threshold_rel`` in ``skimage.feature.corner`` has
  changed from 0.1 to None, which corresponds to letting
  ``skimage.feature.peak_local_max`` decide on the default. This is currently
  equivalent to ``threshold_rel=0``.
- In ``measure.label``, the deprecated ``neighbors`` parameter has been
  removed. (#4942)
- The image returned by ``data.camera`` has changed because of copyright
  issues (#4913).

Bug fixes
---------

- A bug in ``label2rgb`` has been fixed when the input image had np.uint8
  dtype (#4661)
- Fixed incorrect implementation of ``skimage.color.separate_stains`` (#4725)
- Many bug fixes have been made in ``peak_local_max`` (#2592, #4756, #4760,
  #5047)
- Fix bug in ``random_walker`` when input labels have negative values (#4771)
- PSF flipping is now correct for Richardson-Lucy deconvolution work in >2D (#4823)
- Fix equalize_adapthist (CLAHE) for clip value 1.0 (#4828)
- For the RANSAC algorithm, improved the case where all data points are
  outliers, which was previously raising an error
  (#4844)
- An error-causing bug has been corrected for the ``bg_color`` parameter in
  ``label2rgb`` when its value is a string (#4840)
- A normalization bug was fixed in ``metrics.variation_of_information``
  (#4875)
- Euler characteristic property of ``skimage.measure.regionprops`` was erroneous
  for 3D objects, since it did not take tunnels into account. A new implementation
  based on integral geometry fixes this bug (#4380).
- In ``skimage.morphology.selem.rectangle`` the ``height`` argument
  controlled the width and the ``width`` argument controlled the height.
  They have been replaced with ``nrow`` and ``ncol``. (#4906)
- ``skimage.segmentation.flood_fill`` and ``skimage.segmentation.flood``
  now consistently handle negative values for ``seed_point``.
- Segmentation faults in ``segmentation.flood`` have been fixed (#4948, #4972)
- A segfault in ``draw.polygon`` for the case of 0-d input has been fixed
  (#4943).
- In ``registration.phase_cross_correlation``, a ``ValueError`` is raised when
  NaNs are found in the computation (as a result of NaNs in input images).
  Before this fix, an incorrect value could be returned where the input images
  had NaNs (#4886).
- Fix edge filters not respecting padding mode (#4907)
- Use v{} for version tags with pooch (#5104, #5110)
- Fix compilation error in XCode 12 (#5107, #5111)

Deprecations
------------

- The ``indices`` argument in ``skimage.feature.peak_local_max`` has been
  deprecated. Indices will always be returned. (#4752)
- In ``skimage.feature.structure_tensor``, an ``order`` argument has been
  introduced which will default to 'rc' starting in version 0.20. (#4841)
- ``skimage.feature.structure_tensor_eigvals`` has been deprecated and will be
  removed in version 0.20. Use ``skimage.feature.structure_tensor_eigenvalues``
  instead.
- The ``skimage.viewer`` subpackage and the ``skivi`` script have been
  deprecated and will be removed in version 0.20. For interactive visualization
  we recommend using dedicated tools such as `napari <https://napari.org>`_ or
  `plotly <https://plotly.com>`_. In a similar vein, the ``qt`` and ``skivi``
  plugins of ``skimage.io`` have been deprecated
  and will be removed in version 0.20. (#4941, #4954)
- In ``skimage.morphology.selem.rectangle`` the arguments ``width`` and
  ``height`` have been deprecated. Use ``nrow`` and ``ncol`` instead.
- The explicit setting ``threshold_rel=0` was removed from the Examples of the
  following docstrings: ``skimage.feature.BRIEF``,
  ``skimage.feature.corner_harris``, ``skimage.feature.corner_shi_tomasi``,
  ``skimage.feature.corner_foerstner``, ``skimage.feature.corner_fast``,
  ``skimage.feature.corner_subpix``, ``skimage.feature.corner_peaks``,
  ``skimage.feature.corner_orientations``, and
  ``skimage.feature._detect_octave``.
- In ``skimage.restoration._denoise``, the warning regarding
  ``rescale_sigma=None`` was removed.
- In ``skimage.restoration._cycle_spin``, the ``# doctest: +SKIP`` was removed.

Development process
-------------------

- Fix #3327: Add functionality for benchmark coverage (#3329)
- Release process notes have been improved. (#4228)
- ``pyproject.toml`` has been added to the sdist.
- Build and deploy dev/master documentation using GitHub Actions (#4852)
- Website now deploys itself (#4870)
- build doc on circle ci and link artifact (#4881)
- Benchmarks can now run on older scikit-image commits (#4891)
- Website analytics are tracked using plausible.io and can be visualized on
  https://plausible.io/scikit-image.org (#4893)
- Artifacts for the documentation build are now found in each pull request
  (#4881).
- Documentation source files can now be written in Markdown in addition to
  ReST, thanks to ``myst`` (#4863).
- update trove classifiers and tests for Python 3.9 + fix pytest config (#5052)
- fix Azure Pipelines, pytest config, and trove classifiers for Python 3.8 (#5054)
- Moved our testing from Travis to GitHub Actions (#5074)
- We now build our wheels on GitHub Actions on the main repo using
  cibuildwheel. Many thanks to the matplotlib and scikit-learn developers for
  paving the way for us! (#5080)
- Disable Travis-CI builds (#5099, #5111)
- Improvements to CircleCI build: no parallelization and caching) (#5097, #5119)

Other Pull Requests
-------------------

- Manage iradon input and output data type (#4298)
- random walker: Display a warning when the probability is outsite [0,1] for a given tol (#4631)
- MAINT: remove unused cython file (#4633)
- Forget legacy data dir (#4662)
- Setup longdesc markdown and switch to 0.18dev (#4663)
- Optional pooch dependency (#4666)
- Adding new default values to functions on doc/examples/segmentation/plot_ncut (#4676)
- Reintroduced convert with a strong deprecation warning (#4681)
- In release notes, better describe skimage's relationship to ecosystem (#4689)
- Perform some todo tasks for 0.18 (#4690)
- Perform todo tasks for 0.17! (#4691)
- suppressing warnings from gallery examples (#4692)
- release notes for 0.17.2 (#4702)
- Fix gallery example mentioning deprecated argument (#4706)
- Specify the encoding of files opened in the setup phase (#4713)
- Remove duplicate fused type definition (#4724)
- Blacklist cython version 0.29.18 (#4730)
- Fix CI failures related to conversion of np.floating to dtype (#4731)
- Fix Ci failures related to array ragged input numpy deprecation (#4735)
- Unwrap decorators before resolving link to source (sphinx.ext.linkcode) (#4740)
- Fix plotting error in j-invariant denoising tutorial (#4744)
- Highlight all source lines with HTML doc "source" links (sphinx.ext.linkcode) (#4746)
- Turn checklist boxes into bullet points inside the pull request template (#4747)
- Deprecate (min_distance < 1) and (footprint.size < 2) in peak_local_max (#4753)
- forbid dask 2.17.0 to fix CI (#4758)
- try to fix ci which is broken because of pyqt5 last version (#4788)
- Remove unused variable in j invariant docs (#4792)
- include all md files in manifest.in (#4793)
- Remove additional "::" to make plot directive work. (#4798)
- Use optipng to compress images/thumbnails in our gallery (#4800)
- Fix runtime warning in blob.py (#4803)
- Add TODO task for sphinx-gallery>=0.9.0 to remove enforced thumbnail_size (#4804)
- Change SSIM code example to use real MSE (#4807)
- Let biomed example load image data with Pooch. (#4809)
- Tweak threshold_otsu error checking - closes #4811 (#4812)
- Ensure assert messages from Cython rank filters are informative (#4815)
- Simplify equivalent_diameter function (#4819)
- DOC: update subpackage descriptions (#4825)
- style: be explicit when stacking arrays (#4826)
- MAINT: import Iterable from collections.abc (Python 3.9 compatibility) (#4834)
- Silence several warnings in the test suite (#4837)
- Silence a few RuntimeWarnings in the test suite (#4838)
- handle color string mapping correctly (#4840)
- DOC: Autoformat docstrings in ``io.*.py`` (#4845)
- Update min req for pillow due to CVE-2020-10379 and co. (#4861)
- DOC: First pass at format conversion, rst -> myst (#4863)
- Fixed typo in comment (#4867)
- Alternative wording for install guide PR #4750 (#4871)
- DOC: Clarify condition on unique vertices returned by marching cubes (#4872)
- Remove unmaintained wiki page link in contributor guidelines (#4873)
- new matomo config (#4879)
- Fix Incorrect documentation for skimage.util.img_as_int Issue (#4888)
- Minor edit for proper doc rendering (#4897)
- Changelog back-log (#4898)
- minor refactoring in phase_cross_correlation (#4901)
- Fix draw.circle/disk deprecation message, fixes #4884 (#4908)
- Add versionchanged tag for new opt param in measure.find_contours() (#4909)
- Declare build dependencies (#4920)
- Replace words with racial connotations (#4921)
- Fixes to apply_parallel for functions working with multichannel data (#4927)
- Improve description of h_maxima and h_minima functions (#4928) (#4929)
- CI: Skip doc build for PYTHONOPTIMIZE=2 (#4930)
- MAINT: Remove custom fused type in skimage/morphology/_max_tree.pyx (#4931)
- MAINT: remove numpydoc option, issue fixed in numpydoc 1.0 (#4932)
- modify development version string to allow use with NumpyVersion (#4947)
- CI: Add verbose option to avoid travis timeout for OSX install script  (#4956)
- Fix CI: ban sphinx-gallery 0.8.0 (#4960)
- Alias for data.chelsea: data.cat() (#4962)
- Fix typo. (#4963)
- CI: Use Travis wait improved to avoid timeout for OSX builds (#4965)
- Small enhancement in "Contour finding" example: Removed unused variable n (#4967)
- MAINT: remove unused imports (#4968)
- MAINT: Remove conditional import on networkx (#4970)
- forbid latest version of pyqt (#4973)
- Remove warnings/explicit settings on feature, restoration (#4974)
- Docstring improvements for label and regionprops_label (#4983)
- try to fix timeout problem with circleci (#4986)
- improve Euler number example (#4989)
- [website] Standardize Documentation index page. (#4990)
- Proofread INSTALL file. (#4991)
- Catch leftover typos in INSTALL file. (#4992)
- Let tifffile.imread handle additional keyword arguments (#4997)
- Update docstring for random_noise function (#5001)
- Update sphinx mapping for sklearn and numpy (#5003)
- Update docstring slic superpixels (#5014)
- Bump numpy versions to match scipy (kinda) (#5016)
- Fix usage of numpy.pad for old versions of numpy (#5017)
- [MRG] Update documentation to new data.camera() (#5018)
- bumped plotly requirement for docs (#5021)
- Fix IndexError when calling hough_line_peaks with too few angles (#5024)
- Code simplification after latest numpy bump (#5027)
- Fixes broken link to CODE_OF_CONDUCT.md (#5030)
- Specify whether core dev should merge right after second approving review. (#5040)
- Update pytest configuration to include ``test_`` functions (#5044)
- MAINT Build fix for pyodide (#5059)
- reduce OSX build time so that Travis is happy (#5067)
- DOC: document the normalized kernel in prewitt_h, prewitt_v (#5076)
- Some minor tweaks to CI (#5079)
- removed usage of numpy's private functions from util.arraycrop (#5081)
- peak_local_max: remove deprecated `indices` argument from examples (#5082)
- Replace np.bool, np.float, and np.int with bool, float, and int (#5103, #5108)
- change plausible script to track outbound links (#5115, #5123)
- Remove Python 3.6 support (#5117, #5125)
- Optimize ensure_spacing (#5062, #5135)


52 authors added to this release [alphabetical by first name or login]
----------------------------------------------------------------------

A warm thank you to all contributors who added to this release. A fraction of contributors were first-time contributors to open source and a much larger fraction first-time contributors to scikit-image. It's a great feeling for maintainers to welcome new contributors, and the diversity of scikit-image contributors is surely a big strength of the package.

- Abhishek Arya
- Abhishek Patil
- Alexandre de Siqueira
- Ben Nathanson
- Cameron Blocker
- Chris Roat
- Christoph Gohlke
- Clement Ng
- Corey Harris
- David McMahon
- David Mellert
- Devi Sandeep
- Egor Panfilov
- Emmanuelle Gouillart
- François Boulogne
- Genevieve Buckley
- Gregory R. Lee
- Harry Kwon
- iofall (cedarfall)
- Jan Funke
- Juan Nunez-Iglesias
- Julian Gilbey
- Julien Jerphanion
- kalpana
- kolibril13 (kolibril13)
- Kushaan Gupta
- Lars Grüter
- Marianne Corvellec
- Mark Harfouche
- Marvin Albert
- Matthias Bussonnier
- Max Frei
- Nathan
- neeraj3029 (neeraj3029)
- Nick
- notmatthancock (matt)
- OGordon100 (OGordon100)
- Owen Solberg
- Riadh Fezzani
- Robert Haase
- Roman Yurchak
- Ronak Sharma
- Ross Barnowski
- Ruby Werman
- ryanlu41 (ryanlu41)
- Sebastian Wallkötter
- Shyam Saladi
- Stefan van der Walt
- Terence Honles
- Volker Hilsenstein
- Wendy Mak
- Yogendra Sharma

41 reviewers added to this release [alphabetical by first name or login]
------------------------------------------------------------------------

- Abhishek Arya
- Abhishek Patil
- Alexandre de Siqueira
- Ben Nathanson
- Chris Roat
- Clement Ng
- Corey Harris
- Cris Luengo
- David Mellert
- Egor Panfilov
- Emmanuelle Gouillart
- François Boulogne
- Gregory R. Lee
- Harry Kwon
- Jan Funke
- Juan Nunez-Iglesias
- Julien Jerphanion
- kalpana
- Kushaan Gupta
- Lars Grüter
- Marianne Corvellec
- Mark Harfouche
- Marvin Albert
- neeraj3029
- Nick
- OGordon100
- Riadh Fezzani
- Robert Haase
- Ross Barnowski
- Ruby Werman
- ryanlu41
- Scott Trinkle
- Sebastian Wallkötter
- Stanley_Wang
- Stefan van der Walt
- Steven Brown
- Stuart Mumford
- Terence Honles
- Volker Hilsenstein
- Wendy Mak

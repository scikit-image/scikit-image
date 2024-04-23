scikit-image 0.23.2 (2024-04-20)
================================

We're happy to announce the release of scikit-image 0.23.2!

Bug Fixes
---------

- Make sure ``skimage.util.img_as_ubyte`` supports the edge case where ``dtype('uint64').type`` of the provided image is ``np.ulonglong`` instead of ``np.uint64`` (`#7392 <https://github.com/scikit-image/scikit-image/pull/7392>`_).

Documentation
-------------

- Add date to 0.23.1 release notes (`#7384 <https://github.com/scikit-image/scikit-image/pull/7384>`_).
- Fix docstring of ``connectivity`` parameter in ``skimage.segmentation.watershed`` (`#7360 <https://github.com/scikit-image/scikit-image/pull/7360>`_).

Infrastructure
--------------

- Ignore Sphinx warning about unpickable cache (`#7400 <https://github.com/scikit-image/scikit-image/pull/7400>`_).
- Simplify instructions on changelist in PR template (`#7401 <https://github.com/scikit-image/scikit-image/pull/7401>`_).

Maintenance
-----------

- Use ``numpy.inf`` instead of deprecated ``numpy.infty`` (`#7386 <https://github.com/scikit-image/scikit-image/pull/7386>`_).
- Update Ruff config (`#7387 <https://github.com/scikit-image/scikit-image/pull/7387>`_).
- Update matrix and names of Azure pipelines configuration (`#7390 <https://github.com/scikit-image/scikit-image/pull/7390>`_).
- Use upload- and download-artifact v4 (`#7389 <https://github.com/scikit-image/scikit-image/pull/7389>`_).
- Ignore arch specific cast warnings originating from ``astype`` in tests (`#7393 <https://github.com/scikit-image/scikit-image/pull/7393>`_).
- Update link to numpydoc example.py (`#7395 <https://github.com/scikit-image/scikit-image/pull/7395>`_).

Contributors
------------

4 authors added to this release (alphabetically):

- `@pitkajuh <https://github.com/pitkajuh>`_
- Jarrod Millman (`@jarrodmillman <https://github.com/jarrodmillman>`_)
- Lars Grüter (`@lagru <https://github.com/lagru>`_)
- Marianne Corvellec (`@mkcor <https://github.com/mkcor>`_)

4 reviewers added to this release (alphabetically):

- Egor Panfilov (`@soupault <https://github.com/soupault>`_)
- Jarrod Millman (`@jarrodmillman <https://github.com/jarrodmillman>`_)
- Lars Grüter (`@lagru <https://github.com/lagru>`_)
- Marianne Corvellec (`@mkcor <https://github.com/mkcor>`_)

_These lists are automatically generated, and may not be complete or may contain duplicates._

scikit-image 0.23.1 (2024-04-10)
================================

We're happy to announce the release of scikit-image 0.23.1!

.. note::

   Due to an issue with the CI system scikit-image 0.23.0 was never released.
   This release is identical to what 0.23.0 would have been other than the CI
   fix and the version number.

Highlights
----------

- Ensure ``skimage.morphology.closing`` and ``skimage.morphology.opening`` are extensive and anti-extensive, respectively, if the footprint is not mirror symmetric (`#6695 <https://github.com/scikit-image/scikit-image/pull/6695>`_).
- Add parameters ``mode`` and ``cval`` to ``erosion``, ``dilation``, ``opening``, ``closing``, ``white_tophat``, and ``black_tophat`` in ``skimage.morphology``. These new parameters determine how array borders are handled (`#6695 <https://github.com/scikit-image/scikit-image/pull/6695>`_).
- Add parameter ``mode`` to ``binary_erosion``, ``binary_dilation``, ``binary_opening`` and ``binary_closing`` in ``skimage.morphology``. These new parameters determine how array borders are handled (`#6695 <https://github.com/scikit-image/scikit-image/pull/6695>`_).
- Speedup ``skimage.util.map_array`` by parallelization with Cython's ``prange`` (`#7266 <https://github.com/scikit-image/scikit-image/pull/7266>`_).

New Features
------------

- Add new ``intensity_std`` property to ``skimage.measure.regionprops`` which computes the standard deviation of the intensity in a region (`#6712 <https://github.com/scikit-image/scikit-image/pull/6712>`_).
- Add parameters ``mode`` and ``cval`` to ``erosion``, ``dilation``, ``opening``, ``closing``, ``white_tophat``, and ``black_tophat`` in ``skimage.morphology``. These new parameters determine how array borders are handled (`#6695 <https://github.com/scikit-image/scikit-image/pull/6695>`_).
- Add parameter ``mode`` to ``binary_erosion``, ``binary_dilation``, ``binary_opening`` and ``binary_closing`` in ``skimage.morphology``. These new parameters determine how array borders are handled (`#6695 <https://github.com/scikit-image/scikit-image/pull/6695>`_).
- Add functions ``mirror_footprint`` and ``pad_footprint`` to ``skimage.morphology`` (`#6695 <https://github.com/scikit-image/scikit-image/pull/6695>`_).
- Add new parameter ``spacing`` to ``segmentation.expand_labels`` to support anisotropic images (`#7080 <https://github.com/scikit-image/scikit-image/pull/7080>`_).

API Changes
-----------

- Drop support for Python 3.9 (`#7217 <https://github.com/scikit-image/scikit-image/pull/7217>`_).
- Parameters ``shift_x`` and ``shift_y`` in ``skimage.morphology.erosion`` and ``skimage.morphology.dilation`` are deprecated. Use ``pad_footprint`` or modify the footprint manually instead (`#6695 <https://github.com/scikit-image/scikit-image/pull/6695>`_).
- Remove unexpected value scaling in ``skimage.morphology.skeletonize_3d`` for non-binary input images. ``skeletonize_3d`` now always returns a binary array like similar functions (`#7095 <https://github.com/scikit-image/scikit-image/pull/7095>`_).
- Deprecate function ``skimage.feature.plot_matches`` in favor of ``skimage.feature.plot_matched_features`` (`#7255 <https://github.com/scikit-image/scikit-image/pull/7255>`_).
- Deprecate ``skimage.morphology.skeletonize_3d`` in favor of just ``skimage.morphology.skeletonize`` (`#7094 <https://github.com/scikit-image/scikit-image/pull/7094>`_).
- Deprecate parameter ``output`` in ``skimage.filters.gaussian``; use ``out`` instead (`#7225 <https://github.com/scikit-image/scikit-image/pull/7225>`_).
- Change the default value of the parameters ``shift_x``, ``shift_y`` and ``shift_z`` from ``False`` to ``0`` in the ``skimage.filters.rank`` functions. This has not impact on the  results. Warn in case boolean shifts are provided from now on (`#7320 <https://github.com/scikit-image/scikit-image/pull/7320>`_).

Performance
-----------

- Add lazy loading to ``skimage.metrics`` module (`#7211 <https://github.com/scikit-image/scikit-image/pull/7211>`_).
- Speedup ``skimage.util.map_array`` by parallelization with Cython's ``prange`` (`#7266 <https://github.com/scikit-image/scikit-image/pull/7266>`_).

Bug Fixes
---------

- Add exception to avoid surprising result when image is too small for the given parameters in ``skimage.feature.hog`` (`#7153 <https://github.com/scikit-image/scikit-image/pull/7153>`_).
- Ensure ``skimage.morphology.closing`` and ``skimage.morphology.opening`` are extensive and anti-extensive, respectively, if the footprint is not mirror symmetric (`#6695 <https://github.com/scikit-image/scikit-image/pull/6695>`_).
- Avoid a TypeError in ``skimage.registration.phase_cross_correlation`` when the real-time shift cannot be determined (``disambiguate=True``). Display a warning instead (`#7259 <https://github.com/scikit-image/scikit-image/pull/7259>`_).
- Fix logic in ``skimage.graph.pixel_graph`` which raised a ``TypeError`` when the parameter ``edge_function`` was provided without a ``mask`` (`#7310 <https://github.com/scikit-image/scikit-image/pull/7310>`_).
- Ensure cache stays empty when ``cache=False`` is passed to ``skimage.measure.regionprops`` (`#7333 <https://github.com/scikit-image/scikit-image/pull/7333>`_).

Documentation
-------------

- Update instructions for updating dev environment (`#7160 <https://github.com/scikit-image/scikit-image/pull/7160>`_).
- Make titles in RAG gallery examples more explicit (`#7202 <https://github.com/scikit-image/scikit-image/pull/7202>`_).
- Add docstring to ``skimage.graph`` module (`#7192 <https://github.com/scikit-image/scikit-image/pull/7192>`_).
- Use consistent notation for array dimensions in the docstrings (`#3031 <https://github.com/scikit-image/scikit-image/pull/3031>`_).
- Specify default markers in watershed docstring (`#7154 <https://github.com/scikit-image/scikit-image/pull/7154>`_).
- Stop HTML documentation from intercepting left and right-arrow keys to improve keyboard accessibility (`#7226 <https://github.com/scikit-image/scikit-image/pull/7226>`_).
- Fix reference formatting for nitpicky sphinx (`#7228 <https://github.com/scikit-image/scikit-image/pull/7228>`_).
- Document how to deal with other array-likes such as  ``xarray.DataArray`` and ``pandas.DataFrame`` in the crash course on NumPy for images (`#7159 <https://github.com/scikit-image/scikit-image/pull/7159>`_).
- Fix broken function calls and syntax issues in user guide (`#7234 <https://github.com/scikit-image/scikit-image/pull/7234>`_).
- Use correct default mode in docstring of ``skimage.transform.swirl`` (`#7241 <https://github.com/scikit-image/scikit-image/pull/7241>`_).
- Add missing documentation about spacing parameter in ``moments_normalized`` (`#7248 <https://github.com/scikit-image/scikit-image/pull/7248>`_).
- Update docstring & example in the hough_ellipse transform (`#6893 <https://github.com/scikit-image/scikit-image/pull/6893>`_).
- Point binder tag/branch to commit corresponding to docs/release (`#7252 <https://github.com/scikit-image/scikit-image/pull/7252>`_).
- Add example to FundamentalMatrixTransform class (`#6863 <https://github.com/scikit-image/scikit-image/pull/6863>`_).
- Adds explanation of what the optional dependency on Matplotlib offers to the install instructions (`#7286 <https://github.com/scikit-image/scikit-image/pull/7286>`_).
- Use correct symbol θ for tightness in the docstring of  ``skimage.registration.optical_flow_tvl1`` (`#7314 <https://github.com/scikit-image/scikit-image/pull/7314>`_).
- The description of the parameter cval is modified in "int or float". cval is a numerical value not a string (`#7319 <https://github.com/scikit-image/scikit-image/pull/7319>`_).
- Remove obsolete instruction about documenting changes (`#7321 <https://github.com/scikit-image/scikit-image/pull/7321>`_).
- Added comment to clarify that dt corresponds to tau, i.e. the time step. Changed gray scale in grayscale in the entire registration module (`#7324 <https://github.com/scikit-image/scikit-image/pull/7324>`_).
- Create SECURITY.md (`#7230 <https://github.com/scikit-image/scikit-image/pull/7230>`_).
- Remove deprecated parameter ``coordinates`` from docstring  example of ``skimage.segmentation.active_contour`` (`#7329 <https://github.com/scikit-image/scikit-image/pull/7329>`_).
- Include dates in release note headings (`#7269 <https://github.com/scikit-image/scikit-image/pull/7269>`_).
- Update description of how to document pull requests for inclusion in the release notes (`#7267 <https://github.com/scikit-image/scikit-image/pull/7267>`_).
- Clarify description of ``data_range`` parameter in ``skimage.metrics.structural_similarity`` (`#7345 <https://github.com/scikit-image/scikit-image/pull/7345>`_).
- Use  object-oriented Matplotlib style in longer gallery examples and demonstrations (doc/examples/applications) (`#7346 <https://github.com/scikit-image/scikit-image/pull/7346>`_).
- In the gallery example on segmenting human cells (in mitosis), include the border when generating basin markers for watershed (`#7362 <https://github.com/scikit-image/scikit-image/pull/7362>`_).
- Add missing minus sign in docstring of ``skimage.transform.EuclideanTransform`` (`#7097 <https://github.com/scikit-image/scikit-image/pull/7097>`_).

Infrastructure
--------------

- Update wording on the stale bot to assume the core team dropped the ball (`#7196 <https://github.com/scikit-image/scikit-image/pull/7196>`_).
- Update Azure job name following the drop of Python 3.9 (`#7218 <https://github.com/scikit-image/scikit-image/pull/7218>`_).
- Schedule nightly wheel builds at uncommon time (`#7254 <https://github.com/scikit-image/scikit-image/pull/7254>`_).
- Build nightly wheels with nightly NumPy 2.0 (`#7251 <https://github.com/scikit-image/scikit-image/pull/7251>`_).
- Use pytest-doctestplus instead of classic pytest-doctest (`#7289 <https://github.com/scikit-image/scikit-image/pull/7289>`_).
- Update the scientific-python/upload-nightly-action to v0.5.0 for dependency stability and to take advantage of Anaconda Cloud upload bug fixes (`#7325 <https://github.com/scikit-image/scikit-image/pull/7325>`_).
- Add ``assert_stacklevel`` helper to check stacklevel of captured warnings (`#7294 <https://github.com/scikit-image/scikit-image/pull/7294>`_).
- Exclude ``pre-commit[bot]`` from changelist's contributor list (`#7358 <https://github.com/scikit-image/scikit-image/pull/7358>`_).

Maintenance
-----------

- Remove outdated & duplicate "preferred" field in ``version_switcher.json`` (`#7184 <https://github.com/scikit-image/scikit-image/pull/7184>`_).
- Upgrade to spin 0.7 (`#7168 <https://github.com/scikit-image/scikit-image/pull/7168>`_).
- Do not compare types, use isinstance (`#7186 <https://github.com/scikit-image/scikit-image/pull/7186>`_).
- [pre-commit.ci] pre-commit autoupdate (`#7181 <https://github.com/scikit-image/scikit-image/pull/7181>`_).
- Increase tolerance for moments test for 32 bit floats (`#7188 <https://github.com/scikit-image/scikit-image/pull/7188>`_).
- Temporarily pin Cython to <3.0.3 until CI is fixed (`#7189 <https://github.com/scikit-image/scikit-image/pull/7189>`_).
- Remove obsolete meson instructions (`#7193 <https://github.com/scikit-image/scikit-image/pull/7193>`_).
- Temporarily pin Cython to <3.0.3 until CI is fixed, take 2 (`#7201 <https://github.com/scikit-image/scikit-image/pull/7201>`_).
- Fix chocolatey (`#7200 <https://github.com/scikit-image/scikit-image/pull/7200>`_).
- Pin Pillow to <10.1.0 until incompatibility with imageio is fixed (`#7208 <https://github.com/scikit-image/scikit-image/pull/7208>`_).
- Use Black (`#7197 <https://github.com/scikit-image/scikit-image/pull/7197>`_).
- Apply black to ``_hog.py`` after previous merge lacking black (`#7215 <https://github.com/scikit-image/scikit-image/pull/7215>`_).
- Unpin Cython after release of Cython 3.0.4 (`#7214 <https://github.com/scikit-image/scikit-image/pull/7214>`_).
- [pre-commit.ci] pre-commit autoupdate (`#7236 <https://github.com/scikit-image/scikit-image/pull/7236>`_).
- Cleanup for Python 3.12 (`#7173 <https://github.com/scikit-image/scikit-image/pull/7173>`_).
- Make Python 3.12 default CI Python (`#7244 <https://github.com/scikit-image/scikit-image/pull/7244>`_).
- Add explicit ``noexcept`` to address Cython 3.0 warnings (`#7250 <https://github.com/scikit-image/scikit-image/pull/7250>`_).
- Update imageio to fix Pillow incompatibility (`#7245 <https://github.com/scikit-image/scikit-image/pull/7245>`_).
- Upgrade docker/setup-qemu-action to v3 (`#7134 <https://github.com/scikit-image/scikit-image/pull/7134>`_).
- Fix warningfilter for deprecation in SciPy 1.12.0rc1 (`#7275 <https://github.com/scikit-image/scikit-image/pull/7275>`_).
- Update to numpy>=1.23 and matplotlib>=3.6 according to SPEC 0 (`#7284 <https://github.com/scikit-image/scikit-image/pull/7284>`_).
- Add new ``deprecate_parameter`` helper (`#7256 <https://github.com/scikit-image/scikit-image/pull/7256>`_).
- Update meson and Cython (`#7283 <https://github.com/scikit-image/scikit-image/pull/7283>`_).
- Handle floating point warning for empty images in ``skimage.registration.phase_cross_correlation`` (`#7287 <https://github.com/scikit-image/scikit-image/pull/7287>`_).
- Update spin (0.8) (`#7285 <https://github.com/scikit-image/scikit-image/pull/7285>`_).
- Complete deprecations that were scheduled for our 0.23 release.  Remove now unused ``deprecate_kwarg`` and ``remove_arg``; they are  entirely succeeded by ``deprecate_parameter`` (`#7290 <https://github.com/scikit-image/scikit-image/pull/7290>`_).
- For security best practices, use the scientific-python/upload-nightly-action GitHub Action from known commit shas that correspond to tagged releases. These can be updated automatically via Dependabot (`#7306 <https://github.com/scikit-image/scikit-image/pull/7306>`_).
- Update pre-commits repos (`#7303 <https://github.com/scikit-image/scikit-image/pull/7303>`_).
- The test suite can now be run without ``numpydoc`` installed (`#7307 <https://github.com/scikit-image/scikit-image/pull/7307>`_).
- Deal with parallel write warning from Pydata theme (`#7311 <https://github.com/scikit-image/scikit-image/pull/7311>`_).
- Test nightly wheel build with NumPy 2.0 (`#7288 <https://github.com/scikit-image/scikit-image/pull/7288>`_).
- Make it clear that funcs in ``_optical_flow_utils`` are private (`#7328 <https://github.com/scikit-image/scikit-image/pull/7328>`_).
- Update dependencies (spec 0) (`#7335 <https://github.com/scikit-image/scikit-image/pull/7335>`_).
- Follow-up cleaning & fixes for compatibility with NumPy 1 & 2 (`#7326 <https://github.com/scikit-image/scikit-image/pull/7326>`_).
- Replace ignored teardown with autouse fixture in ``test_fits.py`` (`#7340 <https://github.com/scikit-image/scikit-image/pull/7340>`_).
- Address new copy semantics & broadcasting in ``np.solve`` in NumPy 2 (`#7341 <https://github.com/scikit-image/scikit-image/pull/7341>`_).
- Ignore table of execution times by Sphinx gallery (`#7327 <https://github.com/scikit-image/scikit-image/pull/7327>`_).
- Allow a very small floating point tolerance for pearson test (`#7356 <https://github.com/scikit-image/scikit-image/pull/7356>`_).
- Update numpydoc to version 1.7 (`#7355 <https://github.com/scikit-image/scikit-image/pull/7355>`_).
- [pre-commit.ci] pre-commit autoupdate (`#7365 <https://github.com/scikit-image/scikit-image/pull/7365>`_).
- Simplify warning filters in test suite (`#7349 <https://github.com/scikit-image/scikit-image/pull/7349>`_).
- Build against NumPy >=2.0.0rc1 (`#7367 <https://github.com/scikit-image/scikit-image/pull/7367>`_).
- Remove ``ensure_python_version`` function (`#7370 <https://github.com/scikit-image/scikit-image/pull/7370>`_).
- Update GitHub actions to ``setup-python@v5``, ``cache@v4``, ``upload-artifact@v4``,  and ``download-artifact@v4`` (`#7368 <https://github.com/scikit-image/scikit-image/pull/7368>`_).
- Update lazyloader to v0.4 (`#7373 <https://github.com/scikit-image/scikit-image/pull/7373>`_).

Contributors
------------

29 authors added to this release (alphabetically):

- `@GParolini <https://github.com/GParolini>`_
- `@tokiAi <https://github.com/tokiAi>`_
- Adrien Foucart (`@adfoucart <https://github.com/adfoucart>`_)
- Anam Fatima (`@anamfatima1304 <https://github.com/anamfatima1304>`_)
- Ananya Srivastava (`@ana42742 <https://github.com/ana42742>`_)
- Ben Harvie (`@benharvie <https://github.com/benharvie>`_)
- Christian Clauss (`@cclauss <https://github.com/cclauss>`_)
- Cris Luengo (`@crisluengo <https://github.com/crisluengo>`_)
- Egor Panfilov (`@soupault <https://github.com/soupault>`_)
- Grzegorz Bokota (`@Czaki <https://github.com/Czaki>`_)
- Jan Lebert (`@sitic <https://github.com/sitic>`_)
- Jarrod Millman (`@jarrodmillman <https://github.com/jarrodmillman>`_)
- Jeremy Farrell (`@farrjere <https://github.com/farrjere>`_)
- Juan Nunez-Iglesias (`@jni <https://github.com/jni>`_)
- Lars Grüter (`@lagru <https://github.com/lagru>`_)
- Mao Nishino (`@mao1756 <https://github.com/mao1756>`_)
- Marianne Corvellec (`@mkcor <https://github.com/mkcor>`_)
- Mark Harfouche (`@hmaarrfk <https://github.com/hmaarrfk>`_)
- Matthew Feickert (`@matthewfeickert <https://github.com/matthewfeickert>`_)
- Matthew Vine (`@MattTheCuber <https://github.com/MattTheCuber>`_)
- Maxime Corbé (`@Maxime-corbe <https://github.com/Maxime-corbe>`_)
- Michał Górny (`@mgorny <https://github.com/mgorny>`_)
- Neil Shephard (`@ns-rse <https://github.com/ns-rse>`_)
- Ole Streicher (`@olebole <https://github.com/olebole>`_)
- Peter Suter (`@petsuter <https://github.com/petsuter>`_)
- Robert Haase (`@haesleinhuepf <https://github.com/haesleinhuepf>`_)
- Sean McKinney (`@jouyun <https://github.com/jouyun>`_)
- Stefan van der Walt (`@stefanv <https://github.com/stefanv>`_)
- vfdev (`@vfdev-5 <https://github.com/vfdev-5>`_)

21 reviewers added to this release (alphabetically):

- `@GParolini <https://github.com/GParolini>`_
- Adrien Foucart (`@adfoucart <https://github.com/adfoucart>`_)
- Anam Fatima (`@anamfatima1304 <https://github.com/anamfatima1304>`_)
- Ben Harvie (`@benharvie <https://github.com/benharvie>`_)
- Christian Clauss (`@cclauss <https://github.com/cclauss>`_)
- Cris Luengo (`@crisluengo <https://github.com/crisluengo>`_)
- Egor Panfilov (`@soupault <https://github.com/soupault>`_)
- Grzegorz Bokota (`@Czaki <https://github.com/Czaki>`_)
- Jarrod Millman (`@jarrodmillman <https://github.com/jarrodmillman>`_)
- Jeremy Farrell (`@farrjere <https://github.com/farrjere>`_)
- Juan Nunez-Iglesias (`@jni <https://github.com/jni>`_)
- Lars Grüter (`@lagru <https://github.com/lagru>`_)
- M Bussonnier (`@Carreau <https://github.com/Carreau>`_)
- Mao Nishino (`@mao1756 <https://github.com/mao1756>`_)
- Marianne Corvellec (`@mkcor <https://github.com/mkcor>`_)
- Mark Harfouche (`@hmaarrfk <https://github.com/hmaarrfk>`_)
- Maxime Corbé (`@Maxime-corbe <https://github.com/Maxime-corbe>`_)
- P. L. Lim (`@pllim <https://github.com/pllim>`_)
- Peter Suter (`@petsuter <https://github.com/petsuter>`_)
- Sebastian Berg (`@seberg <https://github.com/seberg>`_)
- Stefan van der Walt (`@stefanv <https://github.com/stefanv>`_)

_These lists are automatically generated, and may not be complete or may contain duplicates._

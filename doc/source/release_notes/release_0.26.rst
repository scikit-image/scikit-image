scikit-image 0.26.0  (2025-12-20)
=================================

We're happy to announce the release of scikit-image 0.26.0!

New Features
------------

- Add new parameter ``max_step_cost`` to ``skimage.graph.MCP.find_costs`` which allows limiting the maximal stepping cost between points (`#7625 <https://github.com/scikit-image/scikit-image/pull/7625>`_).
- In ``skimage.transform``, add the ``identity`` class constructor to all geometric transforms. For example, you can now use  ``skimage.transform.PolynomialTransform(dimensionality=2)`` (`#7754 <https://github.com/scikit-image/scikit-image/pull/7754>`_).
- Add new property ``intensity_median`` to ``skimage.measure.regionprops`` (`#7745 <https://github.com/scikit-image/scikit-image/pull/7745>`_).
- ``binary_blobs`` now supports a ``mode`` parameter for the Gaussian filter, allowing periodic boundary conditions with ``mode="wrap"`` (`#7909 <https://github.com/scikit-image/scikit-image/pull/7909>`_).

API Changes
-----------

- In ``skimage.morphology``, deprecate ``binary_erosion``, ``binary_dilation``, ``binary_opening``,  and ``binary_closing`` in favor of ``erosion``, ``dilation``, ``opening``, and ``closing``  respectively. The binary versions weren't actually significantly faster than their non-binary counterparts and sometimes significantly slower. In the future, we might add optimizations internally to the remaining (general, non-binary) functions for  when they're used with binary inputs (`#7665 <https://github.com/scikit-image/scikit-image/pull/7665>`_).
- Deprecate parameter ``max_cost`` in ``skimage.graph.MCP.find_costs``  which previously did nothing. Use the new parameter ``max_step_cost`` instead (`#7625 <https://github.com/scikit-image/scikit-image/pull/7625>`_).
- Deprecate parameter ``max_cumulative_cost`` in ``skimage.graph.MCP.find_costs``  which did nothing (`#7625 <https://github.com/scikit-image/scikit-image/pull/7625>`_).
- In ``skimage.morphology.remove_small_holes``, deprecate the ``area_threshold`` parameter in favor of the new ``max_size`` parameter to make API and behavior clearer. This new threshold removes holes smaller than **or equal to** its value, while the previous parameter only removed smaller ones (`#7739 <https://github.com/scikit-image/scikit-image/pull/7739>`_).
- In ``skimage.morphology.remove_small_objects``, deprecate the ``min_size`` parameter in favor of the new ``max_size`` parameter to make API and behavior clearer. This new threshold removes objects smaller than **or equal to** its value, while the previous parameter only removed smaller ones (`#7739 <https://github.com/scikit-image/scikit-image/pull/7739>`_).
- In ``skimage.transform``, deprecate the use of scalar ``scale``, with ``dimensionality=3``  where this can be passed to a geometric transform contructor. This allows us to generalize the use of the constructors to the case where the parameters must specify the dimensionality, unless you mean to construct an identity transform (`#7754 <https://github.com/scikit-image/scikit-image/pull/7754>`_).
- In ``skimage.transform``, turn all input parameters to transform constructors keyword-only (other than ``matrix``). This avoids confusion due to the positional parameter order being different from the order by which they are applied in ``AffineTransform`` (`#7754 <https://github.com/scikit-image/scikit-image/pull/7754>`_).
- Deprecate parameter ``num_threads`` in ``skimage.restoration.rolling_ball``;  use ``workers`` instead (`#7302 <https://github.com/scikit-image/scikit-image/pull/7302>`_).
- Deprecate parameter ``num_workers`` in ``skimage.restoration.cycle_spin``;  use ``workers`` instead (`#7302 <https://github.com/scikit-image/scikit-image/pull/7302>`_).
- Officially deprecate old properties in ``skimage.measure.regionprops`` and related functions. While we removed the documentation for these some time ago, they where still accessible as keys (via ``__get_item__``) or attributes. Going forward, using deprecated keys or attributes, will emit an appropriate warning (`#7778 <https://github.com/scikit-image/scikit-image/pull/7778>`_).
- In ``skimage.measure``, add a new class method and constructor ``from_estimate`` for  ``LineModelND``, ``CircleModel``, and ``EllipseModel``. This replaces the old API—the now deprecated ``estimate`` method—which required initalizing a model with undefined parameters before calling ``estimate`` (`#7771 <https://github.com/scikit-image/scikit-image/pull/7771>`_).
- In ``skimage.transform``, add a new class method and constructor ``from_estimate`` for ``AffineTransform``, ``EssentialMatrixTransform``, ``EuclideanTransform``, ``FundamentalMatrixTransform``, ``PiecewiseAffineTransform``, ``PolynomialTransform``,  ``ProjectiveTransform``, ``SimilarityTransform``, and ``ThinPlateSplineTransform``. This replaces the old API—the now deprecated ``estimate`` method—which required initializing an undefined transform before calling ``estimate`` (`#7771 <https://github.com/scikit-image/scikit-image/pull/7771>`_).
- Deprecate ``skimage.measure.fit.BaseModel``; after we expire the other ``*Model*`` deprecations, there is no work for an ancestor class to do (`#7789 <https://github.com/scikit-image/scikit-image/pull/7789>`_).
- In ``skimage.measure``, deprecate ``.params`` attributes of the models ``CircleModel``, ``EllipseModel``, and ``LineModelND``.  Instead set model-specific attributes:  ``origin, direction`` for ``LineModelND``; ``center, radius`` for ``CircleModel``, ``center, ax_lens, theta`` for ``EllipseModel`` (`#7789 <https://github.com/scikit-image/scikit-image/pull/7789>`_).
- In ``skimage.measure``, deprecate use of model constructor calls without arguments leaving an uninitialized instance (for example ``CircleModel()``). This applies to ``CircleModel``, ``EllipseModel``, and ``LineModelND``. Instead prefer input arguments to define instances (for example ``CircleModel(center, radius)``). This follows on from prior deprecation of the ``estimate`` method, which had implied the need for the no-argument constructor, of form ``cm = CircleMoldel(); cm.estimate(data)`` (`#7789 <https://github.com/scikit-image/scikit-image/pull/7789>`_).
- In ``skimage.measure``, deprecate use of ``params`` arguments to ``predict*`` calls of  model objects. This applies to ``CircleModel``, ``EllipseModel``, and ``LineModelND``. We now ask instead that the user provide initialization equivalent to the ``params`` content in the class construction. For example, prefer  ``cm = CircleModel((2, 3), 4); x = cm.predict_x(t)`` to ``cm = CircleMoldel(); x = cm.predict_x(t, params=(2, 3, 4))``) (`#7789 <https://github.com/scikit-image/scikit-image/pull/7789>`_).

Enhancements
------------

- Raise a ``ValueError`` instead of a ``TypeError`` in ``CircleModel``, ``EllipseModel``, and ``LineModelND`` in ``skimage.measure``. This applies when failing  to pass a value for ``params`` (or passing ``params=None``) to ``predict`` methods of an uninitialized transform (`#7789 <https://github.com/scikit-image/scikit-image/pull/7789>`_).
- In ``skimage.measure``, the ``RegionProperties`` class that is returned by ``regionprops``, now has a formatted string representation (``__repr__``). This representation includes the label of the region and its bounding box (`#7887 <https://github.com/scikit-image/scikit-image/pull/7887>`_).

Performance
-----------

- Use greedy einsum optimization in ``skimage.measure.moments_central`` (`#7947 <https://github.com/scikit-image/scikit-image/pull/7947>`_).
- Add lazy loading to skimage.segmentation (`#7035 <https://github.com/scikit-image/scikit-image/pull/7035>`_).

Bug Fixes
---------

- Make deconvolution example scientifically sensible (`#7589 <https://github.com/scikit-image/scikit-image/pull/7589>`_).
- In ``skimage.filters.sobel/scharr/prewitt/farid``, when ``mode="constant"`` is used ensure that ``cval`` has an effect. It didn't previously (`#7826 <https://github.com/scikit-image/scikit-image/pull/7826>`_).
- Ensure ``skimage.graph.cut_normalized`` is deterministic when seeded with the  ``rng`` parameter and when SciPy 1.17.0.dev0 or newer is installed. With earlier SciPy versions the internally used function ``scipy.linalg.eigsh`` is not deterministic and can lead to different results (`#7912 <https://github.com/scikit-image/scikit-image/pull/7912>`_).
- Avoid potential integer overflow in ``skimage.morphology.reconstruction`` (`#7938 <https://github.com/scikit-image/scikit-image/pull/7938>`_).
- Handle negative axis lengths due to numerical errors in ``axis_major_length`` and  ``axis_minor_length`` properties of ``skimage.measure.regionprops`` (`#7916 <https://github.com/scikit-image/scikit-image/pull/7916>`_).
- In ``skimage.util.random_noise``, ensure that ``clip`` argument is handled consistently for various modes (`#7924 <https://github.com/scikit-image/scikit-image/pull/7924>`_).
- Apparent fix for Hough transform stray j (`#7974 <https://github.com/scikit-image/scikit-image/pull/7974>`_).

Documentation
-------------

- Reflect deprecation of I/O plugin infrastructure in user guide (`#7710 <https://github.com/scikit-image/scikit-image/pull/7710>`_).
- Display code blocks with a caption and extra indentation (`#7706 <https://github.com/scikit-image/scikit-image/pull/7706>`_).
- Edit gallery examples on segmentation to comply with matplotlib's object-oriented style (`#7531 <https://github.com/scikit-image/scikit-image/pull/7531>`_).
- Add link to 2.0 TODO (`#7716 <https://github.com/scikit-image/scikit-image/pull/7716>`_).
- Denote changes in Frangi filter explicitly (`#7721 <https://github.com/scikit-image/scikit-image/pull/7721>`_).
- Add SPEC badges (`#7737 <https://github.com/scikit-image/scikit-image/pull/7737>`_).
- Fix math equations in docstrings for manders_coloc_coeff and normalized_mutual_information (`#7517 <https://github.com/scikit-image/scikit-image/pull/7517>`_).
- Use the import convention ``import skimage as ski`` in the docstrings of ``ski.measure.euler_number``, ``ski.measure.perimeter`` and ``ski.measure.perimeter_crofton`` (`#7741 <https://github.com/scikit-image/scikit-image/pull/7741>`_).
- Fix the definition of entropy in the docstring of  ``skimage.metrics.normalized_mutual_information`` (`#7750 <https://github.com/scikit-image/scikit-image/pull/7750>`_).
- DOC: document use of environment.yml in installation guide (`#7760 <https://github.com/scikit-image/scikit-image/pull/7760>`_).
- Further document use of regionprops function and fix terms (`#7518 <https://github.com/scikit-image/scikit-image/pull/7518>`_).
- Display actual threshold value in Global Otsu plot (`#7780 <https://github.com/scikit-image/scikit-image/pull/7780>`_).
- DOC: Include missing gain parameter in adjust_gamma equation (`#7763 <https://github.com/scikit-image/scikit-image/pull/7763>`_).
- Document output dtype for transform.resize (`#7792 <https://github.com/scikit-image/scikit-image/pull/7792>`_).
- Use consistent wording in property description (`#7804 <https://github.com/scikit-image/scikit-image/pull/7804>`_).
- Draft migration guide for skimage2 (`#7785 <https://github.com/scikit-image/scikit-image/pull/7785>`_).
- Update import convention in certain gallery examples (`#7764 <https://github.com/scikit-image/scikit-image/pull/7764>`_).
- Document some parameters (`#7753 <https://github.com/scikit-image/scikit-image/pull/7753>`_).
- Make call to ``skimage.measure.ransac`` in the gallery example  "Assemble images with simple image stitching" deterministic. This avoids random non-deterministic failures (`#7851 <https://github.com/scikit-image/scikit-image/pull/7851>`_).
- Improve docstring for Wiener restoration function (`#7523 <https://github.com/scikit-image/scikit-image/pull/7523>`_).
- Describe custom warning-strategy in migration guide (`#7857 <https://github.com/scikit-image/scikit-image/pull/7857>`_).
- Normalize spelling of normalize (`#7865 <https://github.com/scikit-image/scikit-image/pull/7865>`_).
- Restore fast page navigation with in page anchors (`#7899 <https://github.com/scikit-image/scikit-image/pull/7899>`_).
- In ``skimage.feature``, clarify the description of the parameter ``num_sigma`` in ``blob_log`` and ``blob_doh`` (`#7774 <https://github.com/scikit-image/scikit-image/pull/7774>`_).
- Use correct CSS selector to override scroll-behavior (`#7928 <https://github.com/scikit-image/scikit-image/pull/7928>`_).
- Mention image readers other than skimage.io in getting_started docs (`#7929 <https://github.com/scikit-image/scikit-image/pull/7929>`_).
- doc: replace git checkout with git switch (`#7910 <https://github.com/scikit-image/scikit-image/pull/7910>`_).
- Avoid doctest error for -0 vs 0 (`#7950 <https://github.com/scikit-image/scikit-image/pull/7950>`_).
- Update contributing guide to de-emphasize rebasing (`#7953 <https://github.com/scikit-image/scikit-image/pull/7953>`_).
- Update git commands in contributing guide (`#7956 <https://github.com/scikit-image/scikit-image/pull/7956>`_).
- Add Linux Foundation Health Score badge to README (`#7907 <https://github.com/scikit-image/scikit-image/pull/7907>`_).
- Remove contributor docs section on pushing to another contributor's branch (`#7957 <https://github.com/scikit-image/scikit-image/pull/7957>`_).
- Clarify non-native support for masked array in the documentation (`#7968 <https://github.com/scikit-image/scikit-image/pull/7968>`_).
- Designate 0.26.0rc2 release (`#7987 <https://github.com/scikit-image/scikit-image/pull/7987>`_).
- Clarify RELEASE.txt after v0.26.0rc2 (`#7990 <https://github.com/scikit-image/scikit-image/pull/7990>`_).

Infrastructure
--------------

- Add experimental infrastructure for dispatching to a backend (`#7520 <https://github.com/scikit-image/scikit-image/pull/7520>`_).
- Build conda environment.yml from pyproject.toml (`#7758 <https://github.com/scikit-image/scikit-image/pull/7758>`_).
- Report failures on main via issue (`#7752 <https://github.com/scikit-image/scikit-image/pull/7752>`_).
- Make doctest-plus work with spin (`#7786 <https://github.com/scikit-image/scikit-image/pull/7786>`_).
- Temporarily pin to ``pyodide-build==0.30.0``, and ensure that the correct xbuildenvs are used (`#7788 <https://github.com/scikit-image/scikit-image/pull/7788>`_).
- Use ``cibuildwheel`` to build WASM/Pyodide wheels for ``scikit-image``, push nightlies to Anaconda.org (`#7440 <https://github.com/scikit-image/scikit-image/pull/7440>`_).
- CI: Update pypa/gh-action-pypi-publish to v1.12.4 for attestations on PyPI (`#7793 <https://github.com/scikit-image/scikit-image/pull/7793>`_).
- Do not report failure in wheels sub-recipe (`#7806 <https://github.com/scikit-image/scikit-image/pull/7806>`_).
- Temporary fix for Visual Studio & Clang incompatibility in Windows image (`#7835 <https://github.com/scikit-image/scikit-image/pull/7835>`_).
- Setup stub creation with docstub in CI (`#7802 <https://github.com/scikit-image/scikit-image/pull/7802>`_).
- Capture docstub output, but retain error (`#7852 <https://github.com/scikit-image/scikit-image/pull/7852>`_).
- Add dynamic date / git hash to dev version (`#7870 <https://github.com/scikit-image/scikit-image/pull/7870>`_).
- Add dependabot configuration (`#7882 <https://github.com/scikit-image/scikit-image/pull/7882>`_).
- Use pytest's summary over pytest-pretty's table-based one (`#7905 <https://github.com/scikit-image/scikit-image/pull/7905>`_).
- Bump the actions group across 1 directory with 14 updates (`#7895 <https://github.com/scikit-image/scikit-image/pull/7895>`_).
- Add 14 day cooldown for dependabot (`#7915 <https://github.com/scikit-image/scikit-image/pull/7915>`_).
- Skip Cython 3.2.* and address other issues in wheel building CI (`#7901 <https://github.com/scikit-image/scikit-image/pull/7901>`_).
- Simplify build wheel configuration by using of ``pyproject.toml`` configuration (`#7877 <https://github.com/scikit-image/scikit-image/pull/7877>`_).
- Report failures on nightly wheel build, test, or upload (`#7807 <https://github.com/scikit-image/scikit-image/pull/7807>`_).
- Require pytest >=8.3 which hides tracebacks for xfailures (`#7937 <https://github.com/scikit-image/scikit-image/pull/7937>`_).
- Note how to deal with automatic CI failure notifications / issues (`#7940 <https://github.com/scikit-image/scikit-image/pull/7940>`_).
- CI: Add support for building wheels for Windows on ARM (`#7847 <https://github.com/scikit-image/scikit-image/pull/7847>`_).
- Avoid building on macos-13 (`#7949 <https://github.com/scikit-image/scikit-image/pull/7949>`_).
- Revert "Refactor names in Pyodide workflow (#7959)" (`#7963 <https://github.com/scikit-image/scikit-image/pull/7963>`_).
- Avoid uploading unsupported wasm wheels to PyPI (`#7969 <https://github.com/scikit-image/scikit-image/pull/7969>`_).
- Test on macOS intel in CI again (`#7965 <https://github.com/scikit-image/scikit-image/pull/7965>`_).
- Wheels: add option to exclude v2 namespace (`#7958 <https://github.com/scikit-image/scikit-image/pull/7958>`_).
- Refactor Pyodide workflow with matrix (`#7962 <https://github.com/scikit-image/scikit-image/pull/7962>`_).
- Build wheels on Windows & Python 3.14t (`#7978 <https://github.com/scikit-image/scikit-image/pull/7978>`_).

Maintenance
-----------

- Bump to Pyodide 0.27.2 stable and install available optional dependencies for WASM tests (`#7646 <https://github.com/scikit-image/scikit-image/pull/7646>`_).
- Drop Python 3.10 support (`#7673 <https://github.com/scikit-image/scikit-image/pull/7673>`_).
- Remove outdated TODO (`#7713 <https://github.com/scikit-image/scikit-image/pull/7713>`_).
- Remove deprecated shift parameters (`#7714 <https://github.com/scikit-image/scikit-image/pull/7714>`_).
- Bump to Pyodide 0.27.3 (`#7712 <https://github.com/scikit-image/scikit-image/pull/7712>`_).
- Update build (`#7715 <https://github.com/scikit-image/scikit-image/pull/7715>`_).
- Remove deprecated image2 param (`#7719 <https://github.com/scikit-image/scikit-image/pull/7719>`_).
- Update MacOS min versions (`#7720 <https://github.com/scikit-image/scikit-image/pull/7720>`_).
- Cut down runtime of rolling-ball gallery example (`#7705 <https://github.com/scikit-image/scikit-image/pull/7705>`_).
- Pin JasonEtco/create-an-issue action to SHA for v2.9.2 (`#7787 <https://github.com/scikit-image/scikit-image/pull/7787>`_).
- Remove unused & obsolete ``legacy_datasets``, ``legacy_registry`` vars (`#7677 <https://github.com/scikit-image/scikit-image/pull/7677>`_).
- Address deprecations in Pillow 11.3 (`#7828 <https://github.com/scikit-image/scikit-image/pull/7828>`_).
- Only report failure on main branch once (`#7839 <https://github.com/scikit-image/scikit-image/pull/7839>`_).
- Remove superfluous ``mask`` argument in ``_generic_edge_filter`` (`#7827 <https://github.com/scikit-image/scikit-image/pull/7827>`_).
- In ``skimage.transform.FundamentalMatrixTransform``, refactor scaling calculation to make algorithm clearer, and allow original Hartley algorithm if preferred (`#7767 <https://github.com/scikit-image/scikit-image/pull/7767>`_).
- Skip doctest with random component (`#7854 <https://github.com/scikit-image/scikit-image/pull/7854>`_).
- Remove MANIFEST.in, that is no longer needed with Meson (`#7855 <https://github.com/scikit-image/scikit-image/pull/7855>`_).
- Fix simple errors reported by docstub (I) (`#7853 <https://github.com/scikit-image/scikit-image/pull/7853>`_).
- Add package version to skimage2 (`#7871 <https://github.com/scikit-image/scikit-image/pull/7871>`_).
- Use data files from GitLab repo (`#7875 <https://github.com/scikit-image/scikit-image/pull/7875>`_).
- Fix test to not fetch data already presented in test directory, fix tests on Python 3.14 and macos arm (`#7881 <https://github.com/scikit-image/scikit-image/pull/7881>`_).
- Update TODO to check test precision on macOS ARM (`#7884 <https://github.com/scikit-image/scikit-image/pull/7884>`_).
- Add information about time of execution of test on each file (`#7874 <https://github.com/scikit-image/scikit-image/pull/7874>`_).
- Remove np.testing from skimage._shared.utils (`#7891 <https://github.com/scikit-image/scikit-image/pull/7891>`_).
- MAINT: some dependency version consistency cleanups (`#7894 <https://github.com/scikit-image/scikit-image/pull/7894>`_).
- Add forgotten pytest-pretty to Pyodide wheel recipe too (`#7898 <https://github.com/scikit-image/scikit-image/pull/7898>`_).
- Address ResourceWarning in ``_built_utils/version.py`` (`#7904 <https://github.com/scikit-image/scikit-image/pull/7904>`_).
- Mark ``test_rag.py::test_reproducibility`` as flaky for current versions of SciPy (< 1.17.0.dev0) (`#7912 <https://github.com/scikit-image/scikit-image/pull/7912>`_).
- Use ``divmod`` in ``montage`` index computation (`#7914 <https://github.com/scikit-image/scikit-image/pull/7914>`_).
- Test fixes and improvmement from cibuldwheel PR (`#7922 <https://github.com/scikit-image/scikit-image/pull/7922>`_).
- Allow Cython 3.2.0 again (`#7927 <https://github.com/scikit-image/scikit-image/pull/7927>`_).
- Move away from numpy.fix in favor of numpy.trunc (`#7933 <https://github.com/scikit-image/scikit-image/pull/7933>`_).
- Bump the Pyodide version for testing to the latest available (0.29) (`#7931 <https://github.com/scikit-image/scikit-image/pull/7931>`_).
- Fix  ResourceWarning in CI (`#7930 <https://github.com/scikit-image/scikit-image/pull/7930>`_).
- Switch back to using Cython wheels from PyPI (`#7932 <https://github.com/scikit-image/scikit-image/pull/7932>`_).
- moments_hu doctest should ignore tiny differences (`#7944 <https://github.com/scikit-image/scikit-image/pull/7944>`_).
- Relax constraints of regionprops multichannel test on MacOS with NumPy & "Accelerate" (`#7942 <https://github.com/scikit-image/scikit-image/pull/7942>`_).
- Refactor names in Pyodide workflow (`#7959 <https://github.com/scikit-image/scikit-image/pull/7959>`_).
- Use __doctest_requires__ instead of inline importorskip (`#7966 <https://github.com/scikit-image/scikit-image/pull/7966>`_).
- Mark ``test_wrap_around`` as xfail on macOS until 2026-02-01 (`#7985 <https://github.com/scikit-image/scikit-image/pull/7985>`_).

Contributors
------------

40 authors added to this release (alphabetically):

- `@dependabot[bot] <https://github.com/apps/dependabot>`_
- `@EdytaRz <https://github.com/EdytaRz>`_
- `@jakirkham <https://github.com/jakirkham>`_
- `@jdarena66 <https://github.com/jdarena66>`_
- `@jmtayloruk <https://github.com/jmtayloruk>`_
- `@michaelbratsch <https://github.com/michaelbratsch>`_
- Aditi Juneja (`@Schefflera-Arboricola <https://github.com/Schefflera-Arboricola>`_)
- Agriya Khetarpal (`@agriyakhetarpal <https://github.com/agriyakhetarpal>`_)
- Alex Louk (`@AlexLouk <https://github.com/AlexLouk>`_)
- Ananya Srivastava (`@ana42742 <https://github.com/ana42742>`_)
- Brigitta Sipőcz (`@bsipocz <https://github.com/bsipocz>`_)
- Emmanuel Ferdman (`@emmanuel-ferdman <https://github.com/emmanuel-ferdman>`_)
- Eoghan O'Connell (`@PinkShnack <https://github.com/PinkShnack>`_)
- Giuditta Parolini (`@GParolini <https://github.com/GParolini>`_)
- Grzegorz Bokota (`@Czaki <https://github.com/Czaki>`_)
- Jamal Mustafa (`@jimustafa <https://github.com/jimustafa>`_)
- Jan Eglinger (`@imagejan <https://github.com/imagejan>`_)
- Jarrod Millman (`@jarrodmillman <https://github.com/jarrodmillman>`_)
- Jeremy Muhlich (`@jmuhlich <https://github.com/jmuhlich>`_)
- Jonathan Reimer (`@jonathimer <https://github.com/jonathimer>`_)
- Jordão Bragantini (`@JoOkuma <https://github.com/JoOkuma>`_)
- Juan Nunez-Iglesias (`@jni <https://github.com/jni>`_)
- Kevin (`@apetizerr <https://github.com/apetizerr>`_)
- Kimberly Meechan (`@K-Meech <https://github.com/K-Meech>`_)
- Larry Bradley (`@larrybradley <https://github.com/larrybradley>`_)
- Lars Grüter (`@lagru <https://github.com/lagru>`_)
- Marianne Corvellec (`@mkcor <https://github.com/mkcor>`_)
- Mark Harfouche (`@hmaarrfk <https://github.com/hmaarrfk>`_)
- Matt Haberland (`@mdhaber <https://github.com/mdhaber>`_)
- Matthew Brett (`@matthew-brett <https://github.com/matthew-brett>`_)
- Matthew Feickert (`@matthewfeickert <https://github.com/matthewfeickert>`_)
- MS-GITS (`@Greenie0701 <https://github.com/Greenie0701>`_)
- Paweł Rzońca (`@clacrow <https://github.com/clacrow>`_)
- Sebastian Berg (`@seberg <https://github.com/seberg>`_)
- Sigurd Vargdal (`@Tensorboy2 <https://github.com/Tensorboy2>`_)
- Stefan van der Walt (`@stefanv <https://github.com/stefanv>`_)
- Tim Head (`@betatim <https://github.com/betatim>`_)
- Valentin Valls (`@vallsv <https://github.com/vallsv>`_)
- Veit Heller (`@hellerve <https://github.com/hellerve>`_)
- Vicent Caselles-Ballester (`@vcasellesb <https://github.com/vcasellesb>`_)

25 reviewers added to this release (alphabetically):

- `@jakirkham <https://github.com/jakirkham>`_
- `@jmtayloruk <https://github.com/jmtayloruk>`_
- `@michaelbratsch <https://github.com/michaelbratsch>`_
- Aditi Juneja (`@Schefflera-Arboricola <https://github.com/Schefflera-Arboricola>`_)
- Agriya Khetarpal (`@agriyakhetarpal <https://github.com/agriyakhetarpal>`_)
- Alex Louk (`@AlexLouk <https://github.com/AlexLouk>`_)
- Brigitta Sipőcz (`@bsipocz <https://github.com/bsipocz>`_)
- Gregory Lee (`@grlee77 <https://github.com/grlee77>`_)
- Grzegorz Bokota (`@Czaki <https://github.com/Czaki>`_)
- Jan Eglinger (`@imagejan <https://github.com/imagejan>`_)
- Jarrod Millman (`@jarrodmillman <https://github.com/jarrodmillman>`_)
- Juan Nunez-Iglesias (`@jni <https://github.com/jni>`_)
- Larry Bradley (`@larrybradley <https://github.com/larrybradley>`_)
- Lars Grüter (`@lagru <https://github.com/lagru>`_)
- Marianne Corvellec (`@mkcor <https://github.com/mkcor>`_)
- Mark Harfouche (`@hmaarrfk <https://github.com/hmaarrfk>`_)
- Marvin Albert (`@m-albert <https://github.com/m-albert>`_)
- Matthew Brett (`@matthew-brett <https://github.com/matthew-brett>`_)
- MS-GITS (`@Greenie0701 <https://github.com/Greenie0701>`_)
- Paweł Rzońca (`@clacrow <https://github.com/clacrow>`_)
- Sebastian Berg (`@seberg <https://github.com/seberg>`_)
- Sigurd Vargdal (`@Tensorboy2 <https://github.com/Tensorboy2>`_)
- Stefan van der Walt (`@stefanv <https://github.com/stefanv>`_)
- Tim Head (`@betatim <https://github.com/betatim>`_)
- Vicent Caselles-Ballester (`@vcasellesb <https://github.com/vcasellesb>`_)

_These lists are automatically generated, and may not be complete or may contain
duplicates._

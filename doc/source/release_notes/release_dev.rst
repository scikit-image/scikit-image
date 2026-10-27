scikit-image 2.7.0rc0
=====================

We're happy to announce the release of scikit-image 2.7.0rc0!

New Features
------------

- Add a warning when importing experimental skimage2 (`#7873 <https://github.com/scikit-image/scikit-image/pull/7873>`_).
- In ``skimage.feature``, add a new parameter ``prescale`` to the functions  ``blob_doh``, blob_dog``, and ``blob_log``. This parameter controls the internal rescaling behavior which makes it easier to set the ``threshold`` parameter  correctly (`#7858 <https://github.com/scikit-image/scikit-image/pull/7858>`_).

API Changes
-----------

- In ``skimage.io``, remove ``imshow``, ``imshow_collection``, and ``show``. This completes their deprecation. Instead, we recommend using  matplotlib, napari, etc. for visualization (`#7994 <https://github.com/scikit-image/scikit-image/pull/7994>`_).
- In ``skimage.feature``, deprecate the parameter ``threshold_rel`` in  ``blob_doh``, blob_dog``, and ``blob_log``. The behavior of the parameters effects were difficult to reason about. Instead, use ``threshold`` in conjunction with the new parameter ``prescale`` (`#7858 <https://github.com/scikit-image/scikit-image/pull/7858>`_).
- In ``skimage.feature.peak_local_max``, use ``None`` as the new default for the parameters ``num_peaks`` and ``num_peaks_per_label`` – the default behavior is the same as before but passing ``numpy.inf`` is deprecated (`#8039 <https://github.com/scikit-image/scikit-image/pull/8039>`_).

Enhancements
------------

- In ``skimage.measure.ransac``, add the new optional parameter ``model_kwargs``, which allows forwarding keyword parameters to the model's ``from_estimate`` method. ``RansacModelProtocol`` has been updated accordingly (`#8015 <https://github.com/scikit-image/scikit-image/pull/8015>`_).
- In ``skimage.feature.peak_local_max``, support passing floats to the parameter ``min_distance`` (`#8039 <https://github.com/scikit-image/scikit-image/pull/8039>`_).

Bug Fixes
---------

- In ``skimage.morphology``, fix an unintended change in deprecated behavior in ``remove_small_objects`` and ``remove_small_holes``. The deprecation of the  parameters ``min_size`` and ``area_threshold`` introduced an off-by-one error when using them (`#8003 <https://github.com/scikit-image/scikit-image/pull/8003>`_).
- Don't override users test selection with ``spin test`` (`#7889 <https://github.com/scikit-image/scikit-image/pull/7889>`_).
- In ``skimage.util.map_array``, allow read-only arrays  (``WRITEABLE`` flag set to ``False``) as input to the parameters ``input_arr``, ``input_vals``, and ``output_vals`` (`#7535 <https://github.com/scikit-image/scikit-image/pull/7535>`_).
- Ensure ``skimage.feature.peak_local_max`` returns results with the correct shape when no peaks where found and ``labels`` where given (`#8039 <https://github.com/scikit-image/scikit-image/pull/8039>`_).
- Undo (double) mirroring in ``ski2.morphology.dilation`` (`#8060 <https://github.com/scikit-image/scikit-image/pull/8060>`_).
- Unwrap seed refactor (`#8142 <https://github.com/scikit-image/scikit-image/pull/8142>`_).
- Fix structural_similarity win_size / gaussian_weights conflict (`#8162 <https://github.com/scikit-image/scikit-image/pull/8162>`_).
- In ``skimage.feature.corner_peaks``, return peaks in intensity order  when using ``labels`` and ``num_peaks`` as promised (`#8167 <https://github.com/scikit-image/scikit-image/pull/8167>`_).
- Honor NumPy's ``copy`` semantics in the ``__array__`` implementations for geometric transforms and ``ArrayMap`` (`#8279 <https://github.com/scikit-image/scikit-image/pull/8279>`_).

Documentation
-------------

- Use consistent naming for release notes (`#7992 <https://github.com/scikit-image/scikit-image/pull/7992>`_).
- Update SKIP 4 "Transitioning to scikit-image 2.0" (`#7866 <https://github.com/scikit-image/scikit-image/pull/7866>`_).
- Fix source links in HTML docs after switch to src-layout (`#8044 <https://github.com/scikit-image/scikit-image/pull/8044>`_).
- Fix conventions following docstub and misc (`#8055 <https://github.com/scikit-image/scikit-image/pull/8055>`_).
- Add initial AI contribution policy (`#7982 <https://github.com/scikit-image/scikit-image/pull/7982>`_).
- Restore wording of AI policy (`#8123 <https://github.com/scikit-image/scikit-image/pull/8123>`_).
- Document warn_external in contributing guide (`#8163 <https://github.com/scikit-image/scikit-image/pull/8163>`_).
- docs: make source install instructions more discoverable and add build requirements to top level requirements file (`#8165 <https://github.com/scikit-image/scikit-image/pull/8165>`_).
- One page per function / class in HTML docs for skimage2 (`#8154 <https://github.com/scikit-image/scikit-image/pull/8154>`_).
- Update SKIP 4 (import hierarchy, backporting, testing) (`#8097 <https://github.com/scikit-image/scikit-image/pull/8097>`_).
- Reword note on masked arrays (`#8002 <https://github.com/scikit-image/scikit-image/pull/8002>`_).
- Update AI usage guideline (`#8272 <https://github.com/scikit-image/scikit-image/pull/8272>`_).
- Improve docstring of ``skimage2.data.binary_blobs`` (`#8294 <https://github.com/scikit-image/scikit-image/pull/8294>`_).
- Document how to run doctests only (`#8134 <https://github.com/scikit-image/scikit-image/pull/8134>`_).
- Update AI policy in response to recent experience (`#8313 <https://github.com/scikit-image/scikit-image/pull/8313>`_).
- Split contribution guide into separate, smaller, and more focused documents (`#8344 <https://github.com/scikit-image/scikit-image/pull/8344>`_).

Infrastructure
--------------

- Ignore pyparsing DeprecationWarning caused by matplotlib (`#7999 <https://github.com/scikit-image/scikit-image/pull/7999>`_).
- Don't trigger "Release wheels" workflow on pyproject.toml change (`#7995 <https://github.com/scikit-image/scikit-image/pull/7995>`_).
- Update docstub to version 0.6 (`#8013 <https://github.com/scikit-image/scikit-image/pull/8013>`_).
- Run docstub on skimage2 namespace too (`#8021 <https://github.com/scikit-image/scikit-image/pull/8021>`_).
- Add coderabbit review config (`#8032 <https://github.com/scikit-image/scikit-image/pull/8032>`_).
- Adjust coderabbit config to better fit our development style (`#8035 <https://github.com/scikit-image/scikit-image/pull/8035>`_).
- Don't override users test selection with ``spin test`` (`#7889 <https://github.com/scikit-image/scikit-image/pull/7889>`_).
- Try to make coderabbit respect exisiting type labels (`#8042 <https://github.com/scikit-image/scikit-image/pull/8042>`_).
- Fix GIL being re-enabled by C++ extensions (`#8059 <https://github.com/scikit-image/scikit-image/pull/8059>`_).
- Turn off coderabbit auto-labeling & label checks (`#8070 <https://github.com/scikit-image/scikit-image/pull/8070>`_).
- Build HTML API reference for skimage2 (`#8111 <https://github.com/scikit-image/scikit-image/pull/8111>`_).
- Refactor migration doctests and run as tests (`#8133 <https://github.com/scikit-image/scikit-image/pull/8133>`_).
- Update pyodide CI test to match the pattern used by numpy (`#8140 <https://github.com/scikit-image/scikit-image/pull/8140>`_).
- Drop Python 3.11 Support (`#8153 <https://github.com/scikit-image/scikit-image/pull/8153>`_).
- CI: Fix emscripten dependencies and clean up wheel test commands (`#8147 <https://github.com/scikit-image/scikit-image/pull/8147>`_).
- ci: Remove llvm build and add ccache support on windows (`#8164 <https://github.com/scikit-image/scikit-image/pull/8164>`_).
- Drop ``--maxfail=5`` from local pytest config (`#8170 <https://github.com/scikit-image/scikit-image/pull/8170>`_).
- Doctest migration in PRs & on all platforms (`#8141 <https://github.com/scikit-image/scikit-image/pull/8141>`_).
- ci: improve handling of pr labels (`#8169 <https://github.com/scikit-image/scikit-image/pull/8169>`_).
- CI: Use ccache on macos (`#8185 <https://github.com/scikit-image/scikit-image/pull/8185>`_).
- ci: fix ccache reporting on windows and linux (`#8187 <https://github.com/scikit-image/scikit-image/pull/8187>`_).
- CI: Fix module discovery when using spin test flags (`#8173 <https://github.com/scikit-image/scikit-image/pull/8173>`_).
- Reduce manual work when using ``ski2_migration_decorator`` (`#8179 <https://github.com/scikit-image/scikit-image/pull/8179>`_).
- Add pip-only dependencies to environment.yml (`#8144 <https://github.com/scikit-image/scikit-image/pull/8144>`_).
- Update mailmap and add mailmap checker (`#7984 <https://github.com/scikit-image/scikit-image/pull/7984>`_).
- Ignore pyenv virtualenv env name marker (`#8233 <https://github.com/scikit-image/scikit-image/pull/8233>`_).
- Ensure ``spin docs --clean`` also wipes ``doc/source/api2`` (`#8247 <https://github.com/scikit-image/scikit-image/pull/8247>`_).
- Update pre-commit hooks (`#8282 <https://github.com/scikit-image/scikit-image/pull/8282>`_).
- Fix test suite startup penalty due to accidental binary_blobs input (`#8294 <https://github.com/scikit-image/scikit-image/pull/8294>`_).
- Add support for Python 3.15, including free-threaded 3.15t, and publish cp315 wheels (`#8292 <https://github.com/scikit-image/scikit-image/pull/8292>`_).
- Bump kaleido to >=1.0 and provision Chrome for doc builds (`#8308 <https://github.com/scikit-image/scikit-image/pull/8308>`_).
- Fix error reporting (`#8076 <https://github.com/scikit-image/scikit-image/pull/8076>`_).
- Update zizmor to new location (`#8351 <https://github.com/scikit-image/scikit-image/pull/8351>`_).
- CI: Add native Linux ARM64 build and test coverage (`#8359 <https://github.com/scikit-image/scikit-image/pull/8359>`_).
- Bump the actions group across 1 directory with 11 updates (`#8365 <https://github.com/scikit-image/scikit-image/pull/8365>`_).
- Add meson option to exclude v2 namespace (`#8369 <https://github.com/scikit-image/scikit-image/pull/8369>`_).

Maintenance
-----------

- Handle PIL Image.getdata deprecation (`#8010 <https://github.com/scikit-image/scikit-image/pull/8010>`_).
- Fix array doctype syntax & other small docstub fixes (`#8011 <https://github.com/scikit-image/scikit-image/pull/8011>`_).
- Fix invalid syntax (M, N) ndarray: docstub fixes (III) (`#8014 <https://github.com/scikit-image/scikit-image/pull/8014>`_).
- Fix invalid syntax (..., C=3, ...) ndarray: docstub fixes (VI) (`#8026 <https://github.com/scikit-image/scikit-image/pull/8026>`_).
- Fix invalid syntax 2-D array: docstub fixes (IV) (`#8023 <https://github.com/scikit-image/scikit-image/pull/8023>`_).
- Address docstub errors (V) (`#8024 <https://github.com/scikit-image/scikit-image/pull/8024>`_).
- Fix invalid syntax (N, 2) array: docstub fixes (VII) (`#8027 <https://github.com/scikit-image/scikit-image/pull/8027>`_).
- Bump docstub to v0.7.0rc0 and fix errors (8) (`#8028 <https://github.com/scikit-image/scikit-image/pull/8028>`_).
- Fix Unknown name in doctype: 'data-type' docstub error (10) (`#8036 <https://github.com/scikit-image/scikit-image/pull/8036>`_).
- Add a warning when importing experimental skimage2 (`#7873 <https://github.com/scikit-image/scikit-image/pull/7873>`_).
- Add ``skimage._shared._warnings.warn_external`` which automatically sets the correct stacklevel for warnings (`#7976 <https://github.com/scikit-image/scikit-image/pull/7976>`_).
- Add ``skimage2.data.binary_blobs`` with improved signature compared to the old version in ``skimage.data``. The old version is now a simple wrapper around the implementation in ``skimage2`` (`#7976 <https://github.com/scikit-image/scikit-image/pull/7976>`_).
- Port ``skimage.feature.peak_local_max`` to ``skimage2.feature`` with a clearer API and improved defaults (`#8039 <https://github.com/scikit-image/scikit-image/pull/8039>`_).
- Port grayscale morphology operators to skimage2 (`#8046 <https://github.com/scikit-image/scikit-image/pull/8046>`_).
- Avoid circular import in ``feature/corner.py`` (`#8077 <https://github.com/scikit-image/scikit-image/pull/8077>`_).
- Avoid deprecated assign to ndarray.shape (`#8020 <https://github.com/scikit-image/scikit-image/pull/8020>`_).
- MAINT: make ellipse fitting forward compatible (`#8054 <https://github.com/scikit-image/scikit-image/pull/8054>`_).
- Use lazy-loader 0.5 (`#8080 <https://github.com/scikit-image/scikit-image/pull/8080>`_).
- Ensure that ``skimage2`` does not eagerly import ``skimage`` (`#8087 <https://github.com/scikit-image/scikit-image/pull/8087>`_).
- Move ``skimage2`` implementation into ``_skimage2`` (`#8093 <https://github.com/scikit-image/scikit-image/pull/8093>`_).
- Add rescaling API to skimage2 (`#8075 <https://github.com/scikit-image/scikit-image/pull/8075>`_).
- Port ``pad_footprint`` & ``mirror_footprint`` to ``skimage2`` (`#8094 <https://github.com/scikit-image/scikit-image/pull/8094>`_).
- Move ``_shared`` to ``_skimage2`` (`#8102 <https://github.com/scikit-image/scikit-image/pull/8102>`_).
- Revert "Add AI contribution policy (#7982)" (`#8124 <https://github.com/scikit-image/scikit-image/pull/8124>`_).
- Add ``skimage2.feature.canny`` which changes the default of the optional ``mode`` parameter from  ``constant`` to ``nearest`` compared to the old version in ``skimage.data``. The old version is now  a simple wrapper around the implementation in ``skimage2`` (`#8047 <https://github.com/scikit-image/scikit-image/pull/8047>`_).
- Add skimage2 porting decorator (`#8114 <https://github.com/scikit-image/scikit-image/pull/8114>`_).
- Port tests for ``features.canny`` to skimage2 (`#8126 <https://github.com/scikit-image/scikit-image/pull/8126>`_).
- Improve inserted docstring note of ``deprecated_func`` (`#8095 <https://github.com/scikit-image/scikit-image/pull/8095>`_).
- Suppress tifffile DeprecationWarning in test_plugin_deprecation_on_imread (`#8138 <https://github.com/scikit-image/scikit-image/pull/8138>`_).
- Drop Python 3.11 Support (`#8153 <https://github.com/scikit-image/scikit-image/pull/8153>`_).
- Fix minkowski distanced invocation for SciPy >= 1.18 (`#8157 <https://github.com/scikit-image/scikit-image/pull/8157>`_).
- Update deprecated scipy distance function and address underspecified test (`#8156 <https://github.com/scikit-image/scikit-image/pull/8156>`_).
- Static inline for ``__inline`` (`#8159 <https://github.com/scikit-image/scikit-image/pull/8159>`_).
- Port ``gaussian`` to ``skimage2.filters`` (`#8069 <https://github.com/scikit-image/scikit-image/pull/8069>`_).
- Set ``__doctest_requires__`` in requires decorator (`#7814 <https://github.com/scikit-image/scikit-image/pull/7814>`_).
- Remove port_module script (`#8225 <https://github.com/scikit-image/scikit-image/pull/8225>`_).
- Convert contribution guide to Markdown format (`#8188 <https://github.com/scikit-image/scikit-image/pull/8188>`_).
- ``skimage2.graph.pixel_graph`` and ``skimage2.metrics.contingency_table`` always return a ``scipy.sparse.csr_array`` and no longer accept a ``sparse_type`` parameter. The corresponding ``skimage`` functions are unchanged and still return a ``scipy.sparse.csr_matrix`` by default (`#8287 <https://github.com/scikit-image/scikit-image/pull/8287>`_).
- Update lazy-loader (`#8340 <https://github.com/scikit-image/scikit-image/pull/8340>`_).
- Test new milestone action (`#8354 <https://github.com/scikit-image/scikit-image/pull/8354>`_).
- Test new attach-milestone action (`#8355 <https://github.com/scikit-image/scikit-image/pull/8355>`_).
- Update minimum dependencies (SPEC 0) (`#8356 <https://github.com/scikit-image/scikit-image/pull/8356>`_).

Other
-----

- Move ``ensure_spacing`` into ``skimage2`` (`#8067 <https://github.com/scikit-image/scikit-image/pull/8067>`_).
- Grammar (`#8091 <https://github.com/scikit-image/scikit-image/pull/8091>`_).
- Drop 3.13t builds from cibuildwheel configuration (`#8127 <https://github.com/scikit-image/scikit-image/pull/8127>`_).
- Test with pythonoptimize (`#8108 <https://github.com/scikit-image/scikit-image/pull/8108>`_).
- Disable tests using matplotlib on pyodide (`#8082 <https://github.com/scikit-image/scikit-image/pull/8082>`_).
- The scikit-image test suite was updated to avoid using the global NumPy RNG. This makes it safe to run multiple tests simultaneously without any cross-talk (`#8084 <https://github.com/scikit-image/scikit-image/pull/8084>`_).
- Use ccache and reuse builds on Linux (`#8106 <https://github.com/scikit-image/scikit-image/pull/8106>`_).
- make spin build requirement a lower bound (`#8158 <https://github.com/scikit-image/scikit-image/pull/8158>`_).
- Port ``structural_similarly`` to ski2 (`#8104 <https://github.com/scikit-image/scikit-image/pull/8104>`_).
- Test only changed subpackages (`#8103 <https://github.com/scikit-image/scikit-image/pull/8103>`_).
- Pin minimum pytest to 9 (`#8174 <https://github.com/scikit-image/scikit-image/pull/8174>`_).
- Refactor, document migration utilities (`#8184 <https://github.com/scikit-image/scikit-image/pull/8184>`_).
- The project now has testing on the free-threaded-build using pytest-run-parallel to detect thread safety issues (`#8065 <https://github.com/scikit-image/scikit-image/pull/8065>`_).
- TST: delete unnecessary pytest thread-unsafe marks (`#8190 <https://github.com/scikit-image/scikit-image/pull/8190>`_).
- DOC: unmatched backtick (`#8205 <https://github.com/scikit-image/scikit-image/pull/8205>`_).
- Fix flaky unit test ``test_structural_similarity`` (`#8208 <https://github.com/scikit-image/scikit-image/pull/8208>`_).
- Allow positional ``blob_size`` param in ``binary_blobs`` (`#8214 <https://github.com/scikit-image/scikit-image/pull/8214>`_).
- Move source to _skimage2, copy tests (`#8204 <https://github.com/scikit-image/scikit-image/pull/8204>`_).
- Copy color data files back to tests/skimage (`#8229 <https://github.com/scikit-image/scikit-image/pull/8229>`_).
- Harmonize doctest imports (`#8234 <https://github.com/scikit-image/scikit-image/pull/8234>`_).
- Skimage2 -> Skimage1 doctest postprocessing (`#8219 <https://github.com/scikit-image/scikit-image/pull/8219>`_).
- Switch to using redirector app (`#8266 <https://github.com/scikit-image/scikit-image/pull/8266>`_).
- Fix resize_local_mean output shape with non-last channel axis (`#8281 <https://github.com/scikit-image/scikit-image/pull/8281>`_).
- In ``skimage2.morphology``, create ``footprint_rectangle_decomposed`` to cover the decomposition functionality that ``footprint_rectangle`` had. ``footprint_rectangle`` is now a simple wrapper around ``numpy.ones`` (`#8244 <https://github.com/scikit-image/scikit-image/pull/8244>`_).
- Tweak AI policy (`#8293 <https://github.com/scikit-image/scikit-image/pull/8293>`_).
- Use object-oriented matplotlib style in camera example in documentation (`#8257 <https://github.com/scikit-image/scikit-image/pull/8257>`_).
- Make ``from skimage2.subpackage import X`` work (`#8310 <https://github.com/scikit-image/scikit-image/pull/8310>`_).
- Fix typo in astronaut docstring (`#8246 <https://github.com/scikit-image/scikit-image/pull/8246>`_).
- DOC: note radians where missing, or incorrect (`#8329 <https://github.com/scikit-image/scikit-image/pull/8329>`_).
- Fix flaky test_ensure_spacing_max_batch_size (`#8338 <https://github.com/scikit-image/scikit-image/pull/8338>`_).
- Add guard for parallel execution error (`#8336 <https://github.com/scikit-image/scikit-image/pull/8336>`_).
- Fix for Meijering alpha default (`#8333 <https://github.com/scikit-image/scikit-image/pull/8333>`_).
- Raise ValueError in SIFT when sigma_in > sigma_min / upsampling (`#8304 <https://github.com/scikit-image/scikit-image/pull/8304>`_).
- mailmap: add left-out alias for Tony Yu (`#8352 <https://github.com/scikit-image/scikit-image/pull/8352>`_).
- Simplify / humanize the meijering alpha tests (`#8350 <https://github.com/scikit-image/scikit-image/pull/8350>`_).

Contributors
------------

22 authors added to this release (alphabetically):

- `@dependabot[bot] <https://github.com/apps/dependabot>`_
- `@greptile-apps[bot] <https://github.com/apps/greptile-apps>`_
- `@michaelbratsch <https://github.com/michaelbratsch>`_
- A Aswanth Raj (`@aswanth-07 <https://github.com/aswanth-07>`_)
- Edgar Andrés Margffoy Tuay (`@andfoy <https://github.com/andfoy>`_)
- Eric Larson (`@larsoner <https://github.com/larsoner>`_)
- Evgeni Burovski (`@ev-br <https://github.com/ev-br>`_)
- Francesco Siciliani (`@sici17 <https://github.com/sici17>`_)
- Jarrod Millman (`@jarrodmillman <https://github.com/jarrodmillman>`_)
- Jeff (`@jdotc1 <https://github.com/jdotc1>`_)
- Jeremy Muhlich (`@jmuhlich <https://github.com/jmuhlich>`_)
- Juan Nunez-Iglesias (`@jni <https://github.com/jni>`_)
- Lars Grüter (`@lagru <https://github.com/lagru>`_)
- M Bussonnier (`@Carreau <https://github.com/Carreau>`_)
- Marianne Corvellec (`@mkcor <https://github.com/mkcor>`_)
- Matthew Brett (`@matthew-brett <https://github.com/matthew-brett>`_)
- Nathan Goldbaum (`@ngoldbaum <https://github.com/ngoldbaum>`_)
- NIKHIL (`@Nikhi00718 <https://github.com/Nikhi00718>`_)
- Rafael Tappe Maestro (`@Wehzie <https://github.com/Wehzie>`_)
- Ravi Kumar (`@ravikumar266 <https://github.com/ravikumar266>`_)
- Stefan van der Walt (`@stefanv <https://github.com/stefanv>`_)
- Steven Silvester (`@blink1073 <https://github.com/blink1073>`_)

18 reviewers added to this release (alphabetically):

- `@coderabbitai[bot] <https://github.com/apps/coderabbitai>`_
- `@Copilot <https://github.com/apps/copilot-pull-request-reviewer>`_
- `@greptile-apps[bot] <https://github.com/apps/greptile-apps>`_
- `@michaelbratsch <https://github.com/michaelbratsch>`_
- Curtis Rueden (`@ctrueden <https://github.com/ctrueden>`_)
- Evgeni Burovski (`@ev-br <https://github.com/ev-br>`_)
- Igor Stadnyk (`@fallenmi <https://github.com/fallenmi>`_)
- Jarrod Millman (`@jarrodmillman <https://github.com/jarrodmillman>`_)
- Jeff (`@jdotc1 <https://github.com/jdotc1>`_)
- Juan Nunez-Iglesias (`@jni <https://github.com/jni>`_)
- Lars Grüter (`@lagru <https://github.com/lagru>`_)
- Marianne Corvellec (`@mkcor <https://github.com/mkcor>`_)
- Mark Harfouche (`@hmaarrfk <https://github.com/hmaarrfk>`_)
- Matthew Brett (`@matthew-brett <https://github.com/matthew-brett>`_)
- Nathan Goldbaum (`@ngoldbaum <https://github.com/ngoldbaum>`_)
- Ravi Kumar (`@ravikumar266 <https://github.com/ravikumar266>`_)
- Stefan van der Walt (`@stefanv <https://github.com/stefanv>`_)
- Steven Silvester (`@blink1073 <https://github.com/blink1073>`_)

_These lists are automatically generated, and may not be complete or may contain
duplicates._

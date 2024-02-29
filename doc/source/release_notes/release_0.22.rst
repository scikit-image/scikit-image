scikit-image 0.22.0 (2023-11-03)
================================

We're happy to announce the release of scikit-image 0.22.0!

Highlights
----------

- Add new image sequence ``skimage.data.palisades_of_vogt`` showing in-vivo tissue of the palisades of Vogt (`#6858 <https://github.com/scikit-image/scikit-image/pull/6858>`_).
- Add inpainting example to gallery (`#6853 <https://github.com/scikit-image/scikit-image/pull/6853>`_).

New Features
------------

- Add new image sequence ``skimage.data.palisades_of_vogt`` showing in-vivo tissue of the palisades of Vogt (`#6858 <https://github.com/scikit-image/scikit-image/pull/6858>`_).

API Changes
-----------

- Update minimal required NumPy version to 1.22 (`#7005 <https://github.com/scikit-image/scikit-image/pull/7005>`_).
- Update minimal required lazy_loader version to 0.3 (`#7043 <https://github.com/scikit-image/scikit-image/pull/7043>`_).
- Make PyWavelets an optional dependency which is only required for ``skimage.restoration.denoise_wavelet`` and ``skimage.restoration.estimate_sigma`` (`#7156 <https://github.com/scikit-image/scikit-image/pull/7156>`_).
- Remove deprecated ``skimage.filters.inverse``. Use ``skimage.filters.filter_inverse`` instead (`#7161 <https://github.com/scikit-image/scikit-image/pull/7161>`_).
- Remove deprecated function ``skimage.filters.ridges.compute_hessian_eigenvalues``.  Use ``skimage.feature.hessian_matrix_eigvals`` on the results of  ``skimage.feature.hessian_matrix`` instead (`#7162 <https://github.com/scikit-image/scikit-image/pull/7162>`_).
- Remove deprecated automatic detection of the color channel in  ``skimage.filters.gaussian``. ``channel_axis=None`` now indicates a grayscale image. Set the color channel with ``channel_axis=-1`` explicitly (`#7164 <https://github.com/scikit-image/scikit-image/pull/7164>`_).
- Change number of returned arguments in ``skimage.registration.phase_cross_correlation``. The function now always returns the 3 arguments ``shift``, ``error``, and ``phasediff`` (`#7166 <https://github.com/scikit-image/scikit-image/pull/7166>`_).
- Deprecate ``return_error`` in ``skimage.registration.phase_cross_correlation`` (`#7174 <https://github.com/scikit-image/scikit-image/pull/7174>`_).

Performance
-----------

- Add lazy loading to the ``skimage.feature`` submodule (`#6983 <https://github.com/scikit-image/scikit-image/pull/6983>`_).
- Add lazy loading to the ``skimage.graph`` submodule (`#6985 <https://github.com/scikit-image/scikit-image/pull/6985>`_).
- Add lazy loading to the ``skimage.measure`` submodule (`#6999 <https://github.com/scikit-image/scikit-image/pull/6999>`_).
- Add lazy loading to the ``skimage.transform`` submodule (`#7009 <https://github.com/scikit-image/scikit-image/pull/7009>`_).
- Add lazy loading to the ``skimage.restoration`` submodule (`#7021 <https://github.com/scikit-image/scikit-image/pull/7021>`_).
- Add lazy loading to the ``skimage.registration`` submodule (`#7032 <https://github.com/scikit-image/scikit-image/pull/7032>`_).
- Avoid unnecessary padding in ``skimage.measure.block_resize`` (`#7092 <https://github.com/scikit-image/scikit-image/pull/7092>`_).

Bug Fixes
---------

- Add missing properties ``num_pixels`` and ``coords_scaled`` to  ``skimage.measure.regionprops_table`` (`#7039 <https://github.com/scikit-image/scikit-image/pull/7039>`_).
- Fix ``too many values to unpack error`` with imageio 2.31.1 in ``video.rst`` (`#7076 <https://github.com/scikit-image/scikit-image/pull/7076>`_).
- Address error in ``skimage.filters.threshold_triangle`` when images with a uniform intensity are passed. In these cases the uniform intensity is returned as the threshold (`#7098 <https://github.com/scikit-image/scikit-image/pull/7098>`_).
- Fix error in ``skimage.color.lab2rgb`` for 1D input (`#7116 <https://github.com/scikit-image/scikit-image/pull/7116>`_).
- Make ``skimage.registration.phase_cross_correlation`` consistently return an array even if ``disambiguate=True`` (`#7112 <https://github.com/scikit-image/scikit-image/pull/7112>`_).
- Allow ``extra_properties`` of non-equal lengths to be passed correctly to  ``skimage.measure.regionprops_table`` (`#7136 <https://github.com/scikit-image/scikit-image/pull/7136>`_).

Documentation
-------------

- Use a more descriptive title for current inpainting example (`#6989 <https://github.com/scikit-image/scikit-image/pull/6989>`_).
- Fix URLs to the install page (`#6998 <https://github.com/scikit-image/scikit-image/pull/6998>`_).
- Fix equation for Wiener filter in ``restoration.wiener``'s docstring (`#6987 <https://github.com/scikit-image/scikit-image/pull/6987>`_).
- Fix missing links in INSTALL.rst and simplify language (`#6984 <https://github.com/scikit-image/scikit-image/pull/6984>`_).
- Edit installation and contributor guidelines (`#6991 <https://github.com/scikit-image/scikit-image/pull/6991>`_).
- Fix URLs that lead to 404 page (`#7008 <https://github.com/scikit-image/scikit-image/pull/7008>`_).
- Replace with correct reference to eigenvalues in ridge detection docstrings (`#7034 <https://github.com/scikit-image/scikit-image/pull/7034>`_).
- Add inpainting example to gallery (`#6853 <https://github.com/scikit-image/scikit-image/pull/6853>`_).
- Fix grammar in SKIP 1 (`#7081 <https://github.com/scikit-image/scikit-image/pull/7081>`_).
- Add missing minus in ``SimilarityTransform`` docstring (`#6840 <https://github.com/scikit-image/scikit-image/pull/6840>`_).
- Add one-line docstring to ``skimage.segmentation`` (`#6843 <https://github.com/scikit-image/scikit-image/pull/6843>`_).
- Add a short docstring to ``skimage.util`` (`#6831 <https://github.com/scikit-image/scikit-image/pull/6831>`_).
- Enable version warning banners for docs (`#7139 <https://github.com/scikit-image/scikit-image/pull/7139>`_).
- Clarify order of dimensions in ``skimage.transform.EuclideanTransform`` (`#7103 <https://github.com/scikit-image/scikit-image/pull/7103>`_).
- Add docstring to the ``morphology`` module (`#6814 <https://github.com/scikit-image/scikit-image/pull/6814>`_).
- Include Plausible analytics from Scientific Python in our HTML docs (`#7145 <https://github.com/scikit-image/scikit-image/pull/7145>`_).
- Specify coordinate convention in ``skimage.draw.polygon2mask`` (`#7131 <https://github.com/scikit-image/scikit-image/pull/7131>`_).
- Update 0.22.0 release notes (`#7182 <https://github.com/scikit-image/scikit-image/pull/7182>`_).

Infrastructure
--------------

- Pin milestone labeler to v0.1.0 SHA (`#6982 <https://github.com/scikit-image/scikit-image/pull/6982>`_).
- Ensure existing target directory for ``random.js`` (`#7015 <https://github.com/scikit-image/scikit-image/pull/7015>`_).
- Assign next milestone only for PRs targeting ``main`` branch (`#7018 <https://github.com/scikit-image/scikit-image/pull/7018>`_).
- Add missing directories to ``spin docs --clean`` command (`#7019 <https://github.com/scikit-image/scikit-image/pull/7019>`_).
- Rework ``generate_release_notes.py`` and add PR summary parsing (`#6961 <https://github.com/scikit-image/scikit-image/pull/6961>`_).
- Use packaged version of ``generate_release_notes.py`` (``changelist``) (`#7049 <https://github.com/scikit-image/scikit-image/pull/7049>`_).
- Generate requirements files from pyproject.toml (`#7085 <https://github.com/scikit-image/scikit-image/pull/7085>`_).
- Update spin to v0.5 (`#7093 <https://github.com/scikit-image/scikit-image/pull/7093>`_).
- Update to LLVM 16 with choco temporarily (`#7109 <https://github.com/scikit-image/scikit-image/pull/7109>`_).
- Update pytest config in ``pyproject.toml`` with repo-review recommendations (`#7063 <https://github.com/scikit-image/scikit-image/pull/7063>`_).
- Use checkout action version 4 (`#7180 <https://github.com/scikit-image/scikit-image/pull/7180>`_).

Maintenance
-----------

- Don't test numpy prerelease on azure (`#6996 <https://github.com/scikit-image/scikit-image/pull/6996>`_).
- Drop Python 3.8 support per SPEC 0 (`#6990 <https://github.com/scikit-image/scikit-image/pull/6990>`_).
- Upper pin imageio (`#7002 <https://github.com/scikit-image/scikit-image/pull/7002>`_).
- meson: allow proper selection of NumPy, Pythran in cross builds (`#7003 <https://github.com/scikit-image/scikit-image/pull/7003>`_).
- Unpin imageio and add warningfilter (`#7006 <https://github.com/scikit-image/scikit-image/pull/7006>`_).
- Update to latest attach-next-milestone action (`#7014 <https://github.com/scikit-image/scikit-image/pull/7014>`_).
- Avoid deprecated auto-removal of overlapping axes in thresholding example (`#7026 <https://github.com/scikit-image/scikit-image/pull/7026>`_).
- Remove conflicting setuptools upper pin (`#7045 <https://github.com/scikit-image/scikit-image/pull/7045>`_).
- Remove future.graph after v0.21 release (`#6899 <https://github.com/scikit-image/scikit-image/pull/6899>`_).
- Cleanup from move to pyproject.toml (`#7044 <https://github.com/scikit-image/scikit-image/pull/7044>`_).
- Ignore new matplotlib warning (`#7056 <https://github.com/scikit-image/scikit-image/pull/7056>`_).
- Update spin (`#7054 <https://github.com/scikit-image/scikit-image/pull/7054>`_).
- Ignore SciPy 1.12.dev0 deprecation warning (`#7057 <https://github.com/scikit-image/scikit-image/pull/7057>`_).
- Include expected warning for SciPy 1.12 (`#7058 <https://github.com/scikit-image/scikit-image/pull/7058>`_).
- Mark NaN-related deprecation warning from ``np.clip`` as optional in tests (`#7052 <https://github.com/scikit-image/scikit-image/pull/7052>`_).
- Fix abs value function warnings (`#7010 <https://github.com/scikit-image/scikit-image/pull/7010>`_).
- Temporary fix for wheel recipe (`#7059 <https://github.com/scikit-image/scikit-image/pull/7059>`_).
- Temporary fix for wheel building (`#7060 <https://github.com/scikit-image/scikit-image/pull/7060>`_).
- Remove outdated comment (`#7077 <https://github.com/scikit-image/scikit-image/pull/7077>`_).
- Include py.typed file in distribution (PEP 561) (`#7073 <https://github.com/scikit-image/scikit-image/pull/7073>`_).
- Transition user guide to ``import skimage as ski`` (`#7024 <https://github.com/scikit-image/scikit-image/pull/7024>`_).
- Fix for NumPy 1.25 (`#6970 <https://github.com/scikit-image/scikit-image/pull/6970>`_).
- Pin sphinx until sphinx-gallery is fixed (`#7100 <https://github.com/scikit-image/scikit-image/pull/7100>`_).
- Cleanup old Python 3.11 tests (`#7099 <https://github.com/scikit-image/scikit-image/pull/7099>`_).
- Revert "Pin sphinx until sphinx-gallery is fixed (#7100)" (`#7102 <https://github.com/scikit-image/scikit-image/pull/7102>`_).
- MNT: Remove ``np.float_`` alias; it is removed in NumPy 2.0 (`#7118 <https://github.com/scikit-image/scikit-image/pull/7118>`_).
- Fix for NumPy 1.26 (`#7101 <https://github.com/scikit-image/scikit-image/pull/7101>`_).
- Update meson-python (`#7120 <https://github.com/scikit-image/scikit-image/pull/7120>`_).
- We now require sklearn 1.1, as per [SPEC0](https://scientific-python.org/specs/spec-0000/) (`#7121 <https://github.com/scikit-image/scikit-image/pull/7121>`_).
- Update for NumPy 2 namespace cleanup (`#7119 <https://github.com/scikit-image/scikit-image/pull/7119>`_).
- DOC: minor numpydoc syntax update (`#7123 <https://github.com/scikit-image/scikit-image/pull/7123>`_).
- Update for NumPy 2 namespace cleanup (`#7122 <https://github.com/scikit-image/scikit-image/pull/7122>`_).
- Temporary work-around for NEP 51 numpy scalar reprs + doctests (`#7125 <https://github.com/scikit-image/scikit-image/pull/7125>`_).
- Update lazy loader (`#7126 <https://github.com/scikit-image/scikit-image/pull/7126>`_).
- Fix PEP 8 issues (`#7142 <https://github.com/scikit-image/scikit-image/pull/7142>`_).
- Remove single-threaded dask usage in face detection gallery example which fixes issues with running the example on Windows and CI (`#7141 <https://github.com/scikit-image/scikit-image/pull/7141>`_).
- Update spin version to 0.6 (`#7150 <https://github.com/scikit-image/scikit-image/pull/7150>`_).
- Match pep8speaks and ruff line lengths to 88 (`#7148 <https://github.com/scikit-image/scikit-image/pull/7148>`_).
- Remove last reference to distutils in ``_build_utils/tempita.py`` (`#7137 <https://github.com/scikit-image/scikit-image/pull/7137>`_).
- Update sphinx, sphinx-gallery & sphinx_design (`#7155 <https://github.com/scikit-image/scikit-image/pull/7155>`_).
- Update minimal version of numpydoc to 1.6 (`#7106 <https://github.com/scikit-image/scikit-image/pull/7106>`_).
- Build wheels for py3.12 (`#7082 <https://github.com/scikit-image/scikit-image/pull/7082>`_).
- Update label and milestone workflows (`#7163 <https://github.com/scikit-image/scikit-image/pull/7163>`_).
- Update TODO (see #6899) (`#7165 <https://github.com/scikit-image/scikit-image/pull/7165>`_).
- Announce Python 3.12 support (`#7167 <https://github.com/scikit-image/scikit-image/pull/7167>`_).
- Remove pep8speaks config (`#7172 <https://github.com/scikit-image/scikit-image/pull/7172>`_).
- Filter out expected runtime warnings in registation.phase_cross_correlation when disambiguate=True (`#7147 <https://github.com/scikit-image/scikit-image/pull/7147>`_).
- Use pre-commit bot (`#7171 <https://github.com/scikit-image/scikit-image/pull/7171>`_).
- Fix missing warnings import in ``phase_cross_correlation`` (`#7175 <https://github.com/scikit-image/scikit-image/pull/7175>`_).
- Fix release notes error (`#7177 <https://github.com/scikit-image/scikit-image/pull/7177>`_).
- Use trusted publisher (`#7178 <https://github.com/scikit-image/scikit-image/pull/7178>`_).

Contributors
------------

24 authors added to this release (alphabetically):

- `@akonsk <https://github.com/akonsk>`_
- `@patquem <https://github.com/patquem>`_
- `@rraadd88 <https://github.com/rraadd88>`_
- `@scott-vsi <https://github.com/scott-vsi>`_
- Adeyemi Biola  (`@decorouz <https://github.com/decorouz>`_)
- Amund Vedal (`@vedal <https://github.com/vedal>`_)
- Ananya Srivastava (`@ana42742 <https://github.com/ana42742>`_)
- Andrew J. Hesford (`@ahesford <https://github.com/ahesford>`_)
- Antony Lee (`@anntzer <https://github.com/anntzer>`_)
- Elena Pascal (`@elena-pascal <https://github.com/elena-pascal>`_)
- Jarrod Millman (`@jarrodmillman <https://github.com/jarrodmillman>`_)
- Juan Nunez-Iglesias (`@jni <https://github.com/jni>`_)
- Kenfack Anafack Alex Bruno (`@Br-Al <https://github.com/Br-Al>`_)
- Klaus Rettinghaus (`@rettinghaus <https://github.com/rettinghaus>`_)
- Larry Bradley (`@larrybradley <https://github.com/larrybradley>`_)
- Lars Grüter (`@lagru <https://github.com/lagru>`_)
- Marianne Corvellec (`@mkcor <https://github.com/mkcor>`_)
- Marvin Albert (`@m-albert <https://github.com/m-albert>`_)
- Matthias Bussonnier (`@Carreau <https://github.com/Carreau>`_)
- Matthias Nwt (`@matthiasnwt <https://github.com/matthiasnwt>`_)
- Mike Taves (`@mwtoews <https://github.com/mwtoews>`_)
- Riadh Fezzani (`@rfezzani <https://github.com/rfezzani>`_)
- Stefan van der Walt (`@stefanv <https://github.com/stefanv>`_)
- Talley Lambert (`@tlambert03 <https://github.com/tlambert03>`_)

19 reviewers added to this release (alphabetically):

- `@akonsk <https://github.com/akonsk>`_
- `@scott-vsi <https://github.com/scott-vsi>`_
- Adeyemi Biola  (`@decorouz <https://github.com/decorouz>`_)
- Ananya Srivastava (`@ana42742 <https://github.com/ana42742>`_)
- Andrew J. Hesford (`@ahesford <https://github.com/ahesford>`_)
- Egor Panfilov (`@soupault <https://github.com/soupault>`_)
- Grzegorz Bokota (`@Czaki <https://github.com/Czaki>`_)
- Jarrod Millman (`@jarrodmillman <https://github.com/jarrodmillman>`_)
- Juan Nunez-Iglesias (`@jni <https://github.com/jni>`_)
- Kristen Thyng (`@kthyng <https://github.com/kthyng>`_)
- Larry Bradley (`@larrybradley <https://github.com/larrybradley>`_)
- Lars Grüter (`@lagru <https://github.com/lagru>`_)
- Marianne Corvellec (`@mkcor <https://github.com/mkcor>`_)
- Mark Harfouche (`@hmaarrfk <https://github.com/hmaarrfk>`_)
- Marvin Albert (`@m-albert <https://github.com/m-albert>`_)
- Matthias Bussonnier (`@Carreau <https://github.com/Carreau>`_)
- Maxim (`@koshakOK <https://github.com/koshakOK>`_)
- Mike Taves (`@mwtoews <https://github.com/mwtoews>`_)
- Stefan van der Walt (`@stefanv <https://github.com/stefanv>`_)

*These lists are automatically generated, and may not be complete or may contain
duplicates.*

scikit-image 0.25.0rc1
======================

We're happy to announce the release of scikit-image 0.25.0rc1!

New Features
------------

- Add the new Grey-Level Co-occurrence Matrix (GLCM) properties  "mean", "variance", "standard deviation" and "entropy" to  ``skimage.feature.texture.graycoprops`` (`#7375 <https://github.com/scikit-image/scikit-image/pull/7375>`_).

API Changes
-----------

- Complete the deprecation of and remove ``skimage.feature.plot_matches``. Use ``skimage.feature.plot_matched_features`` going forward (`#7487 <https://github.com/scikit-image/scikit-image/pull/7487>`_).
- Deprecate ``skimage.io.imshow``, ``skimage.io.imshow_collection`` and ``skimage.io.show``. Please use ``matplotlib``, ``napari``, etc. to visualize images (`#7508 <https://github.com/scikit-image/scikit-image/pull/7508>`_).

Enhancements
------------

- Improve numerical stability of ``skimage.morphology.local_minima`` for extremely small floats (`#7534 <https://github.com/scikit-image/scikit-image/pull/7534>`_).

Bug Fixes
---------

- Ensure that ``skimage.morphology.remove_objects_by_distance`` doesn't fail  if the given integer dtype cannot be safely cast to the architecture specific size of ``intp``, e.g. on i386 architectures (`#7453 <https://github.com/scikit-image/scikit-image/pull/7453>`_).
- Fix degeneracy in ``skimage.draw.ellipsoid_stats`` when all semi-axes have the same length (`#7473 <https://github.com/scikit-image/scikit-image/pull/7473>`_).
- Prevent ``skimage.morphology.thin`` from accidentally  modifying the input image in case it is of dtype uint8 (`#7469 <https://github.com/scikit-image/scikit-image/pull/7469>`_).
- Fix numerical precision error in ``skimage.measure.ransac``. In some cases, ``ransac`` was stopping at the first iteration (`#7065 <https://github.com/scikit-image/scikit-image/pull/7065>`_).
- Fix numerical precision error in ``skimage.measure.ransac``;  very small probabilities lead to -0 number of max trials (`#7496 <https://github.com/scikit-image/scikit-image/pull/7496>`_).

Documentation
-------------

- In ``skimage.morphology.skeletonize``, clarify the expected image dtypes and how objects of different intensities are handled (`#7456 <https://github.com/scikit-image/scikit-image/pull/7456>`_).
- Fix example section in docstring of ``skimage.feature.graycomatrix`` (`#7297 <https://github.com/scikit-image/scikit-image/pull/7297>`_).
- Use conda-forge consistently in instructions for setting up the development environment (`#7483 <https://github.com/scikit-image/scikit-image/pull/7483>`_).
- Use new ``CITATION.cff`` instead of ``CITATION.bib`` (`#7505 <https://github.com/scikit-image/scikit-image/pull/7505>`_).
- Use correct ``spin test --coverage`` in contribution guide (`#7515 <https://github.com/scikit-image/scikit-image/pull/7515>`_).
- Tweak advice to new developers; remove AI warning (`#7522 <https://github.com/scikit-image/scikit-image/pull/7522>`_).
- Rework installation instructions (`#7434 <https://github.com/scikit-image/scikit-image/pull/7434>`_).
- Improve the description of the ``image`` parameter in ``skimage.restoration.richardson_lucy`` (`#7477 <https://github.com/scikit-image/scikit-image/pull/7477>`_).
- Account for empty arrays when counting segments per contour level in gallery example "Segment human cells (in mitosis)" (`#7551 <https://github.com/scikit-image/scikit-image/pull/7551>`_).

Infrastructure
--------------

- Fix CI tests with minimal dependencies and make dependency resolution more robust (`#7462 <https://github.com/scikit-image/scikit-image/pull/7462>`_).
- Add CI to test scikit-image against free-threaded Python 3.13 (`#7463 <https://github.com/scikit-image/scikit-image/pull/7463>`_).
- Address autosummary.import_cycle warning (`#7486 <https://github.com/scikit-image/scikit-image/pull/7486>`_).
- Temporarily exclude Dask 2024.8.0 to fix CI (`#7493 <https://github.com/scikit-image/scikit-image/pull/7493>`_).
- Uncomment ``currentmodule`` directive again (`#7492 <https://github.com/scikit-image/scikit-image/pull/7492>`_).
- Add CI to release nightly free-threaded wheels (`#7481 <https://github.com/scikit-image/scikit-image/pull/7481>`_).
- Update deprecated configuration (`#7501 <https://github.com/scikit-image/scikit-image/pull/7501>`_).
- Bump spin version to 0.11 (`#7507 <https://github.com/scikit-image/scikit-image/pull/7507>`_).
- Ensure only a single ``type:`` label is present in PRs (`#7512 <https://github.com/scikit-image/scikit-image/pull/7512>`_).
- Update pydata-sphinx-theme (`#7511 <https://github.com/scikit-image/scikit-image/pull/7511>`_).
- Fix OpenBLAS ``s_cmp`` unresolved symbol error, update Emscripten CI testing (`#7525 <https://github.com/scikit-image/scikit-image/pull/7525>`_).

Maintenance
-----------

- Verify all artifacts that have been attested by looping over them in CI (`#7447 <https://github.com/scikit-image/scikit-image/pull/7447>`_).
- Update circleci-artifacts-redirector-action that moved to the Scientific Python org (`#7446 <https://github.com/scikit-image/scikit-image/pull/7446>`_).
- Use NumPy 2.0 stable to build packages (`#7451 <https://github.com/scikit-image/scikit-image/pull/7451>`_).
- FIX Use python3 in Meson version script shebang (`#7482 <https://github.com/scikit-image/scikit-image/pull/7482>`_).
- Refactored tests for skeletonize (`#7459 <https://github.com/scikit-image/scikit-image/pull/7459>`_).
- Remove unused and deprecated dependency pytest-runner (`#7495 <https://github.com/scikit-image/scikit-image/pull/7495>`_).
- Exclude imageio 2.35.0 that forces numpy downgrade (`#7502 <https://github.com/scikit-image/scikit-image/pull/7502>`_).
- Don't test thresholding funcs for Dask compatibility (`#7509 <https://github.com/scikit-image/scikit-image/pull/7509>`_).
- Fix build dependency (`#7510 <https://github.com/scikit-image/scikit-image/pull/7510>`_).
- Add sdist check to ``spin sdist`` (`#7438 <https://github.com/scikit-image/scikit-image/pull/7438>`_).
- Reorder items in TODO list (`#7519 <https://github.com/scikit-image/scikit-image/pull/7519>`_).
- Use ``Rotation.from_euler`` to compute 3D rotation matrix (`#7503 <https://github.com/scikit-image/scikit-image/pull/7503>`_).
- Update spin (0.12) (`#7532 <https://github.com/scikit-image/scikit-image/pull/7532>`_).
- Import ``lazy_loader`` as private symbol in top-level namespaces (`#7540 <https://github.com/scikit-image/scikit-image/pull/7540>`_).
- Set -DNPY_NO_DEPRECATED_API=NPY_1_23_API_VERSION on build (`#7538 <https://github.com/scikit-image/scikit-image/pull/7538>`_).
- Update up/download artifact version (`#7545 <https://github.com/scikit-image/scikit-image/pull/7545>`_).
- Don't use deprecated ``io.show`` and ``io.imshow`` (`#7556 <https://github.com/scikit-image/scikit-image/pull/7556>`_).
- Hide traceback inside ``assert_stacklevel`` (`#7558 <https://github.com/scikit-image/scikit-image/pull/7558>`_).
- Update pre-commit versions (`#7560 <https://github.com/scikit-image/scikit-image/pull/7560>`_).
- Drop Python 3.9 support (`#7561 <https://github.com/scikit-image/scikit-image/pull/7561>`_).
- Update minimum dependencies (SPEC 0) (`#7562 <https://github.com/scikit-image/scikit-image/pull/7562>`_).
- Remove unused PYX files in io/_plugins (`#7557 <https://github.com/scikit-image/scikit-image/pull/7557>`_).
- Support Python 3.13 (`#7565 <https://github.com/scikit-image/scikit-image/pull/7565>`_).
- During deprecation cycles, preserve the value of deprecated parameters that don't have a new parameter as a replacement (`#7552 <https://github.com/scikit-image/scikit-image/pull/7552>`_).
- Fix missing minigalleries by using full names in directives (`#7567 <https://github.com/scikit-image/scikit-image/pull/7567>`_).
- Build Python 3.13 wheels (`#7571 <https://github.com/scikit-image/scikit-image/pull/7571>`_).
- Update TODO (`#7573 <https://github.com/scikit-image/scikit-image/pull/7573>`_).
- Remove deprecated skeletonize_3d (`#7572 <https://github.com/scikit-image/scikit-image/pull/7572>`_).
- Remove deprecated gaussian output parameter (`#7574 <https://github.com/scikit-image/scikit-image/pull/7574>`_).

Contributors
------------

21 authors added to this release (alphabetically):

- `@FedericoWZhaw <https://github.com/FedericoWZhaw>`_
- `@jakirkham <https://github.com/jakirkham>`_
- `@michaelbratsch <https://github.com/michaelbratsch>`_
- Adeyemi Biola  (`@decorouz <https://github.com/decorouz>`_)
- Agriya Khetarpal (`@agriyakhetarpal <https://github.com/agriyakhetarpal>`_)
- Brigitta Sipőcz (`@bsipocz <https://github.com/bsipocz>`_)
- Edgar Andrés Margffoy Tuay (`@andfoy <https://github.com/andfoy>`_)
- Egor Panfilov (`@soupault <https://github.com/soupault>`_)
- Erik Welch (`@eriknw <https://github.com/eriknw>`_)
- Gianluca (`@geeanlooca <https://github.com/geeanlooca>`_)
- Hayato Ikoma (`@hayatoikoma <https://github.com/hayatoikoma>`_)
- Jarrod Millman (`@jarrodmillman <https://github.com/jarrodmillman>`_)
- João Seródio (`@SerodioJ <https://github.com/SerodioJ>`_)
- Kushaan Gupta (`@kushaangupta <https://github.com/kushaangupta>`_)
- Lars Grüter (`@lagru <https://github.com/lagru>`_)
- Loïc Estève (`@lesteve <https://github.com/lesteve>`_)
- Marianne Corvellec (`@mkcor <https://github.com/mkcor>`_)
- Mark Harfouche (`@hmaarrfk <https://github.com/hmaarrfk>`_)
- Matthew Feickert (`@matthewfeickert <https://github.com/matthewfeickert>`_)
- Piyush Amitabh (`@pamitabh <https://github.com/pamitabh>`_)
- Stefan van der Walt (`@stefanv <https://github.com/stefanv>`_)

20 reviewers added to this release (alphabetically):

- `@FedericoWZhaw <https://github.com/FedericoWZhaw>`_
- `@jakirkham <https://github.com/jakirkham>`_
- `@michaelbratsch <https://github.com/michaelbratsch>`_
- Agriya Khetarpal (`@agriyakhetarpal <https://github.com/agriyakhetarpal>`_)
- Brigitta Sipőcz (`@bsipocz <https://github.com/bsipocz>`_)
- Edgar Andrés Margffoy Tuay (`@andfoy <https://github.com/andfoy>`_)
- Egor Panfilov (`@soupault <https://github.com/soupault>`_)
- Gianluca (`@geeanlooca <https://github.com/geeanlooca>`_)
- Hayato Ikoma (`@hayatoikoma <https://github.com/hayatoikoma>`_)
- Jarrod Millman (`@jarrodmillman <https://github.com/jarrodmillman>`_)
- João Seródio (`@SerodioJ <https://github.com/SerodioJ>`_)
- Kushaan Gupta (`@kushaangupta <https://github.com/kushaangupta>`_)
- Lars Grüter (`@lagru <https://github.com/lagru>`_)
- Marianne Corvellec (`@mkcor <https://github.com/mkcor>`_)
- Mark Harfouche (`@hmaarrfk <https://github.com/hmaarrfk>`_)
- Matthew Feickert (`@matthewfeickert <https://github.com/matthewfeickert>`_)
- Nathan Goldbaum (`@ngoldbaum <https://github.com/ngoldbaum>`_)
- Piyush Amitabh (`@pamitabh <https://github.com/pamitabh>`_)
- Ralf Gommers (`@rgommers <https://github.com/rgommers>`_)
- Stefan van der Walt (`@stefanv <https://github.com/stefanv>`_)

_These lists are automatically generated, and may not be complete or may contain
duplicates._


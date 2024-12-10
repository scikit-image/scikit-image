scikit-image 0.25.0rc2
======================

We're happy to announce the release of scikit-image 0.25.0rc2!

New Features
------------

- Add the new Gray-Level Co-occurrence Matrix (GLCM) properties  "mean", "variance", "standard deviation" and "entropy" to  ``skimage.feature.texture.graycoprops`` (`#7375 <https://github.com/scikit-image/scikit-image/pull/7375>`_).
- Add the new ``skimage.morphology.footprint_rectangle`` supporting generation of rectangular or hyper-rectangular footprints in one function (`#7566 <https://github.com/scikit-image/scikit-image/pull/7566>`_).

API Changes
-----------

- Complete the deprecation of and remove ``skimage.feature.plot_matches``. Use ``skimage.feature.plot_matched_features`` going forward (`#7487 <https://github.com/scikit-image/scikit-image/pull/7487>`_).
- Deprecate ``skimage.io.imshow``, ``skimage.io.imshow_collection`` and ``skimage.io.show``. Please use ``matplotlib``, ``napari``, etc. to visualize images (`#7508 <https://github.com/scikit-image/scikit-image/pull/7508>`_).
- Remove deprecated ``skimage.morphology.skeletonize_3d``;  use ``skimage.morphology.skeletonize`` instead (`#7572 <https://github.com/scikit-image/scikit-image/pull/7572>`_).
- Deprecate ``skimage.io`` plugin infrastructure (`#7353 <https://github.com/scikit-image/scikit-image/pull/7353>`_).
- Switched to using the ``scipy.sparse`` array interface. For more details, see the note about the new ``scipy.sparse`` array interface [here](https://docs.scipy.org/doc/scipy/reference/sparse.html) (`#7576 <https://github.com/scikit-image/scikit-image/pull/7576>`_).
- Deprecate ``skimage.morphology.cube`` in favor of the new function ``skimage.morphology.footprint_rectangle`` (`#7566 <https://github.com/scikit-image/scikit-image/pull/7566>`_).
- Deprecate ``skimage.morphology.rectangle`` in favor of the new function ``skimage.morphology.footprint_rectangle`` (`#7566 <https://github.com/scikit-image/scikit-image/pull/7566>`_).
- Deprecate ``skimage.morphology.square`` in favor of the new function ``skimage.morphology.footprint_rectangle`` (`#7566 <https://github.com/scikit-image/scikit-image/pull/7566>`_).

Enhancements
------------

- Improve numerical stability of ``skimage.morphology.local_minima`` for extremely small floats (`#7534 <https://github.com/scikit-image/scikit-image/pull/7534>`_).
- Make sure that ``skimage.feature.plot_matched_features`` uses the same random colors, if ``matches_color`` isn't provided  explicitly (`#7541 <https://github.com/scikit-image/scikit-image/pull/7541>`_).
- Allow passing a sequence of colors to the parameter ``matches_color`` in ``skimage.feature.plot_matched_features`` (`#7541 <https://github.com/scikit-image/scikit-image/pull/7541>`_).

Performance
-----------

- ``skimage.feature.peak_local_max`` will now skip unnecessary distance computations in the case of ``min_distance=1``. This results in performance improvements to functions like ``skimage.feature.blob_dog``, ``skimage.feature.blob_log``,  ``skimage.feature.blob_doh`` and ``skimage.feature.corner_peaks`` that call  ``peak_local_max`` internally (`#7548 <https://github.com/scikit-image/scikit-image/pull/7548>`_).
- In ``skimage.featurepeak_local_max``, skip unnecessary check for cases where  ``min_distance > 1`` is passed (`#7548 <https://github.com/scikit-image/scikit-image/pull/7548>`_).

Bug Fixes
---------

- Ensure that ``skimage.morphology.remove_objects_by_distance`` doesn't fail  if the given integer dtype cannot be safely cast to the architecture specific size of ``intp``, e.g. on i386 architectures (`#7453 <https://github.com/scikit-image/scikit-image/pull/7453>`_).
- Fix degeneracy in ``skimage.draw.ellipsoid_stats`` when all semi-axes have the same length (`#7473 <https://github.com/scikit-image/scikit-image/pull/7473>`_).
- Prevent ``skimage.morphology.thin`` from accidentally  modifying the input image in case it is of dtype uint8 (`#7469 <https://github.com/scikit-image/scikit-image/pull/7469>`_).
- Fix numerical precision error in ``skimage.measure.ransac``. In some cases, ``ransac`` was stopping at the first iteration (`#7065 <https://github.com/scikit-image/scikit-image/pull/7065>`_).
- Fix numerical precision error in ``skimage.measure.ransac``;  very small probabilities lead to -0 number of max trials (`#7496 <https://github.com/scikit-image/scikit-image/pull/7496>`_).
- Ensure that ``RegionProperties`` objects returned by ``skimage.measure.regionprops`` can be deserialized with pickle (`#7569 <https://github.com/scikit-image/scikit-image/pull/7569>`_).
- Fix edge case where setting ``watershed_lines=True`` in ``skimage.segmentation.watershed`` resulted in an incorrect solution (`#7071 <https://github.com/scikit-image/scikit-image/pull/7071>`_).
- Fix the behavior of ``skimage.segmentation.watershed`` when the markers don't align with local minima by making sure every marker is evaluated before successive pixels (`#7071 <https://github.com/scikit-image/scikit-image/pull/7071>`_).
- Fix dtype promotion in ``skimage.segmentation.join_segmentations`` if ``numpy.uint`` is used with NumPy<2 (`#7292 <https://github.com/scikit-image/scikit-image/pull/7292>`_).

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
- Fix typo in morphology doc (`#7606 <https://github.com/scikit-image/scikit-image/pull/7606>`_).
- Change type description of parameter ``radius`` in  ``skimage.morphology.ball`` from ``int`` to ``float`` (`#7627 <https://github.com/scikit-image/scikit-image/pull/7627>`_).

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
- Render paragraphs in dormant message (`#7549 <https://github.com/scikit-image/scikit-image/pull/7549>`_).
- Build sphinx documentation with parallel jobs (`#7579 <https://github.com/scikit-image/scikit-image/pull/7579>`_).
- Don't check test coverage in CI (`#7594 <https://github.com/scikit-image/scikit-image/pull/7594>`_).
- Explicitly setup conda on macos for wheel building (`#7608 <https://github.com/scikit-image/scikit-image/pull/7608>`_).

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
- Remove deprecated gaussian output parameter (`#7574 <https://github.com/scikit-image/scikit-image/pull/7574>`_).
- Test Py3.13 on windows (`#7578 <https://github.com/scikit-image/scikit-image/pull/7578>`_).
- Update ruff linter / formatter (`#7580 <https://github.com/scikit-image/scikit-image/pull/7580>`_).
- Fix formatting issues (`#7581 <https://github.com/scikit-image/scikit-image/pull/7581>`_).
- CI: bump macos image pin from 12 to 13 (`#7582 <https://github.com/scikit-image/scikit-image/pull/7582>`_).
- Update build dependencies (`#7587 <https://github.com/scikit-image/scikit-image/pull/7587>`_).
- Update minimum supported pyamg (`#7586 <https://github.com/scikit-image/scikit-image/pull/7586>`_).
- Update documentation dependencies (`#7590 <https://github.com/scikit-image/scikit-image/pull/7590>`_).
- Bump ``changelist`` to v0.5 (`#7601 <https://github.com/scikit-image/scikit-image/pull/7601>`_).
- Pin kaleido to 0.2.1 (`#7612 <https://github.com/scikit-image/scikit-image/pull/7612>`_).
- Update upload-nightly-action (`#7609 <https://github.com/scikit-image/scikit-image/pull/7609>`_).
- Update pillow (`#7615 <https://github.com/scikit-image/scikit-image/pull/7615>`_).
- Remove Python 2.7 cruft (`#7616 <https://github.com/scikit-image/scikit-image/pull/7616>`_).
- Use ``intersphinx_registry`` package in ``conf.py`` to keep intersphinx urls up to date. This means that building docs now requires the ``intersphinx-registry`` package (`#7611 <https://github.com/scikit-image/scikit-image/pull/7611>`_).
- Update build dependencies (`#7614 <https://github.com/scikit-image/scikit-image/pull/7614>`_).
- Update file extension and reformat Markdown file (`#7617 <https://github.com/scikit-image/scikit-image/pull/7617>`_).
- Add forgotten TODO about deprecated ``square``, ``cube`` & ``rectangle`` (`#7624 <https://github.com/scikit-image/scikit-image/pull/7624>`_).
- Upgrade to spin 0.13 (`#7622 <https://github.com/scikit-image/scikit-image/pull/7622>`_).
- Lazy load legacy imports in ``skimage`` top module (`#6892 <https://github.com/scikit-image/scikit-image/pull/6892>`_).

Contributors
------------

30 authors added to this release (alphabetically):

- `@aeisenbarth <https://github.com/aeisenbarth>`_
- `@FedericoWZhaw <https://github.com/FedericoWZhaw>`_
- `@jakirkham <https://github.com/jakirkham>`_
- `@michaelbratsch <https://github.com/michaelbratsch>`_
- Adeyemi Biola  (`@decorouz <https://github.com/decorouz>`_)
- Aditi Juneja (`@Schefflera-Arboricola <https://github.com/Schefflera-Arboricola>`_)
- Agriya Khetarpal (`@agriyakhetarpal <https://github.com/agriyakhetarpal>`_)
- Brigitta Sipőcz (`@bsipocz <https://github.com/bsipocz>`_)
- Dan Schult (`@dschult <https://github.com/dschult>`_)
- Edgar Andrés Margffoy Tuay (`@andfoy <https://github.com/andfoy>`_)
- Egor Panfilov (`@soupault <https://github.com/soupault>`_)
- Erik Welch (`@eriknw <https://github.com/eriknw>`_)
- Gianluca (`@geeanlooca <https://github.com/geeanlooca>`_)
- Gregory Lee (`@grlee77 <https://github.com/grlee77>`_)
- Hayato Ikoma (`@hayatoikoma <https://github.com/hayatoikoma>`_)
- Henrik Finsberg (`@finsberg <https://github.com/finsberg>`_)
- Jarrod Millman (`@jarrodmillman <https://github.com/jarrodmillman>`_)
- Jordão Bragantini (`@JoOkuma <https://github.com/JoOkuma>`_)
- João Seródio (`@SerodioJ <https://github.com/SerodioJ>`_)
- Kushaan Gupta (`@kushaangupta <https://github.com/kushaangupta>`_)
- Lars Grüter (`@lagru <https://github.com/lagru>`_)
- Loïc Estève (`@lesteve <https://github.com/lesteve>`_)
- M Bussonnier (`@Carreau <https://github.com/Carreau>`_)
- Marianne Corvellec (`@mkcor <https://github.com/mkcor>`_)
- Mark Harfouche (`@hmaarrfk <https://github.com/hmaarrfk>`_)
- Matthew Feickert (`@matthewfeickert <https://github.com/matthewfeickert>`_)
- Paritosh Dahiya (`@hnhparitosh <https://github.com/hnhparitosh>`_)
- Piyush Amitabh (`@pamitabh <https://github.com/pamitabh>`_)
- Ricky Walsh (`@rickymwalsh <https://github.com/rickymwalsh>`_)
- Stefan van der Walt (`@stefanv <https://github.com/stefanv>`_)

25 reviewers added to this release (alphabetically):

- `@aeisenbarth <https://github.com/aeisenbarth>`_
- `@FedericoWZhaw <https://github.com/FedericoWZhaw>`_
- `@jakirkham <https://github.com/jakirkham>`_
- `@michaelbratsch <https://github.com/michaelbratsch>`_
- Agriya Khetarpal (`@agriyakhetarpal <https://github.com/agriyakhetarpal>`_)
- Brigitta Sipőcz (`@bsipocz <https://github.com/bsipocz>`_)
- Dan Schult (`@dschult <https://github.com/dschult>`_)
- Edgar Andrés Margffoy Tuay (`@andfoy <https://github.com/andfoy>`_)
- Egor Panfilov (`@soupault <https://github.com/soupault>`_)
- Gianluca (`@geeanlooca <https://github.com/geeanlooca>`_)
- Gregory Lee (`@grlee77 <https://github.com/grlee77>`_)
- Hayato Ikoma (`@hayatoikoma <https://github.com/hayatoikoma>`_)
- Jarrod Millman (`@jarrodmillman <https://github.com/jarrodmillman>`_)
- Jordão Bragantini (`@JoOkuma <https://github.com/JoOkuma>`_)
- João Seródio (`@SerodioJ <https://github.com/SerodioJ>`_)
- Juan Nunez-Iglesias (`@jni <https://github.com/jni>`_)
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


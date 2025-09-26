scikit-image 0.24.0 (2024-06-18)
================================

We're happy to announce the release of scikit-image 0.24.0!

Highlights
----------

- Add new class ``skimage.transform.ThinPlateSplineTransform``, which can estimate and model non-linear transforms with thin-plate splines and allows image warping with ``skimage.transform.warp`` (`#7040 <https://github.com/scikit-image/scikit-image/pull/7040>`_).

New Features
------------

- Add new class ``skimage.transform.ThinPlateSplineTransform``, which can estimate and model non-linear transforms with thin-plate splines and allows image warping with ``skimage.transform.warp`` (`#7040 <https://github.com/scikit-image/scikit-image/pull/7040>`_).
- Add ``skimage.morphology.remove_objects_by_distance``, which removes labeled objects, ordered by size (default), until the remaining objects are a given distance apart (`#4165 <https://github.com/scikit-image/scikit-image/pull/4165>`_).

Performance
-----------

- In ``skimage.feature.corner_fast``, test four directions earlier, which should more than half the computation time for most cases (`#7394 <https://github.com/scikit-image/scikit-image/pull/7394>`_).

Documentation
-------------

- Remove obsolete instruction about documenting changes (`#7321 <https://github.com/scikit-image/scikit-image/pull/7321>`_).
- Clarify description of ``data_range`` parameter in ``skimage.metrics.structural_similarity`` (`#7345 <https://github.com/scikit-image/scikit-image/pull/7345>`_).
- Update release process notes (`#7402 <https://github.com/scikit-image/scikit-image/pull/7402>`_).
- Fix typo in docstring of ``skimage.measure.regionprops`` (`#7405 <https://github.com/scikit-image/scikit-image/pull/7405>`_).
- Fix typos in ``skimage.measure.find_contours`` (`#7411 <https://github.com/scikit-image/scikit-image/pull/7411>`_).
- Add algorithmic complexity description + suggested alternatives to ``skimage.restoration.rolling_ball`` docstring (`#7424 <https://github.com/scikit-image/scikit-image/pull/7424>`_).
- Remove ineffective PR contribution clause (`#7429 <https://github.com/scikit-image/scikit-image/pull/7429>`_).
- Clarify objection period for lazy consensus in SKIP 1 (`#7020 <https://github.com/scikit-image/scikit-image/pull/7020>`_).
- Add a new gallery example "Use thin-plate splines for image warping" (`#7040 <https://github.com/scikit-image/scikit-image/pull/7040>`_).
- Add a new gallery example on "Removing objects" based on their size or distance (`#4165 <https://github.com/scikit-image/scikit-image/pull/4165>`_).

Infrastructure
--------------

- Escape user-controlled variables in GA workflow (`#7415 <https://github.com/scikit-image/scikit-image/pull/7415>`_).
- Add generation of GitHub artifact attestations to built sdist and wheels before upload to PyPI (`#7427 <https://github.com/scikit-image/scikit-image/pull/7427>`_).
- For publishing actions use the full length commit SHA (`#7433 <https://github.com/scikit-image/scikit-image/pull/7433>`_).
- Be mindful of resources by canceling in-progress workflows (`#7436 <https://github.com/scikit-image/scikit-image/pull/7436>`_).
- Add out-of-tree Pyodide builds in CI for ``scikit-image`` (`#7350 <https://github.com/scikit-image/scikit-image/pull/7350>`_).

Maintenance
-----------

- Replace deprecated nose style setup/teardown with autouse fixtures (`#7343 <https://github.com/scikit-image/scikit-image/pull/7343>`_).
- Temporarily pin macos-12 runner in CI (`#7408 <https://github.com/scikit-image/scikit-image/pull/7408>`_).
- Fix NumPy2 dtype promotion issues in pywt dependent code (`#7414 <https://github.com/scikit-image/scikit-image/pull/7414>`_).
- In ``skimage.util.compare_images``, deprecate the parameter ``image2``. Instead use ``image0``, ``image1`` to pass the compared images. Furthermore, all other parameters will be turned into keyword-only parameters once the deprecation is complete (`#7322 <https://github.com/scikit-image/scikit-image/pull/7322>`_).
- Add support back for Python 3.9 to enhance compatibility with Numpy 2 (`#7412 <https://github.com/scikit-image/scikit-image/pull/7412>`_).
- Disable ruff/pyupgrade rule UP038 (`#7430 <https://github.com/scikit-image/scikit-image/pull/7430>`_).
- Stop verifying wheel attestations temporarily (`#7444 <https://github.com/scikit-image/scikit-image/pull/7444>`_).

Contributors
------------

13 authors added to this release (alphabetically):

- Adeyemi Biola  (`@decorouz <https://github.com/decorouz>`_)
- Agriya Khetarpal (`@agriyakhetarpal <https://github.com/agriyakhetarpal>`_)
- Ananya Srivastava (`@ana42742 <https://github.com/ana42742>`_)
- Curtis Rueden (`@ctrueden <https://github.com/ctrueden>`_)
- Jarrod Millman (`@jarrodmillman <https://github.com/jarrodmillman>`_)
- Juan Nunez-Iglesias (`@jni <https://github.com/jni>`_)
- Lars Grüter (`@lagru <https://github.com/lagru>`_)
- Marianne Corvellec (`@mkcor <https://github.com/mkcor>`_)
- Mark Harfouche (`@hmaarrfk <https://github.com/hmaarrfk>`_)
- Matthew Feickert (`@matthewfeickert <https://github.com/matthewfeickert>`_)
- Pang (`@lartpang <https://github.com/lartpang>`_)
- Stefan van der Walt (`@stefanv <https://github.com/stefanv>`_)
- 武士风度的牛 (`@spdfghi <https://github.com/spdfghi>`_)

15 reviewers added to this release (alphabetically):

- Adeyemi Biola  (`@decorouz <https://github.com/decorouz>`_)
- Agriya Khetarpal (`@agriyakhetarpal <https://github.com/agriyakhetarpal>`_)
- Curtis Rueden (`@ctrueden <https://github.com/ctrueden>`_)
- Egor Panfilov (`@soupault <https://github.com/soupault>`_)
- Jarrod Millman (`@jarrodmillman <https://github.com/jarrodmillman>`_)
- Juan Nunez-Iglesias (`@jni <https://github.com/jni>`_)
- Lars Grüter (`@lagru <https://github.com/lagru>`_)
- Marianne Corvellec (`@mkcor <https://github.com/mkcor>`_)
- Mark Harfouche (`@hmaarrfk <https://github.com/hmaarrfk>`_)
- Matthew Feickert (`@matthewfeickert <https://github.com/matthewfeickert>`_)
- Ralf Gommers (`@rgommers <https://github.com/rgommers>`_)
- Riadh Fezzani (`@rfezzani <https://github.com/rfezzani>`_)
- Sebastian Berg (`@seberg <https://github.com/seberg>`_)
- Stefan van der Walt (`@stefanv <https://github.com/stefanv>`_)
- Tyler Reddy (`@tylerjereddy <https://github.com/tylerjereddy>`_)

_These lists are automatically generated, and may not be complete or may contain duplicates._

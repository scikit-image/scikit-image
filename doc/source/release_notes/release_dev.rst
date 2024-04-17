scikit-image 0.23.2rc1
======================

We're happy to announce the release of scikit-image 0.23.2rc1!

Bug Fixes
---------

- Make sure ``skimage.util.img_as_ubyte`` supports the edge case where ``dtype('uint64').type`` of the provided image is ``np.ulonglong`` instead of ``np.uint64`` (`#7392 <https://github.com/scikit-image/scikit-image/pull/7392>`_).

Documentation
-------------

- Add date to 0.23.1 release notes (`#7384 <https://github.com/scikit-image/scikit-image/pull/7384>`_).
- Fix docstring of ``connectivity`` parameter in ``skimage.segmentation.watershed`` (`#7360 <https://github.com/scikit-image/scikit-image/pull/7360>`_).

Maintenance
-----------

- Use ``numpy.inf`` instead of deprecated ``numpy.infty`` (`#7386 <https://github.com/scikit-image/scikit-image/pull/7386>`_).
- Update Ruff config (`#7387 <https://github.com/scikit-image/scikit-image/pull/7387>`_).
- Update matrix and names of Azure pipelines configuration (`#7390 <https://github.com/scikit-image/scikit-image/pull/7390>`_).
- Ignore arch specific cast warnings originating from ``astype`` in tests (`#7393 <https://github.com/scikit-image/scikit-image/pull/7393>`_).
- Update link to numpydoc example.py (`#7395 <https://github.com/scikit-image/scikit-image/pull/7395>`_).

Contributors
------------

4 authors added to this release (alphabetically):

- `@pitkajuh <https://github.com/pitkajuh>`_
- Jarrod Millman (`@jarrodmillman <https://github.com/jarrodmillman>`_)
- Lars Grüter (`@lagru <https://github.com/lagru>`_)
- Marianne Corvellec (`@mkcor <https://github.com/mkcor>`_)

3 reviewers added to this release (alphabetically):

- Egor Panfilov (`@soupault <https://github.com/soupault>`_)
- Jarrod Millman (`@jarrodmillman <https://github.com/jarrodmillman>`_)
- Lars Grüter (`@lagru <https://github.com/lagru>`_)

_These lists are automatically generated, and may not be complete or may contain duplicates._

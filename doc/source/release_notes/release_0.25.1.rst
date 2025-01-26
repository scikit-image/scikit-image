scikit-image 0.25.1
===================

We're happy to announce the release of scikit-image 0.25.1!

Bug Fixes
---------

- Include ``centroid`` in ``__all__`` of the PYI file in ``skimage.measure`` (`#7652 <https://github.com/scikit-image/scikit-image/pull/7652>`_).
- Improve numerical stability of ``blur_effect`` (`#7643 <https://github.com/scikit-image/scikit-image/pull/7643>`_).
- Because under-determined fits are unreliable, ``skimage.measure.EllipseModel`` will now warn and return ``False`` (no fit) when fewer than 5 data points are provided (`#7648 <https://github.com/scikit-image/scikit-image/pull/7648>`_).
- Explicitly upcast ``data`` with dtype ``float16`` to ``float32`` in  ``skimage.segmentation.random_walker``; this fixes passing ``float16`` on NumPy 1.26 (`#7655 <https://github.com/scikit-image/scikit-image/pull/7655>`_).

Documentation
-------------

- Don't use removed ``QuadContourSet.collections`` in gallery example (`#7638 <https://github.com/scikit-image/scikit-image/pull/7638>`_).
- Change old import convention in the gallery examples (`#7630 <https://github.com/scikit-image/scikit-image/pull/7630>`_).

Infrastructure
--------------

- Make apigen.py work with editable hooks (`#7647 <https://github.com/scikit-image/scikit-image/pull/7647>`_).
- Build Linux ARM wheels natively (`#7664 <https://github.com/scikit-image/scikit-image/pull/7664>`_).

Maintenance
-----------

- Infer floating point type for sigma parameter (`#7637 <https://github.com/scikit-image/scikit-image/pull/7637>`_).
- In ``skimage.segmentation.active_contour``, change the type of the default argument for ``w_line`` to indicate it is a float (`#7645 <https://github.com/scikit-image/scikit-image/pull/7645>`_).
- Temporarily disable parallel building of gallery (`#7656 <https://github.com/scikit-image/scikit-image/pull/7656>`_).
- [pre-commit.ci] pre-commit autoupdate (`#7649 <https://github.com/scikit-image/scikit-image/pull/7649>`_).
- Skip flaky test on azure (`#7669 <https://github.com/scikit-image/scikit-image/pull/7669>`_).

Contributors
------------

8 authors added to this release (alphabetically):

- `@michaelbratsch <https://github.com/michaelbratsch>`_
- `@scrimpys <https://github.com/scrimpys>`_
- Jarrod Millman (`@jarrodmillman <https://github.com/jarrodmillman>`_)
- Jigyasu (`@jgyasu <https://github.com/jgyasu>`_)
- kwikwag (`@kwikwag <https://github.com/kwikwag>`_)
- Lars Grüter (`@lagru <https://github.com/lagru>`_)
- Marianne Corvellec (`@mkcor <https://github.com/mkcor>`_)
- Stefan van der Walt (`@stefanv <https://github.com/stefanv>`_)

8 reviewers added to this release (alphabetically):

- `@michaelbratsch <https://github.com/michaelbratsch>`_
- Dan Schult (`@dschult <https://github.com/dschult>`_)
- Jarrod Millman (`@jarrodmillman <https://github.com/jarrodmillman>`_)
- Jigyasu (`@jgyasu <https://github.com/jgyasu>`_)
- Lars Grüter (`@lagru <https://github.com/lagru>`_)
- Marianne Corvellec (`@mkcor <https://github.com/mkcor>`_)
- Ruth Comer (`@rcomer <https://github.com/rcomer>`_)
- Stefan van der Walt (`@stefanv <https://github.com/stefanv>`_)

_These lists are automatically generated, and may not be complete or may contain duplicates._


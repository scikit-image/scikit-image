:orphan:

scikit-image 0.X.0 release notes
================================

scikit-image is an image processing library for the scientific Python
ecosystem that includes algorithms for segmentation, geometric
transformations, feature detection, registration, color space
manipulation, analysis, filtering, morphology, and more.

For more information, examples, and documentation, please visit our website:

https://scikit-image.org


New Features
------------

- Add parameters ``mode`` and ``cval`` to ``erosion``, ``dilation``, ``opening``, ``closing``, ``white_tophat``, and ``black_tophat`` in ``skimage.morphology``;
  add parameter ``mode`` to ``binary_erosion``, ``binary_dilation``, ``binary_opening`` and ``binary_closing`` in ``skimage.morphology``;
  add functions ``mirror_footprint`` and ``pad_footprint`` to ``skimage.morphology``;
  (`#6695 <https://github.com/scikit-image/scikit-image/pull/6695>`_).

Improvements
------------



Bugfixes
--------

- ``skimage.morphology.closing`` and ``skimage.morphology.opening`` were not extensive and anti-extensive, respectively, if the footprint was not mirror symmetric
  (`#6695 <https://github.com/scikit-image/scikit-image/pull/6695>`_).

Deprecations
------------

- Parameters ``shift_x`` and ``shift_y`` in ``skimage.morphology.erosion`` and ``skimage.morphology.dilation`` are deprecated and a warning is emitted if they are given.
  (`#6695 <https://github.com/scikit-image/scikit-image/pull/6695>`_).

Contributors to this release
----------------------------

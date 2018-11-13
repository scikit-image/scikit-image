Announcement: scikit-image 0.X.0
================================

We're happy to announce the release of scikit-image v0.X.0!

scikit-image is an image processing toolbox for SciPy that includes algorithms
for segmentation, geometric transformations, color space manipulation,
analysis, filtering, morphology, feature detection, and more.

For more information, examples, and documentation, please visit our website:

http://scikit-image.org


New Features
------------

- New options ``connectivity``, ``indices`` and ``allow_borders`` for
  ``skimage.morphology.local_maxima`` and ``.local_minima``. #3022


Improvements
------------

- ``skivi`` is now using ``qtpy`` for Qt4/Qt5/PySide/PySide2 compatibility (a
  new optional dependency).
- Performance of ``skimage.morphology.local_maxima`` and ``.local_minima`` was
  improved with a new Cython-based implementation. #3022


API Changes
-----------

- ``rectangular_grid`` now returns a tuple instead of a list to improve
  compatibility with NumPy 1.15.
- Parameter ``dynamic_range`` in ``skimage.measure.compare_psnr`` has been
  removed. Use parameter ``data_range`` instead.


Bugfixes
--------

- ``skimage.morphology.local_maxima`` and ``skimage.morphology.local_minima``
  no longer raise an error if any dimension of the image is smaller 3 and
  the keyword ``allow_borders`` was false.


Deprecations
------------

- Python 2 support has been dropped in the development version. Users of the
  development version should have Python >= 3.5.
- ``skimage.util.montage2d`` has been removed. Use ``skimage.util.montage`` instead.
- ``skimage.novice`` is deprecated and will be removed in 0.16.
- ``skimage.transform.resize`` and ``skimage.transform.rescale`` option
  ``anti_aliasing`` has been enabled by default.
- ``regionprops`` will use row-column coordinates in 0.16. You can start
  using them now with ``regionprops(..., coordinates='rc')``. You can silence
  warning messages, and retain the old behavior, with
  ``regionprops(..., coordinates='xy')``. However, that option will go away
  in 0.16 and result in an error. This change has a number of consequences.
  Specifically, the "orientation" region property will measure the
  anticlockwise angle from a *vertical* line, i.e. from the vector (1, 0) in
  row-column coordinates.
- ``skimage.morphology.remove_small_holes`` ``min_size`` argument is deprecated
  and will be removed in 0.16. Use ``area_threshold`` instead.


Contributors to this release
----------------------------

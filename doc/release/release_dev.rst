Announcement: scikit-image 0.X.0
================================

We're happy to announce the release of scikit-image v0.X.0!

scikit-image is an image processing library for the scientific Python
ecosystem that includes algorithms for segmentation, geometric
transformations, feature detection, registration, color space
manipulation, analysis, filtering, morphology, and more.

For more information, examples, and documentation, please visit our website:

https://scikit-image.org


New Features
------------

- Add isotropic versions of binary morphological operators (gh-6492)


Improvements
------------



API Changes
-----------

- All references to EN-GB spelling for the word ``neighbour`` and others—e.g.,
  ``neigbourhood``, ``neighboring``, were changed to their EN-US spelling,
  ``neighbor``. With that, ``skimage.measure.perimeter`` parameter ``neighbourhood``
  was deprecated in favor of ``neighborhood`` in 0.19.2.


Backward Incompatible Changes
-----------------------------

- ``skimage.filters.meijering``, ``skimage.filters.sato``,
  ``skimage.filters.frangi``, and ``skimage.filters.hessian`` have all been
  rewritten to match more closely the published algorithms; the output values
  will be different from previously.  The Hessian matrix calculation is now
  done more accurately.  The filters will now correctly be set to zero whenever
  one of the hessian eigenvalues has a sign incompatible with a ridge of the
  desired polarity.  The gamma constant of the Frangi filter is now set
  adaptively based on the maximum Hessian norm.


Bugfixes
--------



Deprecations
------------

- The function ``filters.inverse`` was deprecated. Please use
  ``filters.filter_inverse``.


Contributors to this release
----------------------------

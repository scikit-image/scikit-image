Announcement: scikit-image 0.X.0
================================

We're happy to announce the release of scikit-image v0.X.0!

scikit-image is an image processing toolbox for SciPy that includes algorithms
for segmentation, geometric transformations, color space manipulation,
analysis, filtering, morphology, feature detection, and more.

For more information, examples, and documentation, please visit our website:

https://scikit-image.org


New Features
------------


Documentation
-------------

- A new doc tutorial presenting a 3D biomedical imaging example has been added
  to the gallery (#4946). The technical content benefited from conversations
  with Genevieve Buckley, Kevin Mader, and Volker Hilsenstein.
- Documentation has been added to the contributing notes about how to submit a
  gallery example 


Improvements
------------

- The performance of the SLIC superpixels algorithm
  (``skimage.segmentation.slice``) was improved for the case where a mask
  is supplied by the user (#4903). The specific superpixels produced by
  masked SLIC will not be identical to those produced by prior releases.
- ``exposure.adjust_gamma`` has been accelerated for ``uint8`` images thanks to a
  LUT (#4966).  
- ``measure.label`` has been accelerated for boolean input images, by using
  ``scipy.ndimage``'s implementation for this case (#4945).
- ``util.apply_parallel`` now works with multichannel data (#4927).
- ``skimage.feature.peak_local_max`` supports now any Minkowski distance.


API Changes
-----------

- A default value has been added to ``measure.find_contours``, corresponding to
  the half distance between the min and max values of the image 
  #4862
- ``data.cat`` has been introduced as an alias of ``data.chelsea`` for a more
  descriptive name.
- The ``level`` parameter of ``measure.find_contours`` is now a keyword
  argument, with a default value set to (max(image) - min(image)) / 2.
- ``p_norm`` argument was added to ``skimage.feature.peak_local_max``
  to add support for Minkowski distances.


Bugfixes
--------

- Fixed the behaviour of Richardson-Lucy deconvolution for images with 3
  dimensions or more (#4823)
- ``min_distance`` is now enforced for ``skimage.feature.peak_local_max``
  (#2592).
- Peak detection in labels is fixed in ``skimage.feature.peak_local_max``
  (#4756).
- Input ``labels`` argument renumbering in ``skimage.feature.peak_local_max``
  is avoided (#5047).

Deprecations
------------

- In ``measure.label``, the deprecated ``neighbors`` parameter has been
  removed.


Development process
-------------------


Contributors to this release
----------------------------

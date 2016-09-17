Announcement: scikit-image 0.9.0
================================

We're happy to announce the release of scikit-image v0.9.0!

scikit-image is an image processing toolbox for SciPy that includes algorithms
for segmentation, geometric transformations, color space manipulation,
analysis, filtering, morphology, feature detection, and more.

For more information, examples, and documentation, please visit our website:

    http://scikit-image.org


New Features
------------

`scikit-image` now runs without translation under both Python 2 and 3.

In addition to several bug fixes, speed improvements and examples, the 204 pull
requests merged for this release include the following new features (PR number
in brackets):

Segmentation:

- 3D support in SLIC segmentation (#546)
- SLIC voxel spacing (#719)
- Generalized anisotropic spacing support for random_walker (#775)
- Yen threshold method (#686)

Transforms and filters:

- SART algorithm for tomography reconstruction (#584)
- Gabor filters (#371)
- Hough transform for ellipses (#597)
- Fast resampling of nD arrays (#511)
- Rotation axis center for Radon transforms with inverses. (#654)
- Reconstruction circle in inverse Radon transform (#567)
- Pixelwise image adjustment curves and methods (#505)

Feature detection:

- [experimental API] BRIEF feature descriptor (#591)
- [experimental API] Censure (STAR) Feature Detector (#668)
- Octagon structural element (#669)
- Add non rotation invariant uniform LBPs (#704)

Color and noise:

- Add deltaE color comparison and lab2lch conversion (#665)
- Isotropic denoising (#653)
- Generator to add various types of random noise to images (#625)
- Color deconvolution for immunohistochemical images (#441)
- Color label visualization (#485)

Drawing and visualization:

- Wu's anti-aliased circle, line, bezier curve (#709)
- Linked image viewers and docked plugins (#575)
- Rotated ellipse + bezier curve drawing (#510)
- PySide & PyQt4 compatibility in skimage-viewer (#551)

Other:

- Python 3 support without 2to3. (#620)
- 3D Marching Cubes (#469)
- Line, Circle, Ellipse total least squares fitting and RANSAC algorithm (#440)
- N-dimensional array padding (#577)
- Add a wrapper around `scipy.ndimage.gaussian_filter` with useful default behaviors. (#712)
- Predefined structuring elements for 3D morphology (#484)


API changes
-----------

The following backward-incompatible API changes were made between 0.8 and 0.9:

- No longer wrap ``imread`` output in an ``Image`` class
- Change default value of `sigma` parameter in ``skimage.segmentation.slic``
  to 0
- ``hough_circle`` now returns a stack of arrays that are the same size as the
  input image. Set the ``full_output`` flag to True for the old behavior.
- The following functions were deprecated over two releases:
  `skimage.filter.denoise_tv_chambolle`,
  `skimage.morphology.is_local_maximum`, `skimage.transform.hough`,
  `skimage.transform.probabilistic_hough`,`skimage.transform.hough_peaks`.
  Their functionality still exists, but under different names.


Contributors to this release
----------------------------

This release was made possible by the collaborative efforts of many
contributors, both new and old.  They are listed in alphabetical order by
surname:

- Ankit Agrawal
- K.-Michael Aye
- Chris Beaumont
- François Boulogne
- Luis Pedro Coelho
- Marianne Corvellec
- Olivier Debeir
- Ferdinand Deger
- Kemal Eren
- Jostein Bø Fløystad
- Christoph Gohlke
- Emmanuelle Gouillart
- Christian Horea
- Thouis (Ray) Jones
- Almar Klein
- Xavier Moles Lopez
- Alexis Mignon
- Juan Nunez-Iglesias
- Zachary Pincus
- Nicolas Pinto
- Davin Potts
- Malcolm Reynolds
- Umesh Sharma
- Johannes Schönberger
- Chintak Sheth
- Kirill Shklovsky
- Steven Silvester
- Matt Terry
- Riaan van den Dool
- Stéfan van der Walt
- Josh Warner
- Adam Wisniewski
- Yang Zetian
- Tony S Yu

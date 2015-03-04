Announcement: scikit-image 0.10.0
=================================

We're happy to announce the release of scikit-image v0.10.0!

scikit-image is an image processing toolbox for SciPy that includes algorithms
for segmentation, geometric transformations, color space manipulation,
analysis, filtering, morphology, feature detection, and more.

For more information, examples, and documentation, please visit our website:

   http://scikit-image.org


New Features
------------

In addition to many bug fixes, (speed) improvements and new examples, the 118
pull requests (1112 commits) merged for this release include the following new
or improved features (PR number in brackets):

- BRIEF, ORB and CENSURE features (#834)
- Blob Detection (#903)
- Phase unwrapping (#644)
- Wiener deconvolution (#800)
- IPython notebooks in examples gallery (#1000)
- Luv colorspace conversion (#798)
- Viewer overlays (#810)
- ISODATA thresholding (#859)
- A new rank filter for summation (#844)
- Faster MCP with anisotropy support (#854)
- N-d peak finding (`peak_local_max`) (#906)
- `imread_collection` support for all plugins (#862)
- Enforce SLIC superpixels connectivity and add SLIC-zero (#857, #864)
- Correct mesh orientation (for use in external visualizers) in
  marching cubes algorithm (#882)
- Loading from URL and alpha support in novice module (#916, #946)
- Equality for regionprops (#956)


API changes
-----------

The following backward-incompatible API changes were made between 0.9 and 0.10:

- Removed deprecated functions in `skimage.filter.rank.*`
- Removed deprecated parameter `epsilon` of `skimage.viewer.LineProfile`
- Removed backwards-compatability of `skimage.measure.regionprops`
- Removed {`ratio`, `sigma`} deprecation warnings of `skimage.segmentation.slic`
  and also remove explicit `sigma` parameter from doc-string example
- Changed default mode of random_walker segmentation to 'cg_mg' > 'cg' > 'bf',
  depending on which optional dependencies are available.
- Removed deprecated `out` parameter of `skimage.morphology.binary_*`
- Removed deprecated parameter `depth` in `skimage.segmentation.random_walker`
- Removed deprecated logger function in `skimage/__init__.py`
- Removed deprecated function `filter.median_filter`
- Removed deprecated `skimage.color.is_gray` and `skimage.color.is_rgb`
  functions
- Removed deprecated `skimage.segmentation.visualize_boundaries`
- Removed deprecated `skimage.morphology.greyscale_*`
- Removed deprecated `skimage.exposure.equalize`


Contributors to this release
----------------------------

This release was made possible by the collaborative efforts of many
contributors, both new and old.  They are listed in alphabetical order by
surname:

- Raphael Ackermann
- Ankit Agrawal
- Maximilian Albert
- Pietro Berkes
- Vighnesh Birodkar
- François Boulogne
- Olivier Debeir
- Christoph Deil
- Jaidev Deshpande
- Jaime Frio
- Jostein Bø Fløystad
- Neeraj Gangwar
- Christoph Gohlke
- Michael Hansen
- Almar Klein
- Jeremy Metz
- Juan Nunez-Iglesias
- François Orieux
- Guillem Palou
- Rishabh Raj
- Thomas Robitaille
- Michal Romaniuk
- Johannes L. Schönberger
- Steven Silvester
- Julian Taylor
- Gregor Thalhammer
- Matthew Trentacoste
- Siva Prasad Varma
- Guillem Palou Visa
- Stefan van der Walt
- Joshua Warner
- Tony S Yu
- radioxoma

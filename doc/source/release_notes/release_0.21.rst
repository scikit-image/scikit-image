scikit-image 0.21.0rc1 release notes
====================================

scikit-image is an image processing toolbox for SciPy that includes algorithms
for segmentation, geometric transformations, color space manipulation,
analysis, filtering, morphology, feature detection, and more.

For more information, examples, and documentation, please visit our website:

https://scikit-image.org

Highlights
----------
- Last release to support Python 3.8
- Unified API for PRNGs

Deprecations
------------
- Expose `color.get_xyz_coords` in public API
  (`#6696 <https://github.com/scikit-image/scikit-image/pull/6696>`_).
- Unify pseudo-random seeding interface
  (`#6922 <https://github.com/scikit-image/scikit-image/pull/6922>`_).


Merged Pull Requests
--------------------
- Implement Fisher vectors in scikit-image
  (`#5349 <https://github.com/scikit-image/scikit-image/pull/5349>`_).
- Bounding box crop
  (`#5499 <https://github.com/scikit-image/scikit-image/pull/5499>`_).
- Unify API on seed keyword for random seeds / generator
  (`#6258 <https://github.com/scikit-image/scikit-image/pull/6258>`_).
- Fix and refactor `deprecated` decorator to `deprecate_func`
  (`#6594 <https://github.com/scikit-image/scikit-image/pull/6594>`_).
- Refactor `_invariant_denoise` to `denoise_invariant`
  (`#6660 <https://github.com/scikit-image/scikit-image/pull/6660>`_).
- Document boundary behavior of `draw.polygon` and `draw.polygon2mask`
  (`#6690 <https://github.com/scikit-image/scikit-image/pull/6690>`_).
- Expose `color.get_xyz_coords` in public API
  (`#6696 <https://github.com/scikit-image/scikit-image/pull/6696>`_).
- shift and normalize data before fitting circle or ellipse
  (`#6703 <https://github.com/scikit-image/scikit-image/pull/6703>`_).
- Showcase pydata-sphinx-theme
  (`#6714 <https://github.com/scikit-image/scikit-image/pull/6714>`_).
- Fix matrix calculation for shear angle in `AffineTransform`
  (`#6717 <https://github.com/scikit-image/scikit-image/pull/6717>`_).
- Fix threshold_li(): prevent log(0) on single-value background.
  (`#6745 <https://github.com/scikit-image/scikit-image/pull/6745>`_).
- Add support for y-dimensional shear to the AffineTransform
  (`#6752 <https://github.com/scikit-image/scikit-image/pull/6752>`_).
- allow trivial ransac call
  (`#6755 <https://github.com/scikit-image/scikit-image/pull/6755>`_).
- Fix copy-paste error in `footprints.diamond` test case
  (`#6756 <https://github.com/scikit-image/scikit-image/pull/6756>`_).
- Use imageio v3 API
  (`#6764 <https://github.com/scikit-image/scikit-image/pull/6764>`_).
- Merge duplicate instructions for setting up build environment.
  (`#6770 <https://github.com/scikit-image/scikit-image/pull/6770>`_).
- Prepare CI configuration for merge queue
  (`#6771 <https://github.com/scikit-image/scikit-image/pull/6771>`_).
- Unpin scipy dependency
  (`#6773 <https://github.com/scikit-image/scikit-image/pull/6773>`_).
- Add docstring to `skimage.color` module
  (`#6777 <https://github.com/scikit-image/scikit-image/pull/6777>`_).
- DOC: Fix underline length in `docstring_add_deprecated`
  (`#6778 <https://github.com/scikit-image/scikit-image/pull/6778>`_).
- Link full license to README
  (`#6779 <https://github.com/scikit-image/scikit-image/pull/6779>`_).
- Fix conda instructions for dev env setup.
  (`#6781 <https://github.com/scikit-image/scikit-image/pull/6781>`_).
- Update docstring in skimage.future module
  (`#6782 <https://github.com/scikit-image/scikit-image/pull/6782>`_).
- Make join_segmentations return array maps from output to input labels
  (`#6786 <https://github.com/scikit-image/scikit-image/pull/6786>`_).
- Remove outdated build instructions from README
  (`#6788 <https://github.com/scikit-image/scikit-image/pull/6788>`_).
- Update .devpy/cmds.py to match latest devpy
  (`#6789 <https://github.com/scikit-image/scikit-image/pull/6789>`_).
- Avoid installation of rtoml via conda in installation guide
  (`#6792 <https://github.com/scikit-image/scikit-image/pull/6792>`_).
- Relicense CLAHE code under BSD-3-Clause
  (`#6795 <https://github.com/scikit-image/scikit-image/pull/6795>`_).
- Add docstring to the `transform` module
  (`#6797 <https://github.com/scikit-image/scikit-image/pull/6797>`_).
- Raise error in skeletonize for invalid value to method param
  (`#6805 <https://github.com/scikit-image/scikit-image/pull/6805>`_).
- Handle pip-only dependencies when using conda.
  (`#6806 <https://github.com/scikit-image/scikit-image/pull/6806>`_).
- Pin to devpy 0.1 tag
  (`#6816 <https://github.com/scikit-image/scikit-image/pull/6816>`_).
- Relax reproduce section in bug issue template
  (`#6825 <https://github.com/scikit-image/scikit-image/pull/6825>`_).
- Added examples to the EssentialMatrixTransform class and its estimation function
  (`#6832 <https://github.com/scikit-image/scikit-image/pull/6832>`_).
- Sign error fix in measure.regionprops for orientations of 45 degrees
  (`#6836 <https://github.com/scikit-image/scikit-image/pull/6836>`_).
- Fix returned data type in `segmentation.watershed`
  (`#6839 <https://github.com/scikit-image/scikit-image/pull/6839>`_).
- Rename devpy to spin
  (`#6842 <https://github.com/scikit-image/scikit-image/pull/6842>`_).
- Use lazy loader 0.2
  (`#6844 <https://github.com/scikit-image/scikit-image/pull/6844>`_).
- Cleanup cruft in tools
  (`#6846 <https://github.com/scikit-image/scikit-image/pull/6846>`_).
- Speed up threshold_local function by fixing call to _supported_float_type
  (`#6847 <https://github.com/scikit-image/scikit-image/pull/6847>`_).
- Specify kernel for ipywidgets
  (`#6849 <https://github.com/scikit-image/scikit-image/pull/6849>`_).
- Handle NaNs when clipping in `transform.resize`
  (`#6852 <https://github.com/scikit-image/scikit-image/pull/6852>`_).
- Update references to outdated dev.py with spin
  (`#6856 <https://github.com/scikit-image/scikit-image/pull/6856>`_).
- Added example to AffineTransform class
  (`#6859 <https://github.com/scikit-image/scikit-image/pull/6859>`_).
- Fix failing regionprop_table for multichannel properties
  (`#6861 <https://github.com/scikit-image/scikit-image/pull/6861>`_).
- Update _warps_cy.pyx
  (`#6867 <https://github.com/scikit-image/scikit-image/pull/6867>`_).
- Bump 0.21 removals to 0.22
  (`#6868 <https://github.com/scikit-image/scikit-image/pull/6868>`_).
- Update dependencies
  (`#6869 <https://github.com/scikit-image/scikit-image/pull/6869>`_).
- Update pre-commits
  (`#6870 <https://github.com/scikit-image/scikit-image/pull/6870>`_).
- Add test for radon transform on circular phantom
  (`#6873 <https://github.com/scikit-image/scikit-image/pull/6873>`_).
- Do not allow 64-bit integer inputs; add test to ensure masked and unmasked modes are aligned
  (`#6875 <https://github.com/scikit-image/scikit-image/pull/6875>`_).
- Don't use mutable types as default values for arguments
  (`#6876 <https://github.com/scikit-image/scikit-image/pull/6876>`_).
- Fix typo in apply_parallel introduced in #6876
  (`#6881 <https://github.com/scikit-image/scikit-image/pull/6881>`_).
- Point `version_switcher.json` URL at dev docs
  (`#6882 <https://github.com/scikit-image/scikit-image/pull/6882>`_).
- Fix LPI filter for data with even dimensions
  (`#6883 <https://github.com/scikit-image/scikit-image/pull/6883>`_).
- Add back parallel tests that were removed as part of Meson build
  (`#6884 <https://github.com/scikit-image/scikit-image/pull/6884>`_).
- Use legacy datasets without creating a `data_dir`
  (`#6886 <https://github.com/scikit-image/scikit-image/pull/6886>`_).
- Remove `codecov` dependency which disappeared from PyPI
  (`#6887 <https://github.com/scikit-image/scikit-image/pull/6887>`_).
- Add CircleCI API token; fixes status link to built docs
  (`#6894 <https://github.com/scikit-image/scikit-image/pull/6894>`_).
- Fix docstring underline lengths
  (`#6895 <https://github.com/scikit-image/scikit-image/pull/6895>`_).
- Raise error when source_range is not correct
  (`#6898 <https://github.com/scikit-image/scikit-image/pull/6898>`_).
- Remove old doc cruft
  (`#6901 <https://github.com/scikit-image/scikit-image/pull/6901>`_).
- Corrected energy calculation in chan vese
  (`#6902 <https://github.com/scikit-image/scikit-image/pull/6902>`_).
- Temporarily pin imageio to <2.28
  (`#6909 <https://github.com/scikit-image/scikit-image/pull/6909>`_).
- Enable use of `rescale_intensity` with dask array
  (`#6910 <https://github.com/scikit-image/scikit-image/pull/6910>`_).
- Add missing backticks to DOI role in docstring of `area_opening`
  (`#6913 <https://github.com/scikit-image/scikit-image/pull/6913>`_).
- Add PR links to release notes generating script
  (`#6917 <https://github.com/scikit-image/scikit-image/pull/6917>`_).
- Unify pseudo-random seeding interface
  (`#6922 <https://github.com/scikit-image/scikit-image/pull/6922>`_).
- Unify pseudo-random seeding interface follow-up
  (`#6924 <https://github.com/scikit-image/scikit-image/pull/6924>`_).
- Add 0.21 release notes
  (`#6925 <https://github.com/scikit-image/scikit-image/pull/6925>`_).
- Simplify installation instruction document
  (`#6927 <https://github.com/scikit-image/scikit-image/pull/6927>`_).
- Use official meson-python release
  (`#6928 <https://github.com/scikit-image/scikit-image/pull/6928>`_).

28 authors added to this release [alphabetical by first name or login]
----------------------------------------------------------------------
- Adam J. Stewart
- Adeyemi Biola
- aeisenbarth (aeisenbarth)
- Ananya Srivastava
- Bohumír Zámečník
- Carlos Horn
- Daniel Angelov
- DavidTorpey (DavidTorpey)
- Dipkumar Patel
- Eric Prestat
- GGoussar (GGoussar)
- Gregory Lee
- harshitha kolipaka
- Hayato Ikoma
- i-aki-y (i-aki-y)
- Jake Martin
- Jarrod Millman
- Juan Nunez-Iglesias
- Kevin MEETOOA
- Lars Grüter
- mahamtariq58 (mahamtariq58)
- Marianne Corvellec
- Mark Harfouche
- Matthias Bussonnier
- Michael Görner
- Ramyashri Padmanabhakumar
- scott-vsi (scott-vsi)
- Stefan van der Walt


19 reviewers added to this release [alphabetical by first name or login]
------------------------------------------------------------------------
- Adeyemi Biola
- aeisenbarth
- Ananya Srivastava
- Carlos Horn
- DavidTorpey
- Dipkumar Patel
- Gregory Lee
- Henry Pinkard
- i-aki-y
- Jarrod Millman
- Juan Nunez-Iglesias
- Kevin MEETOOA
- Lars Grüter
- Marianne Corvellec
- Mark Harfouche
- Ramyashri Padmanabhakumar
- Riadh Fezzani
- Stefan van der Walt

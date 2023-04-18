scikit-image 0.21.0 release notes (unreleased)
==============================================

We're happy to announce the release of scikit-image v0.21.0 (unreleased)!

scikit-image is an image processing library for the scientific Python
ecosystem that includes algorithms for segmentation, geometric
transformations, feature detection, registration, color space
manipulation, analysis, filtering, morphology, and more.

For more information, examples, and documentation, please visit our website:

https://scikit-image.org

Highlights
----------

- Last release to support Python 3.8
- Unified API for PRNGs

Merged Pull Requests
--------------------

- Implement Fisher vectors in scikit-image (#5349)
- Bounding box crop (#5499)
- Unify API on seed keyword for random seeds / generator (#6258)
- Fix and refactor `deprecated` decorator to `deprecate_func` (#6594)
- Refactor `_invariant_denoise` to `denoise_invariant` (#6660)
- Document boundary behavior of `draw.polygon` and `draw.polygon2mask` (#6690)
- Expose `color.get_xyz_coords` in public API (#6696)
- shift and normalize data before fitting circle or ellipse (#6703)
- Showcase pydata-sphinx-theme (#6714)
- Fix matrix calculation for shear angle in `AffineTransform` (#6717)
- Fix threshold_li(): prevent log(0) on single-value background. (#6745)
- Add support for y-dimensional shear to the AffineTransform (#6752)
- allow trivial ransac call (#6755)
- Fix copy-paste error in `footprints.diamond` test case (#6756)
- Merge duplicate instructions for setting up build environment. (#6770)
- Prepare CI configuration for merge queue (#6771)
- Unpin scipy dependency (#6773)
- Add docstring to `skimage.color` module (#6777)
- DOC: Fix underline length in `docstring_add_deprecated` (#6778)
- Link full license to README (#6779)
- Fix conda instructions for dev env setup. (#6781)
- Update docstring in skimage.future module (#6782)
- Make join_segmentations return array maps from output to input labels (#6786)
- Remove outdated build instructions from README (#6788)
- Update .devpy/cmds.py to match latest devpy (#6789)
- Avoid installation of rtoml via conda in installation guide (#6792)
- Relicense CLAHE code under BSD-3-Clause (#6795)
- Add docstring to the `transform` module (#6797)
- Raise error in skeletonize for invalid value to method param (#6805)
- Handle pip-only dependencies when using conda. (#6806)
- Pin to devpy 0.1 tag (#6816)
- Relax reproduce section in bug issue template (#6825)
- Added examples to the EssentialMatrixTransform class and its estimation function (#6832)
- Sign error fix in measure.regionprops for orientations of 45 degrees (#6836)
- Fix returned data type in `segmentation.watershed` (#6839)
- Rename devpy to spin (#6842)
- Use lazy loader 0.2 (#6844)
- Cleanup cruft in tools (#6846)
- Speed up threshold_local function by fixing call to _supported_float_type (#6847)
- Specify kernel for ipywidgets (#6849)
- Handle NaNs when clipping in `transform.resize` (#6852)
- Update references to outdated dev.py with spin (#6856)
- Added example to AffineTransform class (#6859)
- Fix failing regionprop_table for multichannel properties (#6861)
- Update _warps_cy.pyx (#6867)
- Bump 0.21 removals to 0.22 (#6868)
- Update dependencies (#6869)
- Update pre-commits (#6870)
- Add test for radon transform on circular phantom (#6873)
- Do not allow 64-bit integer inputs; add test to ensure masked and unmasked modes are aligned (#6875)
- Don't use mutable types as default values for arguments (#6876)
- Fix typo in apply_parallel introduced in #6876 (#6881)
- Point `version_switcher.json` URL at dev docs (#6882)
- Fix LPI filter for data with even dimensions (#6883)
- Add back parallel tests that were removed as part of Meson build (#6884)
- Use legacy datasets without creating a `data_dir` (#6886)
- Remove `codecov` dependency which disappeared from PyPI (#6887)
- Add CircleCI API token; fixes status link to built docs (#6894)
- Fix docstring underline lengths (#6895)

25 authors added to this release [alphabetical by first name or login]
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
- Gregory Lee
- harshitha kolipaka
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

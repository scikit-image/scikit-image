scikit-image 0.21.0 (2023-06-02)
================================

We're happy to announce the release of scikit-image 0.21.0!
scikit-image is an image processing toolbox for SciPy that includes algorithms
for segmentation, geometric transformations, color space manipulation,
analysis, filtering, morphology, feature detection, and more.

For more information, examples, and documentation, please visit our website:
https://scikit-image.org

Highlights
----------
- Last release to support Python 3.8
- Unified API for PRNGs

New Features
------------
- Implement Fisher vectors in scikit-image
  (`#5349 <https://github.com/scikit-image/scikit-image/pull/5349>`_).
- Add support for y-dimensional shear to the AffineTransform
  (`#6752 <https://github.com/scikit-image/scikit-image/pull/6752>`_).

API Changes
-----------
In this release, we unify the way seeds are specified for algorithms that make use of
pseudo-random numbers. Before, various keyword arguments (``sample_seed``, ``seed``,
``random_seed``, and ``random_state``) served the same purpose in different places.
These have all been replaced with a single ``rng`` argument, that handles both integer
seeds and NumPy Generators. Please see the related `SciPy discussion`_, as well as
`Scientific Python SPEC 7`_ that attempts to summarize the argument.

.. _SciPy discussion: https://github.com/scipy/scipy/issues/14322
.. _Scientific Python SPEC 7: https://github.com/scientific-python/specs/pull/180

- Unify API on seed keyword for random seeds / generator
  (`#6258 <https://github.com/scikit-image/scikit-image/pull/6258>`_).
- Refactor ``_invariant_denoise`` to ``denoise_invariant``
  (`#6660 <https://github.com/scikit-image/scikit-image/pull/6660>`_).
- Expose ``color.get_xyz_coords`` in public API
  (`#6696 <https://github.com/scikit-image/scikit-image/pull/6696>`_).
- Make join_segmentations return array maps from output to input labels
  (`#6786 <https://github.com/scikit-image/scikit-image/pull/6786>`_).
- Unify pseudo-random seeding interface
  (`#6922 <https://github.com/scikit-image/scikit-image/pull/6922>`_).
- Change geometric transform inverse to property
  (`#6926 <https://github.com/scikit-image/scikit-image/pull/6926>`_).

Enhancements
------------
- Bounding box crop
  (`#5499 <https://github.com/scikit-image/scikit-image/pull/5499>`_).
- Add support for y-dimensional shear to the AffineTransform
  (`#6752 <https://github.com/scikit-image/scikit-image/pull/6752>`_).
- Make join_segmentations return array maps from output to input labels
  (`#6786 <https://github.com/scikit-image/scikit-image/pull/6786>`_).
- Check if ``spacing`` parameter is tuple in ``regionprops``
  (`#6907 <https://github.com/scikit-image/scikit-image/pull/6907>`_).
- Enable use of ``rescale_intensity`` with dask array
  (`#6910 <https://github.com/scikit-image/scikit-image/pull/6910>`_).

Performance
-----------
- Add lazy loading to skimage.color submodule
  (`#6967 <https://github.com/scikit-image/scikit-image/pull/6967>`_).
- Add Lazy loading to skimage.draw submodule
  (`#6971 <https://github.com/scikit-image/scikit-image/pull/6971>`_).
- Add Lazy loader to skimage.exposure
  (`#6978 <https://github.com/scikit-image/scikit-image/pull/6978>`_).
- Add lazy loading to skimage.future module
  (`#6981 <https://github.com/scikit-image/scikit-image/pull/6981>`_).

Bug Fixes
---------
- Fix and refactor ``deprecated`` decorator to ``deprecate_func``
  (`#6594 <https://github.com/scikit-image/scikit-image/pull/6594>`_).
- Refactor ``_invariant_denoise`` to ``denoise_invariant``
  (`#6660 <https://github.com/scikit-image/scikit-image/pull/6660>`_).
- Expose ``color.get_xyz_coords`` in public API
  (`#6696 <https://github.com/scikit-image/scikit-image/pull/6696>`_).
- shift and normalize data before fitting circle or ellipse
  (`#6703 <https://github.com/scikit-image/scikit-image/pull/6703>`_).
- Showcase pydata-sphinx-theme
  (`#6714 <https://github.com/scikit-image/scikit-image/pull/6714>`_).
- Fix matrix calculation for shear angle in ``AffineTransform``
  (`#6717 <https://github.com/scikit-image/scikit-image/pull/6717>`_).
- Fix threshold_li(): prevent log(0) on single-value background.
  (`#6745 <https://github.com/scikit-image/scikit-image/pull/6745>`_).
- Fix copy-paste error in ``footprints.diamond`` test case
  (`#6756 <https://github.com/scikit-image/scikit-image/pull/6756>`_).
- Update .devpy/cmds.py to match latest devpy
  (`#6789 <https://github.com/scikit-image/scikit-image/pull/6789>`_).
- Avoid installation of rtoml via conda in installation guide
  (`#6792 <https://github.com/scikit-image/scikit-image/pull/6792>`_).
- Raise error in skeletonize for invalid value to method param
  (`#6805 <https://github.com/scikit-image/scikit-image/pull/6805>`_).
- Sign error fix in measure.regionprops for orientations of 45 degrees
  (`#6836 <https://github.com/scikit-image/scikit-image/pull/6836>`_).
- Fix returned data type in ``segmentation.watershed``
  (`#6839 <https://github.com/scikit-image/scikit-image/pull/6839>`_).
- Handle NaNs when clipping in ``transform.resize``
  (`#6852 <https://github.com/scikit-image/scikit-image/pull/6852>`_).
- Fix failing regionprop_table for multichannel properties
  (`#6861 <https://github.com/scikit-image/scikit-image/pull/6861>`_).
- Do not allow 64-bit integer inputs; add test to ensure masked and unmasked modes are aligned
  (`#6875 <https://github.com/scikit-image/scikit-image/pull/6875>`_).
- Fix typo in apply_parallel introduced in #6876
  (`#6881 <https://github.com/scikit-image/scikit-image/pull/6881>`_).
- Fix LPI filter for data with even dimensions
  (`#6883 <https://github.com/scikit-image/scikit-image/pull/6883>`_).
- Use legacy datasets without creating a ``data_dir``
  (`#6886 <https://github.com/scikit-image/scikit-image/pull/6886>`_).
- Raise error when source_range is not correct
  (`#6898 <https://github.com/scikit-image/scikit-image/pull/6898>`_).
- apply spacing rescaling when computing centroid_weighted
  (`#6900 <https://github.com/scikit-image/scikit-image/pull/6900>`_).
- Corrected energy calculation in Chan Vese
  (`#6902 <https://github.com/scikit-image/scikit-image/pull/6902>`_).
- Add missing backticks to DOI role in docstring of ``area_opening``
  (`#6913 <https://github.com/scikit-image/scikit-image/pull/6913>`_).
- Fix inclusion of ``random.js`` in HTML output
  (`#6935 <https://github.com/scikit-image/scikit-image/pull/6935>`_).
- Fix URL of random gallery links
  (`#6937 <https://github.com/scikit-image/scikit-image/pull/6937>`_).
- Use context manager to ensure urlopen buffer is closed
  (`#6942 <https://github.com/scikit-image/scikit-image/pull/6942>`_).
- Fix sparse index type casting in skimage.graph._ncut
  (`#6975 <https://github.com/scikit-image/scikit-image/pull/6975>`_).

Maintenance
-----------
- Fix and refactor ``deprecated`` decorator to ``deprecate_func``
  (`#6594 <https://github.com/scikit-image/scikit-image/pull/6594>`_).
- allow trivial ransac call
  (`#6755 <https://github.com/scikit-image/scikit-image/pull/6755>`_).
- Fix copy-paste error in ``footprints.diamond`` test case
  (`#6756 <https://github.com/scikit-image/scikit-image/pull/6756>`_).
- Use imageio v3 API
  (`#6764 <https://github.com/scikit-image/scikit-image/pull/6764>`_).
- Unpin scipy dependency
  (`#6773 <https://github.com/scikit-image/scikit-image/pull/6773>`_).
- Update .devpy/cmds.py to match latest devpy
  (`#6789 <https://github.com/scikit-image/scikit-image/pull/6789>`_).
- Relicense CLAHE code under BSD-3-Clause
  (`#6795 <https://github.com/scikit-image/scikit-image/pull/6795>`_).
- Relax reproduce section in bug issue template
  (`#6825 <https://github.com/scikit-image/scikit-image/pull/6825>`_).
- Rename devpy to spin
  (`#6842 <https://github.com/scikit-image/scikit-image/pull/6842>`_).
- Speed up threshold_local function by fixing call to _supported_float_type
  (`#6847 <https://github.com/scikit-image/scikit-image/pull/6847>`_).
- Specify kernel for ipywidgets
  (`#6849 <https://github.com/scikit-image/scikit-image/pull/6849>`_).
- Make ``image_fetcher`` and ``create_image_fetcher`` in ``data`` private
  (`#6855 <https://github.com/scikit-image/scikit-image/pull/6855>`_).
- Update references to outdated dev.py with spin
  (`#6856 <https://github.com/scikit-image/scikit-image/pull/6856>`_).
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
- Point ``version_switcher.json`` URL at dev docs
  (`#6882 <https://github.com/scikit-image/scikit-image/pull/6882>`_).
- Add back parallel tests that were removed as part of Meson build
  (`#6884 <https://github.com/scikit-image/scikit-image/pull/6884>`_).
- Use legacy datasets without creating a ``data_dir``
  (`#6886 <https://github.com/scikit-image/scikit-image/pull/6886>`_).
- Remove old doc cruft
  (`#6901 <https://github.com/scikit-image/scikit-image/pull/6901>`_).
- Temporarily pin imageio to <2.28
  (`#6909 <https://github.com/scikit-image/scikit-image/pull/6909>`_).
- Unify pseudo-random seeding interface follow-up
  (`#6924 <https://github.com/scikit-image/scikit-image/pull/6924>`_).
- Use pytest.warn instead of custom context manager
  (`#6931 <https://github.com/scikit-image/scikit-image/pull/6931>`_).
- Follow-up to move to pydata-sphinx-theme
  (`#6933 <https://github.com/scikit-image/scikit-image/pull/6933>`_).
- Mark functions as ``noexcept`` to support Cython 3
  (`#6936 <https://github.com/scikit-image/scikit-image/pull/6936>`_).
- Skip unstable test in ``ransac``'s docstring
  (`#6938 <https://github.com/scikit-image/scikit-image/pull/6938>`_).
- Stabilize EllipseModel fitting parameters
  (`#6943 <https://github.com/scikit-image/scikit-image/pull/6943>`_).
- Point logo in generated HTML docs at scikit-image.org
  (`#6947 <https://github.com/scikit-image/scikit-image/pull/6947>`_).
- If user provides RNG, spawn it before deepcopying
  (`#6948 <https://github.com/scikit-image/scikit-image/pull/6948>`_).
- Skip ransac doctest
  (`#6953 <https://github.com/scikit-image/scikit-image/pull/6953>`_).
- Expose ``GeometricTransform.residuals`` in HTML doc
  (`#6968 <https://github.com/scikit-image/scikit-image/pull/6968>`_).
- Fix NumPy 1.25 deprecation warnings
  (`#6969 <https://github.com/scikit-image/scikit-image/pull/6969>`_).
- Revert jupyterlite
  (`#6972 <https://github.com/scikit-image/scikit-image/pull/6972>`_).
- Don't test numpy nightlies due to transcendental functions issue
  (`#6973 <https://github.com/scikit-image/scikit-image/pull/6973>`_).
- Ignore tight layout warning from matplotlib pre-release
  (`#6976 <https://github.com/scikit-image/scikit-image/pull/6976>`_).
- Remove temporary constraint <2.28 for imageio
  (`#6980 <https://github.com/scikit-image/scikit-image/pull/6980>`_).

Documentation
-------------
- Document boundary behavior of ``draw.polygon`` and ``draw.polygon2mask``
  (`#6690 <https://github.com/scikit-image/scikit-image/pull/6690>`_).
- Showcase pydata-sphinx-theme
  (`#6714 <https://github.com/scikit-image/scikit-image/pull/6714>`_).
- Merge duplicate instructions for setting up build environment.
  (`#6770 <https://github.com/scikit-image/scikit-image/pull/6770>`_).
- Add docstring to ``skimage.color`` module
  (`#6777 <https://github.com/scikit-image/scikit-image/pull/6777>`_).
- DOC: Fix underline length in ``docstring_add_deprecated``
  (`#6778 <https://github.com/scikit-image/scikit-image/pull/6778>`_).
- Link full license to README
  (`#6779 <https://github.com/scikit-image/scikit-image/pull/6779>`_).
- Fix conda instructions for dev env setup.
  (`#6781 <https://github.com/scikit-image/scikit-image/pull/6781>`_).
- Update docstring in skimage.future module
  (`#6782 <https://github.com/scikit-image/scikit-image/pull/6782>`_).
- Remove outdated build instructions from README
  (`#6788 <https://github.com/scikit-image/scikit-image/pull/6788>`_).
- Add docstring to the ``transform`` module
  (`#6797 <https://github.com/scikit-image/scikit-image/pull/6797>`_).
- Handle pip-only dependencies when using conda.
  (`#6806 <https://github.com/scikit-image/scikit-image/pull/6806>`_).
- Added examples to the EssentialMatrixTransform class and its estimation function
  (`#6832 <https://github.com/scikit-image/scikit-image/pull/6832>`_).
- Fix returned data type in ``segmentation.watershed``
  (`#6839 <https://github.com/scikit-image/scikit-image/pull/6839>`_).
- Update references to outdated dev.py with spin
  (`#6856 <https://github.com/scikit-image/scikit-image/pull/6856>`_).
- Added example to AffineTransform class
  (`#6859 <https://github.com/scikit-image/scikit-image/pull/6859>`_).
- Update _warps_cy.pyx
  (`#6867 <https://github.com/scikit-image/scikit-image/pull/6867>`_).
- Point ``version_switcher.json`` URL at dev docs
  (`#6882 <https://github.com/scikit-image/scikit-image/pull/6882>`_).
- Fix docstring underline lengths
  (`#6895 <https://github.com/scikit-image/scikit-image/pull/6895>`_).
- ENH Add JupyterLite button to gallery examples
  (`#6911 <https://github.com/scikit-image/scikit-image/pull/6911>`_).
- Add missing backticks to DOI role in docstring of ``area_opening``
  (`#6913 <https://github.com/scikit-image/scikit-image/pull/6913>`_).
- Add 0.21 release notes
  (`#6925 <https://github.com/scikit-image/scikit-image/pull/6925>`_).
- Simplify installation instruction document
  (`#6927 <https://github.com/scikit-image/scikit-image/pull/6927>`_).
- Follow-up to move to pydata-sphinx-theme
  (`#6933 <https://github.com/scikit-image/scikit-image/pull/6933>`_).
- Update release notes
  (`#6944 <https://github.com/scikit-image/scikit-image/pull/6944>`_).
- MNT Fix typo in JupyterLite comment
  (`#6945 <https://github.com/scikit-image/scikit-image/pull/6945>`_).
- Point logo in generated HTML docs at scikit-image.org
  (`#6947 <https://github.com/scikit-image/scikit-image/pull/6947>`_).
- Add missing PRs to release notes
  (`#6949 <https://github.com/scikit-image/scikit-image/pull/6949>`_).
- fix bad link in CODE_OF_CONDUCT.md
  (`#6952 <https://github.com/scikit-image/scikit-image/pull/6952>`_).
- Expose ``GeometricTransform.residuals`` in HTML doc
  (`#6968 <https://github.com/scikit-image/scikit-image/pull/6968>`_).

Infrastructure
--------------
- Showcase pydata-sphinx-theme
  (`#6714 <https://github.com/scikit-image/scikit-image/pull/6714>`_).
- Prepare CI configuration for merge queue
  (`#6771 <https://github.com/scikit-image/scikit-image/pull/6771>`_).
- Pin to devpy 0.1 tag
  (`#6816 <https://github.com/scikit-image/scikit-image/pull/6816>`_).
- Relax reproduce section in bug issue template
  (`#6825 <https://github.com/scikit-image/scikit-image/pull/6825>`_).
- Rename devpy to spin
  (`#6842 <https://github.com/scikit-image/scikit-image/pull/6842>`_).
- Use lazy loader 0.2
  (`#6844 <https://github.com/scikit-image/scikit-image/pull/6844>`_).
- Cleanup cruft in tools
  (`#6846 <https://github.com/scikit-image/scikit-image/pull/6846>`_).
- Update pre-commits
  (`#6870 <https://github.com/scikit-image/scikit-image/pull/6870>`_).
- Remove ``codecov`` dependency which disappeared from PyPI
  (`#6887 <https://github.com/scikit-image/scikit-image/pull/6887>`_).
- Add CircleCI API token; fixes status link to built docs
  (`#6894 <https://github.com/scikit-image/scikit-image/pull/6894>`_).
- Temporarily pin imageio to <2.28
  (`#6909 <https://github.com/scikit-image/scikit-image/pull/6909>`_).
- Add PR links to release notes generating script
  (`#6917 <https://github.com/scikit-image/scikit-image/pull/6917>`_).
- Use official meson-python release
  (`#6928 <https://github.com/scikit-image/scikit-image/pull/6928>`_).
- Fix inclusion of ``random.js`` in HTML output
  (`#6935 <https://github.com/scikit-image/scikit-image/pull/6935>`_).
- Fix URL of random gallery links
  (`#6937 <https://github.com/scikit-image/scikit-image/pull/6937>`_).
- Respect SPHINXOPTS and add --install-deps flags to ``spin docs``
  (`#6940 <https://github.com/scikit-image/scikit-image/pull/6940>`_).
- Build skimage before generating docs
  (`#6946 <https://github.com/scikit-image/scikit-image/pull/6946>`_).
- Enable testing against nightly upstream wheels
  (`#6956 <https://github.com/scikit-image/scikit-image/pull/6956>`_).
- Add nightly wheel builder
  (`#6957 <https://github.com/scikit-image/scikit-image/pull/6957>`_).
- Run weekly tests on nightly wheels
  (`#6959 <https://github.com/scikit-image/scikit-image/pull/6959>`_).
- CI: ensure that a "type: " label is present on each PR
  (`#6960 <https://github.com/scikit-image/scikit-image/pull/6960>`_).
- Add PR milestone labeler
  (`#6977 <https://github.com/scikit-image/scikit-image/pull/6977>`_).

33 authors added to this release (alphabetical)
-----------------------------------------------

- `Adam J. Stewart (@adamjstewart) <https://github.com/scikit-image/scikit-image/commits?author=adamjstewart>`_
- `Adeyemi Biola  (@decorouz) <https://github.com/scikit-image/scikit-image/commits?author=decorouz>`_
- `aeisenbarth (@aeisenbarth) <https://github.com/scikit-image/scikit-image/commits?author=aeisenbarth>`_
- `Ananya Srivastava (@ana42742) <https://github.com/scikit-image/scikit-image/commits?author=ana42742>`_
- `Bohumír Zámečník (@bzamecnik) <https://github.com/scikit-image/scikit-image/commits?author=bzamecnik>`_
- `Carlos Horn (@carloshorn) <https://github.com/scikit-image/scikit-image/commits?author=carloshorn>`_
- `Daniel Angelov (@23pointsNorth) <https://github.com/scikit-image/scikit-image/commits?author=23pointsNorth>`_
- `DavidTorpey (@DavidTorpey) <https://github.com/scikit-image/scikit-image/commits?author=DavidTorpey>`_
- `Dipkumar Patel (@immortal3) <https://github.com/scikit-image/scikit-image/commits?author=immortal3>`_
- `Enrico Tagliavini (@enricotagliavini) <https://github.com/scikit-image/scikit-image/commits?author=enricotagliavini>`_
- `Eric Prestat (@ericpre) <https://github.com/scikit-image/scikit-image/commits?author=ericpre>`_
- `GGoussar (@GGoussar) <https://github.com/scikit-image/scikit-image/commits?author=GGoussar>`_
- `Gregory Lee (@grlee77) <https://github.com/scikit-image/scikit-image/commits?author=grlee77>`_
- `harshitha kolipaka (@harshithakolipaka) <https://github.com/scikit-image/scikit-image/commits?author=harshithakolipaka>`_
- `Hayato Ikoma (@hayatoikoma) <https://github.com/scikit-image/scikit-image/commits?author=hayatoikoma>`_
- `i-aki-y (@i-aki-y) <https://github.com/scikit-image/scikit-image/commits?author=i-aki-y>`_
- `Jake Martin (@jakeMartin1234) <https://github.com/scikit-image/scikit-image/commits?author=jakeMartin1234>`_
- `Jarrod Millman (@jarrodmillman) <https://github.com/scikit-image/scikit-image/commits?author=jarrodmillman>`_
- `Juan Nunez-Iglesias (@jni) <https://github.com/scikit-image/scikit-image/commits?author=jni>`_
- `Kevin MEETOOA (@kevinmeetooa) <https://github.com/scikit-image/scikit-image/commits?author=kevinmeetooa>`_
- `Lars Grüter (@lagru) <https://github.com/scikit-image/scikit-image/commits?author=lagru>`_
- `Loïc Estève (@lesteve) <https://github.com/scikit-image/scikit-image/commits?author=lesteve>`_
- `mahamtariq58 (@mahamtariq58) <https://github.com/scikit-image/scikit-image/commits?author=mahamtariq58>`_
- `Marianne Corvellec (@mkcor) <https://github.com/scikit-image/scikit-image/commits?author=mkcor>`_
- `Mark Harfouche (@hmaarrfk) <https://github.com/scikit-image/scikit-image/commits?author=hmaarrfk>`_
- `Matthias Bussonnier (@Carreau) <https://github.com/scikit-image/scikit-image/commits?author=Carreau>`_
- `Matus Valo (@matusvalo) <https://github.com/scikit-image/scikit-image/commits?author=matusvalo>`_
- `Michael Görner (@v4hn) <https://github.com/scikit-image/scikit-image/commits?author=v4hn>`_
- `Ramyashri Padmanabhakumar (@rum1887) <https://github.com/scikit-image/scikit-image/commits?author=rum1887>`_
- `scott-vsi (@scott-vsi) <https://github.com/scikit-image/scikit-image/commits?author=scott-vsi>`_
- `Sean Quinn (@seanpquinn) <https://github.com/scikit-image/scikit-image/commits?author=seanpquinn>`_
- `Stefan van der Walt (@stefanv) <https://github.com/scikit-image/scikit-image/commits?author=stefanv>`_
- `Tony Reina (@tonyreina) <https://github.com/scikit-image/scikit-image/commits?author=tonyreina>`_


27 reviewers added to this release (alphabetical)
-------------------------------------------------

- `Adeyemi Biola  (@decorouz) <https://github.com/scikit-image/scikit-image/commits?author=decorouz>`_
- `aeisenbarth (@aeisenbarth) <https://github.com/scikit-image/scikit-image/commits?author=aeisenbarth>`_
- `Ananya Srivastava (@ana42742) <https://github.com/scikit-image/scikit-image/commits?author=ana42742>`_
- `Brigitta Sipőcz (@bsipocz) <https://github.com/scikit-image/scikit-image/commits?author=bsipocz>`_
- `Carlos Horn (@carloshorn) <https://github.com/scikit-image/scikit-image/commits?author=carloshorn>`_
- `Cris Luengo (@crisluengo) <https://github.com/scikit-image/scikit-image/commits?author=crisluengo>`_
- `DavidTorpey (@DavidTorpey) <https://github.com/scikit-image/scikit-image/commits?author=DavidTorpey>`_
- `Dipkumar Patel (@immortal3) <https://github.com/scikit-image/scikit-image/commits?author=immortal3>`_
- `Enrico Tagliavini (@enricotagliavini) <https://github.com/scikit-image/scikit-image/commits?author=enricotagliavini>`_
- `Gregory Lee (@grlee77) <https://github.com/scikit-image/scikit-image/commits?author=grlee77>`_
- `Henry Pinkard (@henrypinkard) <https://github.com/scikit-image/scikit-image/commits?author=henrypinkard>`_
- `i-aki-y (@i-aki-y) <https://github.com/scikit-image/scikit-image/commits?author=i-aki-y>`_
- `Jarrod Millman (@jarrodmillman) <https://github.com/scikit-image/scikit-image/commits?author=jarrodmillman>`_
- `Juan Nunez-Iglesias (@jni) <https://github.com/scikit-image/scikit-image/commits?author=jni>`_
- `Kevin MEETOOA (@kevinmeetooa) <https://github.com/scikit-image/scikit-image/commits?author=kevinmeetooa>`_
- `kzuiderveld (@kzuiderveld) <https://github.com/scikit-image/scikit-image/commits?author=kzuiderveld>`_
- `Lars Grüter (@lagru) <https://github.com/scikit-image/scikit-image/commits?author=lagru>`_
- `Marianne Corvellec (@mkcor) <https://github.com/scikit-image/scikit-image/commits?author=mkcor>`_
- `Mark Harfouche (@hmaarrfk) <https://github.com/scikit-image/scikit-image/commits?author=hmaarrfk>`_
- `Ramyashri Padmanabhakumar (@rum1887) <https://github.com/scikit-image/scikit-image/commits?author=rum1887>`_
- `Riadh Fezzani (@rfezzani) <https://github.com/scikit-image/scikit-image/commits?author=rfezzani>`_
- `Sean Quinn (@seanpquinn) <https://github.com/scikit-image/scikit-image/commits?author=seanpquinn>`_
- `Sebastian Berg (@seberg) <https://github.com/scikit-image/scikit-image/commits?author=seberg>`_
- `Sebastian Wallkötter (@FirefoxMetzger) <https://github.com/scikit-image/scikit-image/commits?author=FirefoxMetzger>`_
- `Stefan van der Walt (@stefanv) <https://github.com/scikit-image/scikit-image/commits?author=stefanv>`_
- `Tony Reina (@tonyreina) <https://github.com/scikit-image/scikit-image/commits?author=tonyreina>`_
- `Tony Reina (@tony-res) <https://github.com/scikit-image/scikit-image/commits?author=tony-res>`_

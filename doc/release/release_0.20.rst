Announcement: scikit-image 0.20.0
=================================

We're happy to announce the release of scikit-image v0.20.0!

scikit-image is an image processing toolbox built on SciPy that includes algorithms
for segmentation, geometric transformations, color space manipulation,
analysis, filtering, morphology, feature detection, and more.

For more information, examples, and documentation, please visit our website:

https://scikit-image.org


New features and improvements
-----------------------------
- Add support for footprint decomposition to several footprint generating and consuming functions in ``skimage.morphology``.
  By decomposing a footprint into several smaller ones, morphological operations can potentially be sped up.
  The decomposed footprint can be generated with the new ``decomposition`` parameter of the functions ``rectangle``, ``diamond``, ``disk``, ``cube``, ``octahedron``, ``ball``, and ``octagon`` in ``skimage.morphology``.
  The ``footprint`` parameter of the functions ``binary_erosion``, ``binary_dilation``, ``binary_opening``, ``binary_closing``, ``erosion``, ``dilation``, ``opening``, ``closing``, ``white_tophat``, and ``black_tophat`` in ``skimage.morphology`` now accepts a sequence of smaller footprints that are applied consecutively as well. See the respective docstrings for more details
  (`#5482 <https://github.com/scikit-image/scikit-image/pull/5482>`_, `#6151 <https://github.com/scikit-image/scikit-image/pull/6151>`_).
- Add support for the Modified Hausdorff Distance (MHD) metric in ``skimage.metrics.hausdorff_distance`` via the new parameter ``method``.
  The MHD can be more robust against outliers than the directed Hausdorff Distance (HD)
  (`#5581 <https://github.com/scikit-image/scikit-image/pull/5581>`_).
- Added two datasets ``skimage.data.protein_transport`` and ``skimage.data.nickel_solidification``
  (`#6087 <https://github.com/scikit-image/scikit-image/pull/6087>`_).
- Incorporate RANSAC improvements from scikit-learn into ``skimage.measure.ransac`` which should lower iteration times
  (`#6046 <https://github.com/scikit-image/scikit-image/pull/6046>`_).
- Add the new parameter ``use_gaussian_derivatives`` to ``skimage.feature.hessian_matrix`` which allows the computation of the Hessian matrix by convolving with Gaussian derivatives
  (`#6149 <https://github.com/scikit-image/scikit-image/pull/6149>`_).
- Allow footprints with non-adjacent pixels as neighbors in ``skimage.morphology.flood_fill``
  (`#6236 <https://github.com/scikit-image/scikit-image/pull/6236>`_).
- Add new parameters ``squared_butterworth`` and ``npad`` to ``skimage.filters.butterworth``, which add support for squaring the filter and edge padding
  (`#6251 <https://github.com/scikit-image/scikit-image/pull/6251>`_).
- Add support for n-dimensional images in ``skimage.filters.farid`` (Farid & Simoncelli filter)
  (`#6257 <https://github.com/scikit-image/scikit-image/pull/6257>`_).

Changes and new deprecations
----------------------------

Completed deprecations
----------------------
- Remove ``skimage.viewer`` which was scheduled for the postponed version 1.0
  (`#6160 <https://github.com/scikit-image/scikit-image/pull/6160>`_).
- Remove deprecated parameter ``indices`` from ``skimage.feature.peak_local_max``
  (`#6161 <https://github.com/scikit-image/scikit-image/pull/6161>`_).
- Remove ``skimage.feature.structure_tensor_eigvals`` (it was replaced by ``skimage.feature.structure_tensor_eigenvalues``) and change the default parameter value in ``skimage.feature.structure_tensor`` to ``order="rc"``
  (`#6162 <https://github.com/scikit-image/scikit-image/pull/6162>`_).
- Remove deprecated parameter ``array`` in favor of ``image`` from ``skimage.measure.find_contours``
  (`#6163 <https://github.com/scikit-image/scikit-image/pull/6163>`_).
- Remove deprecated Qt IO plugin and the ``skivi`` console script
  (`#6164 <https://github.com/scikit-image/scikit-image/pull/6164>`_).
- Remove deprecated parameter value ``method='_lorensen'`` in ``skimage.measure.marching_cubes``
  (`#6230 <https://github.com/scikit-image/scikit-image/pull/6230>`_).

Bug fixes
---------

Documentation
-------------
- Add a textbook-like tutorial on measuring fluorescence at the nuclear envelope of a cell
  (`#5262 <https://github.com/scikit-image/scikit-image/pull/5262>`_).
- Add new gallery example on decomposing flat footprints
  (`#6151 <https://github.com/scikit-image/scikit-image/pull/6151>`_).
- Add a new and gallery example "Butterworth Filters" and improve docstring of ``skimage.filters.butterworth``
  (`#6251 <https://github.com/scikit-image/scikit-image/pull/6251>`_).
- Improve the the gallery example "Measure perimeters with different estimators"
  (`#6200 <https://github.com/scikit-image/scikit-image/pull/6200>`_, `#6121 <https://github.com/scikit-image/scikit-image/pull/6121>`_).
- Add ``SUPPORT.md`` to repository to help users from GitHub find appropriate support
  resources
  (`#6171 <https://github.com/scikit-image/scikit-image/pull/6171>`_).
- Add ``CITATION.bib`` to repository to help with citing scikit-image
  (`#6195 <https://github.com/scikit-image/scikit-image/pull/6195>`_).
- Remove references to the ``git://`` connection protocol
  (`#6201 <https://github.com/scikit-image/scikit-image/pull/6201>`_).
- Fix formatting in the docstring of ``skimage.metrics.hausdorff_distance``
  (`#6203 <https://github.com/scikit-image/scikit-image/pull/6203>`_).
- Change "neighbour" to EN-US spelling "neighbor"
  (`#6204 <https://github.com/scikit-image/scikit-image/pull/6204>`_).
- Update scikit-image's mailing addresses to the new domain discuss.scientific-python.org
  (`#6255 <https://github.com/scikit-image/scikit-image/pull/6255>`_).

Other and development related updates
-------------------------------------
- Add benchmarks for ``morphology.local_maxima``
  (`#3255 <https://github.com/scikit-image/scikit-image/pull/3255>`_).
- Fix the autogeneration of API docs for lazy loaded subpackages
  (`#6075 <https://github.com/scikit-image/scikit-image/pull/6075>`_).
- Checkout gh-pages with a shallow clone
  (`#6085 <https://github.com/scikit-image/scikit-image/pull/6085>`_).
- Fix dev doc build
  (`#6091 <https://github.com/scikit-image/scikit-image/pull/6091>`_).
- Expand reviewer guidelines in pull request template
  (`#6208 <https://github.com/scikit-image/scikit-image/pull/6208>`_).

TODO
----
- Fix inpaint_biharmonic for images with Fortran-ordered memory layout (`#6263 <https://github.com/scikit-image/scikit-image/pull/6263>`_)
- Fix balance in example code (`#6265 <https://github.com/scikit-image/scikit-image/pull/6265>`_)
- Support array-likes consistently in geometric transforms (`#6270 <https://github.com/scikit-image/scikit-image/pull/6270>`_)
- hough_line_peaks fix for corner case with optimal angle=0 (`#6271 <https://github.com/scikit-image/scikit-image/pull/6271>`_)
- add warning on non-integer image inputs to local_binary_pattern (`#6272 <https://github.com/scikit-image/scikit-image/pull/6272>`_)
- More flexible collections with custom load_func. (`#6276 <https://github.com/scikit-image/scikit-image/pull/6276>`_)
- clarify behavior of watershed segmentation line with touching markers (`#6280 <https://github.com/scikit-image/scikit-image/pull/6280>`_)
- Stop using `git://` for submodules (`#6283 <https://github.com/scikit-image/scikit-image/pull/6283>`_)
- Fix adjust_gamma round-off error (`#6285 <https://github.com/scikit-image/scikit-image/pull/6285>`_)
- Update for the `MultiImage` docstring. (`#6290 <https://github.com/scikit-image/scikit-image/pull/6290>`_)
- Polish the `MultiImage` docstring. (`#6292 <https://github.com/scikit-image/scikit-image/pull/6292>`_)
- Update plot_pyramid.py demo to work for diversified shaped images and downsample factors (`#6293 <https://github.com/scikit-image/scikit-image/pull/6293>`_)
- remove extraneous function in createluts.py (and move mc_meta reference code) (`#6294 <https://github.com/scikit-image/scikit-image/pull/6294>`_)
- Add spacing to regionprops and moments. (`#6296 <https://github.com/scikit-image/scikit-image/pull/6296>`_)
- Update data urls to point to a specific commit (`#6297 <https://github.com/scikit-image/scikit-image/pull/6297>`_)
- New thumbnails for General-purpose images and scientific images (`#6298 <https://github.com/scikit-image/scikit-image/pull/6298>`_)
- New thumbnail for "Datasets" example  by adjusting contrast (`#6300 <https://github.com/scikit-image/scikit-image/pull/6300>`_)
- New thumbnail for Specific images (`#6301 <https://github.com/scikit-image/scikit-image/pull/6301>`_)
- drop codecov badge from README (`#6302 <https://github.com/scikit-image/scikit-image/pull/6302>`_)
- Cython style: prefer cnp.float32_t and cnp.float64_t for clarity (`#6303 <https://github.com/scikit-image/scikit-image/pull/6303>`_)
- Fix for error in 'Using Polar and Log-Polar Transformations for Registration' (`#6304 <https://github.com/scikit-image/scikit-image/pull/6304>`_) (`#6306 <https://github.com/scikit-image/scikit-image/pull/6306>`_)
- Remove undefined 'python_to_notebook' in doc/ext/notebook_doc.py (`#6307 <https://github.com/scikit-image/scikit-image/pull/6307>`_)
- Fix CI by pinning to Pillow!=9.1.0 (`#6315 <https://github.com/scikit-image/scikit-image/pull/6315>`_)
- Fix skeletonize behavior (`#6322 <https://github.com/scikit-image/scikit-image/pull/6322>`_)
- parameterize moments tests (`#6323 <https://github.com/scikit-image/scikit-image/pull/6323>`_)
- skeletonize maintenance (`#6327 <https://github.com/scikit-image/scikit-image/pull/6327>`_)
- Fix issue with newer versions of matplotlib in manual segmentation (`#6328 <https://github.com/scikit-image/scikit-image/pull/6328>`_)
- warp/rotate: fixed a bug with clipping when cval is not in the input range (`#6335 <https://github.com/scikit-image/scikit-image/pull/6335>`_)
- Add skip-4 draft (`#6339 <https://github.com/scikit-image/scikit-image/pull/6339>`_)
- add int64 support to `filters.rank_order` and `morphology.reconstruction` (`#6342 <https://github.com/scikit-image/scikit-image/pull/6342>`_)
- avoid warnings about change to v3 API from imageio (`#6343 <https://github.com/scikit-image/scikit-image/pull/6343>`_)
- Fix smoothed image computation when mask is None in canny (`#6348 <https://github.com/scikit-image/scikit-image/pull/6348>`_)
- Fix channel_axis default for cycle_spin (`#6352 <https://github.com/scikit-image/scikit-image/pull/6352>`_)
- Fix SKIP4 header and links (`#6353 <https://github.com/scikit-image/scikit-image/pull/6353>`_)
- Improve histogram matching performance on unsigned integer data (resume `#6209 <https://github.com/scikit-image/scikit-image/pull/6209>`_) (`#6354 <https://github.com/scikit-image/scikit-image/pull/6354>`_)
- remove use of deprecated kwargs from `test_tifffile_kwarg_passthrough` (`#6355 <https://github.com/scikit-image/scikit-image/pull/6355>`_)
- Document support for Path objects in io functions (`#6361 <https://github.com/scikit-image/scikit-image/pull/6361>`_)
- Add 3D rotation and translation properties for EuclideanTransform object, and 3D scale for SimilarityTransform (`#6367 <https://github.com/scikit-image/scikit-image/pull/6367>`_)
-  Fixing typo in _probabilistic_hough_line (`#6373 <https://github.com/scikit-image/scikit-image/pull/6373>`_)
- Improve multi-Otsu error message and maintenance of threshold.py (`#6375 <https://github.com/scikit-image/scikit-image/pull/6375>`_)
- Removing reference to `marching_cubes_lewiner` from `plot_marching_cubes.py`  (`#6377 <https://github.com/scikit-image/scikit-image/pull/6377>`_)
- pin to pip<22.1 (`#6379 <https://github.com/scikit-image/scikit-image/pull/6379>`_)
- Update GH actions (`#6382 <https://github.com/scikit-image/scikit-image/pull/6382>`_)
- Update matplotlib minimum version (`#6383 <https://github.com/scikit-image/scikit-image/pull/6383>`_)
- Don't use pillow 9.1.1 (`#6384 <https://github.com/scikit-image/scikit-image/pull/6384>`_)
- Update minimum supported numpy, scipy, and networkx (`#6385 <https://github.com/scikit-image/scikit-image/pull/6385>`_)
- Canny: cythonize non-maximum suppression (`#6387 <https://github.com/scikit-image/scikit-image/pull/6387>`_)
- derive OBJECT_COLUMNS from COL_DTYPES in regionprops (`#6389 <https://github.com/scikit-image/scikit-image/pull/6389>`_)
- DOC: add original plot in examples/segmentation/plot_expand_labels.py (`#6396 <https://github.com/scikit-image/scikit-image/pull/6396>`_)
- Add support for NumPy 1.23 (`#6400 <https://github.com/scikit-image/scikit-image/pull/6400>`_)
- Use supported circleci images (`#6401 <https://github.com/scikit-image/scikit-image/pull/6401>`_)
- Update minimum pillow dependency (`#6402 <https://github.com/scikit-image/scikit-image/pull/6402>`_)
- In newer PIL, palette may contain <256 entries (`#6405 <https://github.com/scikit-image/scikit-image/pull/6405>`_)
- Use artifact-redirector (`#6407 <https://github.com/scikit-image/scikit-image/pull/6407>`_)
- Sync numpy minimum version (`#6409 <https://github.com/scikit-image/scikit-image/pull/6409>`_)
- Fix computation of histogram bins for multichannel integer-valued images (`#6413 <https://github.com/scikit-image/scikit-image/pull/6413>`_)
- forward-port 0.19.3 release notes (`#6416 <https://github.com/scikit-image/scikit-image/pull/6416>`_)
- forwardport gh-6369: Fix windows wheels: use vsdevcmd.bat to make sure rc.exe is on the path (`#6417 <https://github.com/scikit-image/scikit-image/pull/6417>`_)
- Adding missing copyrights to LICENSE.txt, formatting according to SPDX identifiers (`#6419 <https://github.com/scikit-image/scikit-image/pull/6419>`_)
- Document refactoring from grey* to graymatrix and graycoprops in 0.19 with versionchanged directive (`#6420 <https://github.com/scikit-image/scikit-image/pull/6420>`_)
- [MINOR] centre -> center in doc/examples/applications/plot_morphology.py (`#6421 <https://github.com/scikit-image/scikit-image/pull/6421>`_)
- [MINOR] colour -> color in doc/examples/applications/plot_3d_interaction.py (`#6422 <https://github.com/scikit-image/scikit-image/pull/6422>`_)
- Restrict GitHub Actions permissions only for required ones (`#6426 <https://github.com/scikit-image/scikit-image/pull/6426>`_)
- Exclude submodules of doc from package install (`#6428 <https://github.com/scikit-image/scikit-image/pull/6428>`_)
- Substitute vertices with simplices in `transform/_geometric.py` (`#6430 <https://github.com/scikit-image/scikit-image/pull/6430>`_)
- example to render text onto an image (`#6431 <https://github.com/scikit-image/scikit-image/pull/6431>`_)
- Fix minor typo in sato() implemntation. (`#6434 <https://github.com/scikit-image/scikit-image/pull/6434>`_)
- Simplify sort-by-absolute-value in ridge filters. (`#6440 <https://github.com/scikit-image/scikit-image/pull/6440>`_)
- Speedup ~2x hessian_matrix_eigvals and 2D structure_tensor_eigenvalues. (`#6441 <https://github.com/scikit-image/scikit-image/pull/6441>`_)
- removed the completed items in 0.2 (`#6442 <https://github.com/scikit-image/scikit-image/pull/6442>`_)
- doc: replaced broken links (`#6445 <https://github.com/scikit-image/scikit-image/pull/6445>`_)
- Rewrite the meijering, sato, and frangi ridge filters. (`#6446 <https://github.com/scikit-image/scikit-image/pull/6446>`_)
- No valueerror for underdetermined (`#6453 <https://github.com/scikit-image/scikit-image/pull/6453>`_)
- Make Wiener restoration N-d (`#6454 <https://github.com/scikit-image/scikit-image/pull/6454>`_)
- Remove repeated import in canny_py (`#6457 <https://github.com/scikit-image/scikit-image/pull/6457>`_)
- Refactor occurences of `f = open(...)` using `with open(...) as f` instead (`#6458 <https://github.com/scikit-image/scikit-image/pull/6458>`_)
- Add multiscale structural similarity (`#6470 <https://github.com/scikit-image/scikit-image/pull/6470>`_)
- Add `alpha` argument to `adapted_rand_error`  (`#6472 <https://github.com/scikit-image/scikit-image/pull/6472>`_)
- Fix broken link to skimage.filters.sobel. (`#6474 <https://github.com/scikit-image/scikit-image/pull/6474>`_)
- Use broadcast_to instead of as_strided to generate broadcasted arrays. (`#6476 <https://github.com/scikit-image/scikit-image/pull/6476>`_)
- Update Ubuntu LTS version on Actions workflows (`#6478 <https://github.com/scikit-image/scikit-image/pull/6478>`_)
- changed image1 to moving_image in tvl1 parameter docs (`#6480 <https://github.com/scikit-image/scikit-image/pull/6480>`_)
- Use matplotlib.colormaps instead of deprecated cm.get_cmap in show_rag (`#6483 <https://github.com/scikit-image/scikit-image/pull/6483>`_)
- Use context manager when possible (`#6484 <https://github.com/scikit-image/scikit-image/pull/6484>`_)
- Document inclusion criteria for new functionality in core developer guide (`#6488 <https://github.com/scikit-image/scikit-image/pull/6488>`_)
- Use pyplot.get_cmap for compatiblity with matplotlib 3.3 to 3.6 in in show_rag (`#6490 <https://github.com/scikit-image/scikit-image/pull/6490>`_)
- Replace reference to api_changes.rst with release_dev.rst (`#6495 <https://github.com/scikit-image/scikit-image/pull/6495>`_)
- Support float input to skimage.draw.rectangle() [`#4283 <https://github.com/scikit-image/scikit-image/pull/4283>`_] (`#6501 <https://github.com/scikit-image/scikit-image/pull/6501>`_)
- Find peaks at border with `peak_local_max with `exclude_border=0` (`#6502 <https://github.com/scikit-image/scikit-image/pull/6502>`_)
- Fix resize anti_aliazing default value when input dtype is integer and order == 0 (`#6503 <https://github.com/scikit-image/scikit-image/pull/6503>`_)
- Add Github actions/stale to label "dormant" issues and PRs (`#6506 <https://github.com/scikit-image/scikit-image/pull/6506>`_)
- Clarify header pointing to notes for latest version released. (`#6508 <https://github.com/scikit-image/scikit-image/pull/6508>`_)
- Reduce ridge filters memory footprints (`#6509 <https://github.com/scikit-image/scikit-image/pull/6509>`_)
- Update benchmark environment to recent Python and NumPy versions (`#6511 <https://github.com/scikit-image/scikit-image/pull/6511>`_)
- Add new flag to convex_hull_image and grid_points_in_poly (`#6515 <https://github.com/scikit-image/scikit-image/pull/6515>`_)
- relax label name comparison in benchmarks.yaml (`#6520 <https://github.com/scikit-image/scikit-image/pull/6520>`_)
- update plot_euler_number.py for maplotlib 3.6 compatibility (`#6522 <https://github.com/scikit-image/scikit-image/pull/6522>`_)
- Use mask during rescaling in segmentation.slic and improve handling of error cases (`#6525 <https://github.com/scikit-image/scikit-image/pull/6525>`_)
- make non-functional change to build.txt to fix cache issue on CircleCI (`#6528 <https://github.com/scikit-image/scikit-image/pull/6528>`_)
- update setup.cfg field from license_file to license_files (`#6529 <https://github.com/scikit-image/scikit-image/pull/6529>`_)
- Fix wrong doc on connected pixels in flood (`#6534 <https://github.com/scikit-image/scikit-image/pull/6534>`_)
- Minor doc fix: add missing print statement in the `plot_segmentations.py` example (`#6535 <https://github.com/scikit-image/scikit-image/pull/6535>`_)
- Apply codespell to fix common spelling mistakes (`#6537 <https://github.com/scikit-image/scikit-image/pull/6537>`_)
- Ignore codespell fixes with git blame (`#6539 <https://github.com/scikit-image/scikit-image/pull/6539>`_)
- Add missing spaces to regionprops error message. (`#6545 <https://github.com/scikit-image/scikit-image/pull/6545>`_)
- Update "Mark dormant issues" workflow (`#6546 <https://github.com/scikit-image/scikit-image/pull/6546>`_)
- Add missing space in math directive in normalized_mutual_information's docstring (`#6549 <https://github.com/scikit-image/scikit-image/pull/6549>`_)
- Add missing option stale-pr-label for "Mark dormant issues" workflow (`#6552 <https://github.com/scikit-image/scikit-image/pull/6552>`_)
- Remove FUNDING.yml in preference of org version (`#6553 <https://github.com/scikit-image/scikit-image/pull/6553>`_)

Pull Requests in this release
-----------------------------

Includes backported changes to earlier versions.

- Add benchmarks for morphology.local_maxima (`#3255 <https://github.com/scikit-image/scikit-image/pull/3255>`_)
- Add textbook-like tutorial on measuring fluorescence at nuclear envelope. (`#5262 <https://github.com/scikit-image/scikit-image/pull/5262>`_)
- Footprint decomposition for faster morphology (part 1) (`#5482 <https://github.com/scikit-image/scikit-image/pull/5482>`_)
- Implementation of the Modified Hausdorff Distance (MHD) metric (`#5581 <https://github.com/scikit-image/scikit-image/pull/5581>`_)
- Fix typo in moments_hu docstring (`#6016 <https://github.com/scikit-image/scikit-image/pull/6016>`_)
- Transplant the change of scikit-learn into scikit-image for RANSAC  (`#6046 <https://github.com/scikit-image/scikit-image/pull/6046>`_)
- Fix API docs autogeneration for lazy loaded subpackages (`#6075 <https://github.com/scikit-image/scikit-image/pull/6075>`_)
- checkout gh-pages with a shallow clone (`#6085 <https://github.com/scikit-image/scikit-image/pull/6085>`_)
- Add two datasets for use in upcoming scientific tutorials. (`#6087 <https://github.com/scikit-image/scikit-image/pull/6087>`_)
- Skip tests requiring fetched data (`#6089 <https://github.com/scikit-image/scikit-image/pull/6089>`_)
- Fix dev doc build (`#6091 <https://github.com/scikit-image/scikit-image/pull/6091>`_)
- Preserve backwards compatibility for `channel_axis` parameter in transform functions (`#6095 <https://github.com/scikit-image/scikit-image/pull/6095>`_)
- restore non-underscore functions in skimage.data (`#6097 <https://github.com/scikit-image/scikit-image/pull/6097>`_)
- forward port of `#6098 <https://github.com/scikit-image/scikit-image/pull/6098>`_ (fix MacOS arm64 wheels and Windows Python 3.10 AMD64 wheel) (`#6101 <https://github.com/scikit-image/scikit-image/pull/6101>`_)
- make rank filter test comparisons robust across architectures (`#6103 <https://github.com/scikit-image/scikit-image/pull/6103>`_)
- pass a specific random_state into ransac in test_ransac_geometric (`#6105 <https://github.com/scikit-image/scikit-image/pull/6105>`_)
- Add linker flags to strip debug symbols during wheel building (`#6109 <https://github.com/scikit-image/scikit-image/pull/6109>`_)
- relax test condition to make it more robust to variable CI load (`#6114 <https://github.com/scikit-image/scikit-image/pull/6114>`_)
- respect SKIMAGE_TEST_STRICT_WARNINGS_GLOBAL setting in tests.yml (`#6118 <https://github.com/scikit-image/scikit-image/pull/6118>`_)
- Fixed minor typos in perimeters example (`#6121 <https://github.com/scikit-image/scikit-image/pull/6121>`_)
- bump deprecated Azure windows environment (`#6130 <https://github.com/scikit-image/scikit-image/pull/6130>`_)
- Update user warning message for viewer module. (`#6133 <https://github.com/scikit-image/scikit-image/pull/6133>`_)
- fix phase_cross_correlation typo (`#6139 <https://github.com/scikit-image/scikit-image/pull/6139>`_)
- Fix channel_axis handling in pyramid_gaussian and pyramid_laplace (`#6145 <https://github.com/scikit-image/scikit-image/pull/6145>`_)
- deprecate n_iter_max (should be max_num_iter) (`#6148 <https://github.com/scikit-image/scikit-image/pull/6148>`_)
- Update of Meijering algorithm (resumed) (`#6149 <https://github.com/scikit-image/scikit-image/pull/6149>`_)
- Implement 2D ellipse footprint decomposition (`#6151 <https://github.com/scikit-image/scikit-image/pull/6151>`_)
- specify python version used by mybinder.org for gallery demos (`#6152 <https://github.com/scikit-image/scikit-image/pull/6152>`_)
- remove skimage.viewer (`#6160 <https://github.com/scikit-image/scikit-image/pull/6160>`_)
- remove deprecated indices kwarg from peak_local_max (`#6161 <https://github.com/scikit-image/scikit-image/pull/6161>`_)
- remove structure_tensor_eigvals and change default structure_tensor order (`#6162 <https://github.com/scikit-image/scikit-image/pull/6162>`_)
- remove deprecate_kwarg decorator from find_contours (`#6163 <https://github.com/scikit-image/scikit-image/pull/6163>`_)
- Remove deprecated Qt IO plugin and skivi script (`#6164 <https://github.com/scikit-image/scikit-image/pull/6164>`_)
- Fix unintended change to output dtype of match_histograms (`#6169 <https://github.com/scikit-image/scikit-image/pull/6169>`_)
- add SUPPORT.md (helps point users from GitHub to appropriate support resources) (`#6171 <https://github.com/scikit-image/scikit-image/pull/6171>`_)
- Fix decorators warnings stacklevel (`#6183 <https://github.com/scikit-image/scikit-image/pull/6183>`_)
- Fix SIFT wrong octave indices + typo (`#6184 <https://github.com/scikit-image/scikit-image/pull/6184>`_)
- Fix issue6190 - inconsistent default parameters in pyramids.py (`#6191 <https://github.com/scikit-image/scikit-image/pull/6191>`_)
- Adding CITATION.bib (`#6195 <https://github.com/scikit-image/scikit-image/pull/6195>`_)
- Improve writing for perimeter estimation example. (`#6200 <https://github.com/scikit-image/scikit-image/pull/6200>`_)
- Removing references to git connection protocol (`#6201 <https://github.com/scikit-image/scikit-image/pull/6201>`_)
- DOC: Minor cosmetic fixup to address UserWarning. (`#6203 <https://github.com/scikit-image/scikit-image/pull/6203>`_)
- Changing occurrences of "neighbour" to EN-US spelling, "neighbor" (`#6204 <https://github.com/scikit-image/scikit-image/pull/6204>`_)
- Always set params to nan when ProjectiveTransform.estimate fails (`#6207 <https://github.com/scikit-image/scikit-image/pull/6207>`_)
- expand reviewer guidelines in pull request template (`#6208 <https://github.com/scikit-image/scikit-image/pull/6208>`_)
- PiecewiseAffineTransform.estimate return should reflect underlying transforms (`#6211 <https://github.com/scikit-image/scikit-image/pull/6211>`_)
- EuclideanTransform.estimate should return False when NaNs are present (`#6214 <https://github.com/scikit-image/scikit-image/pull/6214>`_)
- Allow the output_shape argument to be any iterable for resize and resize_local_mean (`#6219 <https://github.com/scikit-image/scikit-image/pull/6219>`_)
- Update filename in testing instructions. (`#6223 <https://github.com/scikit-image/scikit-image/pull/6223>`_)
- Fix calculation of Z normal in marching cubes (`#6227 <https://github.com/scikit-image/scikit-image/pull/6227>`_)
- Remove redundant testing on Appveyor (`#6229 <https://github.com/scikit-image/scikit-image/pull/6229>`_)
- remove deprecated marching_cubes '_lorensen' option (`#6230 <https://github.com/scikit-image/scikit-image/pull/6230>`_)
- Update imports/refs from deprecated scipy.ndimage.filters namespace (`#6231 <https://github.com/scikit-image/scikit-image/pull/6231>`_)
- Include Cython sources via package_data (`#6232 <https://github.com/scikit-image/scikit-image/pull/6232>`_)
- Allow non-adjacent footprints in flood_fill. (`#6236 <https://github.com/scikit-image/scikit-image/pull/6236>`_)
- DOC: fix SciPy intersphinx (`#6239 <https://github.com/scikit-image/scikit-image/pull/6239>`_)
- Fix bug in SLIC superpixels with `enforce_connectivity=True` and `start_label > 0` (`#6242 <https://github.com/scikit-image/scikit-image/pull/6242>`_)
- Fowardport PR `#6249 <https://github.com/scikit-image/scikit-image/pull/6249>`_ on branch main (update MacOS libomp installation in wheel building script) (`#6250 <https://github.com/scikit-image/scikit-image/pull/6250>`_)
- improve butterworth docstring and add new kwargs and gallery example (`#6251 <https://github.com/scikit-image/scikit-image/pull/6251>`_)
- Forward port v0.19.1 and v0.19.2 release notes (`#6253 <https://github.com/scikit-image/scikit-image/pull/6253>`_)
- Update skimage mailing addresses (`#6255 <https://github.com/scikit-image/scikit-image/pull/6255>`_)
- implement nD skimage.filters.farid (Farid & Simoncelli filter) (`#6257 <https://github.com/scikit-image/scikit-image/pull/6257>`_)
- Ignore sparse matrix deprecation warning (`#6261 <https://github.com/scikit-image/scikit-image/pull/6261>`_)
- Fix inpaint_biharmonic for images with Fortran-ordered memory layout (`#6263 <https://github.com/scikit-image/scikit-image/pull/6263>`_)
- Fix balance in example code (`#6265 <https://github.com/scikit-image/scikit-image/pull/6265>`_)
- Support array-likes consistently in geometric transforms (`#6270 <https://github.com/scikit-image/scikit-image/pull/6270>`_)
- hough_line_peaks fix for corner case with optimal angle=0 (`#6271 <https://github.com/scikit-image/scikit-image/pull/6271>`_)
- add warning on non-integer image inputs to local_binary_pattern (`#6272 <https://github.com/scikit-image/scikit-image/pull/6272>`_)
- More flexible collections with custom load_func. (`#6276 <https://github.com/scikit-image/scikit-image/pull/6276>`_)
- clarify behavior of watershed segmentation line with touching markers (`#6280 <https://github.com/scikit-image/scikit-image/pull/6280>`_)
- Stop using `git://` for submodules (`#6283 <https://github.com/scikit-image/scikit-image/pull/6283>`_)
- Fix adjust_gamma round-off error (`#6285 <https://github.com/scikit-image/scikit-image/pull/6285>`_)
- Update for the `MultiImage` docstring. (`#6290 <https://github.com/scikit-image/scikit-image/pull/6290>`_)
- Polish the `MultiImage` docstring. (`#6292 <https://github.com/scikit-image/scikit-image/pull/6292>`_)
- Update plot_pyramid.py demo to work for diversified shaped images and downsample factors (`#6293 <https://github.com/scikit-image/scikit-image/pull/6293>`_)
- remove extraneous function in createluts.py (and move mc_meta reference code) (`#6294 <https://github.com/scikit-image/scikit-image/pull/6294>`_)
- Add spacing to regionprops and moments. (`#6296 <https://github.com/scikit-image/scikit-image/pull/6296>`_)
- Update data urls to point to a specific commit (`#6297 <https://github.com/scikit-image/scikit-image/pull/6297>`_)
- New thumbnails for General-purpose images and scientific images (`#6298 <https://github.com/scikit-image/scikit-image/pull/6298>`_)
- New thumbnail for "Datasets" example  by adjusting contrast (`#6300 <https://github.com/scikit-image/scikit-image/pull/6300>`_)
- New thumbnail for Specific images (`#6301 <https://github.com/scikit-image/scikit-image/pull/6301>`_)
- drop codecov badge from README (`#6302 <https://github.com/scikit-image/scikit-image/pull/6302>`_)
- Cython style: prefer cnp.float32_t and cnp.float64_t for clarity (`#6303 <https://github.com/scikit-image/scikit-image/pull/6303>`_)
- Fix for error in 'Using Polar and Log-Polar Transformations for Registration' (`#6304 <https://github.com/scikit-image/scikit-image/pull/6304>`_) (`#6306 <https://github.com/scikit-image/scikit-image/pull/6306>`_)
- Remove undefined 'python_to_notebook' in doc/ext/notebook_doc.py (`#6307 <https://github.com/scikit-image/scikit-image/pull/6307>`_)
- Fix CI by pinning to Pillow!=9.1.0 (`#6315 <https://github.com/scikit-image/scikit-image/pull/6315>`_)
- Fix skeletonize behavior (`#6322 <https://github.com/scikit-image/scikit-image/pull/6322>`_)
- parameterize moments tests (`#6323 <https://github.com/scikit-image/scikit-image/pull/6323>`_)
- skeletonize maintenance (`#6327 <https://github.com/scikit-image/scikit-image/pull/6327>`_)
- Fix issue with newer versions of matplotlib in manual segmentation (`#6328 <https://github.com/scikit-image/scikit-image/pull/6328>`_)
- warp/rotate: fixed a bug with clipping when cval is not in the input range (`#6335 <https://github.com/scikit-image/scikit-image/pull/6335>`_)
- Add skip-4 draft (`#6339 <https://github.com/scikit-image/scikit-image/pull/6339>`_)
- add int64 support to `filters.rank_order` and `morphology.reconstruction` (`#6342 <https://github.com/scikit-image/scikit-image/pull/6342>`_)
- avoid warnings about change to v3 API from imageio (`#6343 <https://github.com/scikit-image/scikit-image/pull/6343>`_)
- Fix smoothed image computation when mask is None in canny (`#6348 <https://github.com/scikit-image/scikit-image/pull/6348>`_)
- Fix channel_axis default for cycle_spin (`#6352 <https://github.com/scikit-image/scikit-image/pull/6352>`_)
- Fix SKIP4 header and links (`#6353 <https://github.com/scikit-image/scikit-image/pull/6353>`_)
- Improve histogram matching performance on unsigned integer data (resume `#6209 <https://github.com/scikit-image/scikit-image/pull/6209>`_) (`#6354 <https://github.com/scikit-image/scikit-image/pull/6354>`_)
- remove use of deprecated kwargs from `test_tifffile_kwarg_passthrough` (`#6355 <https://github.com/scikit-image/scikit-image/pull/6355>`_)
- Document support for Path objects in io functions (`#6361 <https://github.com/scikit-image/scikit-image/pull/6361>`_)
- Add 3D rotation and translation properties for EuclideanTransform object, and 3D scale for SimilarityTransform (`#6367 <https://github.com/scikit-image/scikit-image/pull/6367>`_)
-  Fixing typo in _probabilistic_hough_line (`#6373 <https://github.com/scikit-image/scikit-image/pull/6373>`_)
- Improve multi-Otsu error message and maintenance of threshold.py (`#6375 <https://github.com/scikit-image/scikit-image/pull/6375>`_)
- Removing reference to `marching_cubes_lewiner` from `plot_marching_cubes.py`  (`#6377 <https://github.com/scikit-image/scikit-image/pull/6377>`_)
- pin to pip<22.1 (`#6379 <https://github.com/scikit-image/scikit-image/pull/6379>`_)
- Update GH actions (`#6382 <https://github.com/scikit-image/scikit-image/pull/6382>`_)
- Update matplotlib minimum version (`#6383 <https://github.com/scikit-image/scikit-image/pull/6383>`_)
- Don't use pillow 9.1.1 (`#6384 <https://github.com/scikit-image/scikit-image/pull/6384>`_)
- Update minimum supported numpy, scipy, and networkx (`#6385 <https://github.com/scikit-image/scikit-image/pull/6385>`_)
- Canny: cythonize non-maximum suppression (`#6387 <https://github.com/scikit-image/scikit-image/pull/6387>`_)
- derive OBJECT_COLUMNS from COL_DTYPES in regionprops (`#6389 <https://github.com/scikit-image/scikit-image/pull/6389>`_)
- DOC: add original plot in examples/segmentation/plot_expand_labels.py (`#6396 <https://github.com/scikit-image/scikit-image/pull/6396>`_)
- Add support for NumPy 1.23 (`#6400 <https://github.com/scikit-image/scikit-image/pull/6400>`_)
- Use supported circleci images (`#6401 <https://github.com/scikit-image/scikit-image/pull/6401>`_)
- Update minimum pillow dependency (`#6402 <https://github.com/scikit-image/scikit-image/pull/6402>`_)
- In newer PIL, palette may contain <256 entries (`#6405 <https://github.com/scikit-image/scikit-image/pull/6405>`_)
- Use artifact-redirector (`#6407 <https://github.com/scikit-image/scikit-image/pull/6407>`_)
- Sync numpy minimum version (`#6409 <https://github.com/scikit-image/scikit-image/pull/6409>`_)
- Fix computation of histogram bins for multichannel integer-valued images (`#6413 <https://github.com/scikit-image/scikit-image/pull/6413>`_)
- forward-port 0.19.3 release notes (`#6416 <https://github.com/scikit-image/scikit-image/pull/6416>`_)
- forwardport gh-6369: Fix windows wheels: use vsdevcmd.bat to make sure rc.exe is on the path (`#6417 <https://github.com/scikit-image/scikit-image/pull/6417>`_)
- Adding missing copyrights to LICENSE.txt, formatting according to SPDX identifiers (`#6419 <https://github.com/scikit-image/scikit-image/pull/6419>`_)
- Document refactoring from grey* to graymatrix and graycoprops in 0.19 with versionchanged directive (`#6420 <https://github.com/scikit-image/scikit-image/pull/6420>`_)
- [MINOR] centre -> center in doc/examples/applications/plot_morphology.py (`#6421 <https://github.com/scikit-image/scikit-image/pull/6421>`_)
- [MINOR] colour -> color in doc/examples/applications/plot_3d_interaction.py (`#6422 <https://github.com/scikit-image/scikit-image/pull/6422>`_)
- Restrict GitHub Actions permissions only for required ones (`#6426 <https://github.com/scikit-image/scikit-image/pull/6426>`_)
- Exclude submodules of doc from package install (`#6428 <https://github.com/scikit-image/scikit-image/pull/6428>`_)
- Substitute vertices with simplices in `transform/_geometric.py` (`#6430 <https://github.com/scikit-image/scikit-image/pull/6430>`_)
- example to render text onto an image (`#6431 <https://github.com/scikit-image/scikit-image/pull/6431>`_)
- Fix minor typo in sato() implemntation. (`#6434 <https://github.com/scikit-image/scikit-image/pull/6434>`_)
- Simplify sort-by-absolute-value in ridge filters. (`#6440 <https://github.com/scikit-image/scikit-image/pull/6440>`_)
- Speedup ~2x hessian_matrix_eigvals and 2D structure_tensor_eigenvalues. (`#6441 <https://github.com/scikit-image/scikit-image/pull/6441>`_)
- removed the completed items in 0.2 (`#6442 <https://github.com/scikit-image/scikit-image/pull/6442>`_)
- doc: replaced broken links (`#6445 <https://github.com/scikit-image/scikit-image/pull/6445>`_)
- Rewrite the meijering, sato, and frangi ridge filters. (`#6446 <https://github.com/scikit-image/scikit-image/pull/6446>`_)
- No valueerror for underdetermined (`#6453 <https://github.com/scikit-image/scikit-image/pull/6453>`_)
- Make Wiener restoration N-d (`#6454 <https://github.com/scikit-image/scikit-image/pull/6454>`_)
- Remove repeated import in canny_py (`#6457 <https://github.com/scikit-image/scikit-image/pull/6457>`_)
- Refactor occurences of `f = open(...)` using `with open(...) as f` instead (`#6458 <https://github.com/scikit-image/scikit-image/pull/6458>`_)
- Add multiscale structural similarity (`#6470 <https://github.com/scikit-image/scikit-image/pull/6470>`_)
- Add `alpha` argument to `adapted_rand_error`  (`#6472 <https://github.com/scikit-image/scikit-image/pull/6472>`_)
- Fix broken link to skimage.filters.sobel. (`#6474 <https://github.com/scikit-image/scikit-image/pull/6474>`_)
- Use broadcast_to instead of as_strided to generate broadcasted arrays. (`#6476 <https://github.com/scikit-image/scikit-image/pull/6476>`_)
- Update Ubuntu LTS version on Actions workflows (`#6478 <https://github.com/scikit-image/scikit-image/pull/6478>`_)
- changed image1 to moving_image in tvl1 parameter docs (`#6480 <https://github.com/scikit-image/scikit-image/pull/6480>`_)
- Use matplotlib.colormaps instead of deprecated cm.get_cmap in show_rag (`#6483 <https://github.com/scikit-image/scikit-image/pull/6483>`_)
- Use context manager when possible (`#6484 <https://github.com/scikit-image/scikit-image/pull/6484>`_)
- Document inclusion criteria for new functionality in core developer guide (`#6488 <https://github.com/scikit-image/scikit-image/pull/6488>`_)
- Use pyplot.get_cmap for compatiblity with matplotlib 3.3 to 3.6 in in show_rag (`#6490 <https://github.com/scikit-image/scikit-image/pull/6490>`_)
- Replace reference to api_changes.rst with release_dev.rst (`#6495 <https://github.com/scikit-image/scikit-image/pull/6495>`_)
- Support float input to skimage.draw.rectangle() [`#4283 <https://github.com/scikit-image/scikit-image/pull/4283>`_] (`#6501 <https://github.com/scikit-image/scikit-image/pull/6501>`_)
- Find peaks at border with `peak_local_max with `exclude_border=0` (`#6502 <https://github.com/scikit-image/scikit-image/pull/6502>`_)
- Fix resize anti_aliazing default value when input dtype is integer and order == 0 (`#6503 <https://github.com/scikit-image/scikit-image/pull/6503>`_)
- Add Github actions/stale to label "dormant" issues and PRs (`#6506 <https://github.com/scikit-image/scikit-image/pull/6506>`_)
- Clarify header pointing to notes for latest version released. (`#6508 <https://github.com/scikit-image/scikit-image/pull/6508>`_)
- Reduce ridge filters memory footprints (`#6509 <https://github.com/scikit-image/scikit-image/pull/6509>`_)
- Update benchmark environment to recent Python and NumPy versions (`#6511 <https://github.com/scikit-image/scikit-image/pull/6511>`_)
- Add new flag to convex_hull_image and grid_points_in_poly (`#6515 <https://github.com/scikit-image/scikit-image/pull/6515>`_)
- relax label name comparison in benchmarks.yaml (`#6520 <https://github.com/scikit-image/scikit-image/pull/6520>`_)
- update plot_euler_number.py for maplotlib 3.6 compatibility (`#6522 <https://github.com/scikit-image/scikit-image/pull/6522>`_)
- Use mask during rescaling in segmentation.slic and improve handling of error cases (`#6525 <https://github.com/scikit-image/scikit-image/pull/6525>`_)
- make non-functional change to build.txt to fix cache issue on CircleCI (`#6528 <https://github.com/scikit-image/scikit-image/pull/6528>`_)
- update setup.cfg field from license_file to license_files (`#6529 <https://github.com/scikit-image/scikit-image/pull/6529>`_)
- Fix wrong doc on connected pixels in flood (`#6534 <https://github.com/scikit-image/scikit-image/pull/6534>`_)
- Minor doc fix: add missing print statement in the `plot_segmentations.py` example (`#6535 <https://github.com/scikit-image/scikit-image/pull/6535>`_)
- Apply codespell to fix common spelling mistakes (`#6537 <https://github.com/scikit-image/scikit-image/pull/6537>`_)
- Ignore codespell fixes with git blame (`#6539 <https://github.com/scikit-image/scikit-image/pull/6539>`_)
- Add missing spaces to regionprops error message. (`#6545 <https://github.com/scikit-image/scikit-image/pull/6545>`_)
- Update "Mark dormant issues" workflow (`#6546 <https://github.com/scikit-image/scikit-image/pull/6546>`_)
- Add missing space in math directive in normalized_mutual_information's docstring (`#6549 <https://github.com/scikit-image/scikit-image/pull/6549>`_)
- Add missing option stale-pr-label for "Mark dormant issues" workflow (`#6552 <https://github.com/scikit-image/scikit-image/pull/6552>`_)
- Remove FUNDING.yml in preference of org version (`#6553 <https://github.com/scikit-image/scikit-image/pull/6553>`_)

56 authors added to this release [alphabetical by first name or login]
----------------------------------------------------------------------
- Adeel Hassan
- Albert Y. Shih
- AleixBP
- Alexandr Kalinin
- Alexandre de Siqueira
- Antony Lee
- Balint Varga
- Ben Greiner
- bsmietanka
- Chris Roat
- Chris Wood
- Dave Mellert
- Dudu Lasry
- Elena Pascal
- Fabian Schneider
- Frank A. Krueger
- Gregory Lee
- Hande Gözükan
- Jacob Rosenthal
- James Gao
- Jan Kadlec
- Jan-Hendrik Müller
- Jan-Lukas Wynen
- Jarrod Millman
- johnthagen
- Joshua Newton
- Juan DF
- Juan Nunez-Iglesias
- Judd Storrs
- Larry Bradley
- Lars Grüter
- Lucas Johnson
- maldil (maldil)
- Marianne Corvellec
- Mark Harfouche
- Marvin Albert
- Miles Lucas
- Naveen
- Preston Buscay
- Peter Bell
- Ray Bell
- Riadh Fezzani
- Robin Thibaut
- Ross Barnowski
- Sandeep N Menon
- Sanghyeok Hyun
- Sebastian Wallkötter
- Simon-Martin Schröder
- Stefan van der Walt
- Teemu Kumpumäki
- Thomas Voigtmann
- Tim-Oliver Buchholz
- Tyler Reddy


30 reviewers added to this release [alphabetical by first name or login]
------------------------------------------------------------------------
- Abhijeet Parida
- Albert Y. Shih
- Alexandre de Siqueira
- Antony Lee
- Ben Greiner
- Carlo
- Chris Roat
- Dudu Lasry
- François Boulogne
- Gregory Lee
- Jacob Rosenthal
- James Gao
- Jan-Hendrik Müller
- Jarrod Millman
- Juan DF
- Juan Nunez-Iglesias
- Lars Grüter
- maldil
- Marianne Corvellec
- Mark Harfouche
- Marvin Albert
- Riadh Fezzani
- Robert Haase
- Robin Thibaut
- Sandeep N Menon
- Sanghyeok Hyun
- Sebastian Wallkötter
- Stefan van der Walt
- Thomas Voigtmann
- Tim-Oliver Buchholz


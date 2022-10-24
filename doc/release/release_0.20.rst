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
- Add the isotropic binary morphological operators ``isotropic_closing``, ``isotropic_dilation``, ``isotropic_erosion``, and ``isotropic_opening`` in ``skimage.morphology``.
  These function return the same results as their non-isotropic counterparts but perform faster for large circular structuring elements
  (`#6492 <https://github.com/scikit-image/scikit-image/pull/6492>`_).
- Support footprint decomposition to several footprint generating and consuming functions in ``skimage.morphology``.
  By decomposing a footprint into several smaller ones, morphological operations can potentially be sped up.
  The decomposed footprint can be generated with the new ``decomposition`` parameter of the functions ``rectangle``, ``diamond``, ``disk``, ``cube``, ``octahedron``, ``ball``, and ``octagon`` in ``skimage.morphology``.
  The ``footprint`` parameter of the functions ``binary_erosion``, ``binary_dilation``, ``binary_opening``, ``binary_closing``, ``erosion``, ``dilation``, ``opening``, ``closing``, ``white_tophat``, and ``black_tophat`` in ``skimage.morphology`` now accepts a sequence of 2-element tuples ``(footprint_i, num_iter_i)`` where each entry, ``i``, of the sequence contains a footprint and the number of times it should be iteratively applied. This is the form produced by the footprint decompositions mentioned above
  (`#5482 <https://github.com/scikit-image/scikit-image/pull/5482>`_, `#6151 <https://github.com/scikit-image/scikit-image/pull/6151>`_).
- Support anisotropic images with different voxel spacings.
  Spacings can be defined with the new parameter ``spacing`` of the following functions in ``skimage.measure``: ``regionprops``, ``regionprops_table``, ``moments``, ``moments_central``, ``moments_normalized``, ``centroid``, ``inertia_tensor``, and ``inertia_tensor_eigvals``.
  Voxel spacing is taken into account for the following existing properties in ``skimage.measure.regionprops``: ``area``, ``area_bbox``, ``centroid``, ``area_convex``, ``extent``, ``feret_diameter_max``, ``area_filled``, ``inertia_tensor``, ``moments``, ``moments_central``, ``moments_hu``, ``moments_normalized``, ``perimeter``, ``perimeter_crofton``, ``solidity``, ``moments_weighted_central``, and ``moments_weighted_hu``.
  The new properties ``num_pixels`` and ``coords_scaled`` are available as well.
  See the respective docstrings for more details
  (`#6296 <https://github.com/scikit-image/scikit-image/pull/6296>`_).
- Support the Modified Hausdorff Distance (MHD) metric in ``skimage.metrics.hausdorff_distance`` via the new parameter ``method``.
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
- Support n-dimensional images in ``skimage.filters.farid`` (Farid & Simoncelli filter)
  (`#6257 <https://github.com/scikit-image/scikit-image/pull/6257>`_).
- Support n-dimensional images in ``skimage.restoration.wiener``
  (`#6454 <https://github.com/scikit-image/scikit-image/pull/6454>`_).
- Support the construction of ``skimage.io.ImageCollection`` from a ``load_pattern`` with an arbitrary sequence as long as a matching ``load_func`` is provided
  (`#6276 <https://github.com/scikit-image/scikit-image/pull/6276>`_).
- Warn for non-integer image inputs to ``skimage.feature.local_binary_pattern``.
  Applying the function to floating-point images may give unexpected results when small numerical differences between adjacent pixels are present
  (`#6272 <https://github.com/scikit-image/scikit-image/pull/6272>`_).
- Use the minimal required unsigned integer size in ``skimage.filters.rank_order`` which allows to operate the function with higher precision or on larger arrays.
  Previously, the returned ``labels`` and and ``original_values`` were always of type uint32.
  (`#6342 <https://github.com/scikit-image/scikit-image/pull/6342>`_).
- Use the minimal required unsigned integer internally in ``skimage.morphology.reconstruction`` which allows to operate the function with higher precision or on larger arrays.
  Previously, int32 was used.
  (`#6342 <https://github.com/scikit-image/scikit-image/pull/6342>`_).
- Improve histogram matching performance on unsigned integer data with ``skimage.exposure.match_histograms``.
  (`#6209 <https://github.com/scikit-image/scikit-image/pull/6209>`_, `#6354 <https://github.com/scikit-image/scikit-image/pull/6354>`_).
- Support three dimensions for the properties ``rotation`` and ``translation`` in ``skimage.transform.EuclideanTransform`` as well as for ``skimage.transform.SimilarityTransform.scale``
  (`#6367 <https://github.com/scikit-image/scikit-image/pull/6367>`_).
- Improve performance (~2x speedup) of ``skimage.feature.canny`` by porting a part of its implementation to Cython
  (`#6387 <https://github.com/scikit-image/scikit-image/pull/6387>`_).
- Improve performance (~2x speedup) of ``skimage.feature.hessian_matrix_eigvals`` and 2D ``skimage.feature.structure_tensor_eigenvalues``
  (`#6441 <https://github.com/scikit-image/scikit-image/pull/6441>`_).
- Add new parameter ``alpha`` to ``skimage.metrics.adapted_rand_error`` allowing control over the weight given to precision and recall
  (`#6472 <https://github.com/scikit-image/scikit-image/pull/6472>`_).
- Add new parameter ``binarize`` to ``skimage.measure.grid_points_in_poly`` to optionally return labels that tell whether a pixel is inside, outside or on the border of the polygon
  (`#6515 <https://github.com/scikit-image/scikit-image/pull/6515>`_).
  Add new parameter ``include_borders`` to ``skimage.measure.convex_hull_image`` to optionally exclude vertices or edges from the final hull mask
  (`#6515 <https://github.com/scikit-image/scikit-image/pull/6515>`_).
- Reduce the memory footprint of the ridge filters ``meijering``, ``sato``, ``frangi``, and ``hessian`` in ``skimage.filters``
  (`#6509 <https://github.com/scikit-image/scikit-image/pull/6509>`_).
- Improve performance of ``skimage.measure.moments_central`` by avoiding redundant computations
  (`#6188 <https://github.com/scikit-image/scikit-image/pull/6188>`_).
- Reduce import time of ``skimage.io`` by loading the matplotlib plugin only when required
  (`#6550 <https://github.com/scikit-image/scikit-image/pull/6550>`_).

Changes and new deprecations
----------------------------
- Rewrite ``skimage.filters.meijering``, ``skimage.filters.sato``,
  ``skimage.filters.frangi``, and ``skimage.filters.hessian`` to match the published algorithms more closely.
  This change is backward incompatible and will lead to different output values compared to the previous implementation.
  The Hessian matrix calculation is now done more accurately.
  The filters will now correctly be set to zero whenever one of the hessian eigenvalues has a sign which is incompatible with a ridge of the desired polarity.
  The gamma constant of the Frangi filter is now set adaptively based on the maximum Hessian norm
  (`#6446 <https://github.com/scikit-image/scikit-image/pull/6446>`_).
- Return ``False`` in ``skimage.measure.LineModelND.estimate`` instead of raising an error if the model is under-determined.
  Return ``False`` in ``skimage.measure.CircleModel.estimate`` instead of warning if the model is under-determined
  (`#6453 <https://github.com/scikit-image/scikit-image/pull/6453>`_).
- Rename ``skimage.filter.inverse`` to ``skimage.filter.inverse_filter``.
  ``skimage.filter.inverse`` is deprecated and will be removed in the next release
  (`#6418 <https://github.com/scikit-image/scikit-image/pull/6418>`_).
- Update minimal supported dependencies to ``numpy>=1.20``
  (`#6565 <https://github.com/scikit-image/scikit-image/pull/6565>`_).
- Update minimal supported dependencies to ``scipy>=1.8``
  (`#6564 <https://github.com/scikit-image/scikit-image/pull/6564>`_).
- Update minimal supported dependencies to ``networkx>=2.8``
  (`#6564 <https://github.com/scikit-image/scikit-image/pull/6564>`_).
- Update minimal supported dependency to ``pillow>=9.0.1``
  (`#6402 <https://github.com/scikit-image/scikit-image/pull/6402>`_).
- Update optional, minimal supported dependency to ``matplotlib>=3.3``
  (`#6383 <https://github.com/scikit-image/scikit-image/pull/6383>`_).

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
- Fix round-off error in ``skimage.exposure.adjust_gamma``
  (`#6285 <https://github.com/scikit-image/scikit-image/pull/6285>`_).
- Round and convert output coordinates of ``skimage.draw.rectangle`` to ``int`` even if the input coordinates use ``float``.
  This fix ensures that the output can be used for indexing similar to other draw functions
  (`#6501 <https://github.com/scikit-image/scikit-image/pull/6501>`_).
- Avoid unexpected exclusion of peaks near the image border in ``skimage.feature.peak_local_max`` if the peak value is smaller 0
  (`#6502 <https://github.com/scikit-image/scikit-image/pull/6502>`_).
- Avoid anti-aliasing in ``skimage.transform.resize`` by default when using nearest neighbor interpolation (``order == 0``) with an integer input data type
  (`#6503 <https://github.com/scikit-image/scikit-image/pull/6503>`_).
- Use mask during rescaling in ``skimage.segmentation.slic``.
  Previously, the mask was ignored when rescaling the image to make choice of compactness insensitive to the image values.
  The new behavior makes it possible to mask values such as `numpy.nan` or `numpy.infinity`.
  Additionally, raise an error if the input ``image`` has two dimensions and a ``channel_axis`` is specified - indicating that the image is multi-channel
  (`#6525 <https://github.com/scikit-image/scikit-image/pull/6525>`_).
- Fix unexpected error when passing a tuple to the parameter ``exclude_border`` in ``skimage.feature.blog_dog`` and ``skimage.feature.blob_log``
  (`#6533 <https://github.com/scikit-image/scikit-image/pull/6533>`_).
- Raise a specific error message in ``skimage.segmentation.random_walker`` if no seeds are provided as positive values in the parameter ``labels``
  (`#6562 <https://github.com/scikit-image/scikit-image/pull/6562>`_).
- Raise a specific error message when accessing region properties from ``skimage.measure.regionprops`` when the required  ``intensity_image`` is unavailable
  (`#6584 <https://github.com/scikit-image/scikit-image/pull/6584>`_).
- Avoid errors in ``skimage.feature.ORB.detect_and_extract`` by breaking early if the octave image is too small
  (`#6590 <https://github.com/scikit-image/scikit-image/pull/6590>`_).

Documentation
-------------
- Add a textbook-like tutorial on measuring fluorescence at the nuclear envelope of a cell
  (`#5262 <https://github.com/scikit-image/scikit-image/pull/5262>`_).
- Add new gallery example on decomposing flat footprints
  (`#6151 <https://github.com/scikit-image/scikit-image/pull/6151>`_).
- Add a new and gallery example "Butterworth Filters" and improve docstring of ``skimage.filters.butterworth``
  (`#6251 <https://github.com/scikit-image/scikit-image/pull/6251>`_).
- Add a new gallery example "Render text onto an image"
  (`#6431 <https://github.com/scikit-image/scikit-image/pull/6431>`_).
- Improve the the gallery example "Measure perimeters with different estimators"
  (`#6200 <https://github.com/scikit-image/scikit-image/pull/6200>`_, `#6121 <https://github.com/scikit-image/scikit-image/pull/6121>`_).
- Adapt the gallery example "Build image pyramids" to more diversified shaped images and downsample factors
  (`#6293 <https://github.com/scikit-image/scikit-image/pull/6293>`_).
- Add ``SUPPORT.md`` to repository to help users from GitHub find appropriate support
  resources
  (`#6171 <https://github.com/scikit-image/scikit-image/pull/6171>`_).
- Add ``CITATION.bib`` to repository to help with citing scikit-image
  (`#6195 <https://github.com/scikit-image/scikit-image/pull/6195>`_).
- Stop using the ``git://`` connection protocol and remove references to it
  (`#6201 <https://github.com/scikit-image/scikit-image/pull/6201>`_, `#6283 <https://github.com/scikit-image/scikit-image/pull/6283>`_).
- Fix formatting in the docstring of ``skimage.metrics.hausdorff_distance``
  (`#6203 <https://github.com/scikit-image/scikit-image/pull/6203>`_).
- Tweak ``balance`` in the docstring example of ``skimage.restoration.wiener`` for a less blurry result
  (`#6265 <https://github.com/scikit-image/scikit-image/pull/6265>`_).
- Change "neighbour" to EN-US spelling "neighbor"
  (`#6204 <https://github.com/scikit-image/scikit-image/pull/6204>`_).
- Update scikit-image's mailing addresses to the new domain discuss.scientific-python.org
  (`#6255 <https://github.com/scikit-image/scikit-image/pull/6255>`_).
- Clarify that the enabled ``watershed_line`` parameter will not catch borders between adjacent marker regions in ``skimage.segmentation.watershed``
  (`#6280 <https://github.com/scikit-image/scikit-image/pull/6280>`_).
- Describe the behavior of ``skimage.io.MultiImage`` more precisely in its docstring
  (`#6290 <https://github.com/scikit-image/scikit-image/pull/6290>`_, `#6292 <https://github.com/scikit-image/scikit-image/pull/6292>`_).
- Use gridded thumbnails in our gallery to demonstrate the different images and datasets available in ``skimage.data``
  (`#6298 <https://github.com/scikit-image/scikit-image/pull/6298>`_, `#6300 <https://github.com/scikit-image/scikit-image/pull/6300>`_, `#6301 <https://github.com/scikit-image/scikit-image/pull/6301>`_).
- Clarify that ``skimage.morphology.skeletonize`` accepts an ``image`` of any input type
  (`#6322 <https://github.com/scikit-image/scikit-image/pull/6322>`_).
- Document support for Path objects in ``skimage.io.imread`` and ``skimage.io.imsave``
  (`#6361 <https://github.com/scikit-image/scikit-image/pull/6361>`_).
- Improve error message in ``skimage.filters.threshold_multiotsu`` if the discretized image cannot be thresholded
  (`#6375 <https://github.com/scikit-image/scikit-image/pull/6375>`_).
- Show original unlabeled image as well in the gallery example "Expand segmentation labels without overlap"
  (`#6396 <https://github.com/scikit-image/scikit-image/pull/6396>`_).
- Add missing copyrights to LICENSE.txt and use formatting according to SPDX identifiers
  (`#6419 <https://github.com/scikit-image/scikit-image/pull/6419>`_).
- Document the refactoring of ``grey*`` to ``skimage.feature.graymatrix`` and ``skimage.feature.graycoprops`` in the release 0.19
  (`#6420 <https://github.com/scikit-image/scikit-image/pull/6420>`_).
- Document inclusion criteria for new functionality in core developer guide
  (`#6488 <https://github.com/scikit-image/scikit-image/pull/6488>`_).
- Fix description of ``connectivity`` parameter in the docstring of ``skimage.morphology.flood``
  (`#6534 <https://github.com/scikit-image/scikit-image/pull/6534>`_).
- Print the number of segments after applying the Watershed in the gallery example "Comparison of segmentation and superpixel algorithms"
  (`#6535 <https://github.com/scikit-image/scikit-image/pull/6535>`_).
- Include ``skimage.morphology.footprint_from_sequence`` in the public API documentation
  (`#6555 <https://github.com/scikit-image/scikit-image/pull/6555>`_).
- Fix typo in docstring of ``skimage.measure.moments_hu``
  (`#6016 <https://github.com/scikit-image/scikit-image/pull/6016>`_).
- Fix formatting of mode parameter in ``skimage.util.random_noise``
  (`#6532 <https://github.com/scikit-image/scikit-image/pull/6532>`_).
- List only the two primary OS-independent methods of installing scikit-image
  (`#6557 <https://github.com/scikit-image/scikit-image/pull/6557>`_, `#6560 <https://github.com/scikit-image/scikit-image/pull/6560>`_).
- Remove references to deprecated mailing list in ``doc/source/user_guide/getting_help.rst``
  (`#6575 <https://github.com/scikit-image/scikit-image/pull/6575>`_).
- Update support page on GitHub (``.github/SUPPORT.md``)
  (`#6575 <https://github.com/scikit-image/scikit-image/pull/6575>`_).
- Correct note about return type in the docstring of ``skimage.exposure.rescale_intensity``
  (`#6582 <https://github.com/scikit-image/scikit-image/pull/6582>`_).

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
- Move ``skimage/measure/mc_meta`` folder into ``tools/precompute/`` folder to avoid its unnecessary distribution to users
  (`#6294 <https://github.com/scikit-image/scikit-image/pull/6294>`_).
- Remove unused function ``getLutNames`` in ``tools/precompute/mc_meta/createluts.py``
  (`#6294 <https://github.com/scikit-image/scikit-image/pull/6294>`_).
- Point urls for data files to a specific commit
  (`#6297 <https://github.com/scikit-image/scikit-image/pull/6297>`_).
- Drop Codecov badge from project README
  (`#6302 <https://github.com/scikit-image/scikit-image/pull/6302>`_).
- Use ``cnp.float32_t`` and ``cnp.float64_t`` over ``float`` and ``double`` in Cython code
  (`#6303 <https://github.com/scikit-image/scikit-image/pull/6303>`_).
- Remove undefined reference to ``'python_to_notebook'`` in ``doc/ext/notebook_doc.py``
  (`#6307 <https://github.com/scikit-image/scikit-image/pull/6307>`_).
- Fix CI by excluding Pillow 9.1.0
  (`#6315 <https://github.com/scikit-image/scikit-image/pull/6315>`_).
- Parameterize tests in ``skimage.measure.tests.test_moments``
  (`#6323 <https://github.com/scikit-image/scikit-image/pull/6323>`_).
- Avoid unnecessary copying in ``skimage.morphology.skeletonize`` and update code style and tests
  (`#6327 <https://github.com/scikit-image/scikit-image/pull/6327>`_).
- Add draft of SKIP 4 "Transitioning to scikit-image 2.0"
  (`#6339 <https://github.com/scikit-image/scikit-image/pull/6339>`_, `#6353 <https://github.com/scikit-image/scikit-image/pull/6353>`_).
- Add benchmarks for ``skimage.morphology.reconstruction``
  (`#6342 <https://github.com/scikit-image/scikit-image/pull/6342>`_).
- Fixing typo in ``_probabilistic_hough_line``
  (`#6373 <https://github.com/scikit-image/scikit-image/pull/6373>`_).
- Remove reference to ``marching_cubes_lewiner`` from ``plot_marching_cubes.py``
  (`#6377 <https://github.com/scikit-image/scikit-image/pull/6377>`_).
- Pin pip pip to <22.1 in ``tools/github/before_install.sh``
  (`#6379 <https://github.com/scikit-image/scikit-image/pull/6379>`_).
- Update GH actions from v2 to v3
  (`#6382 <https://github.com/scikit-image/scikit-image/pull/6382>`_).
- Exclude pillow 9.1.1 from supported requirements
  (`#6384 <https://github.com/scikit-image/scikit-image/pull/6384>`_).
- Derive OBJECT_COLUMNS from COL_DTYPES in ``skimage.measure._regionprops``
  (`#6389 <https://github.com/scikit-image/scikit-image/pull/6389>`_).
- Support ``loadtxt`` of NumPy 1.23 with ``skimage/feature/orb_descriptor_positions.txt``
  (`#6400 <https://github.com/scikit-image/scikit-image/pull/6400>`_).
- Update to supported CircleCI images
  (`#6401 <https://github.com/scikit-image/scikit-image/pull/6401>`_).
- Use artifact-redirector
  (`#6407 <https://github.com/scikit-image/scikit-image/pull/6407>`_).
- Use the same numpy version dependencies for building as used by default
  (`#6409 <https://github.com/scikit-image/scikit-image/pull/6409>`_).
- Forward-port 0.19.3 release notes
  (`#6416 <https://github.com/scikit-image/scikit-image/pull/6416>`_).
- Forward-port gh-6369: Fix windows wheels: use vsdevcmd.bat to make sure rc.exe is on the path
  (`#6417 <https://github.com/scikit-image/scikit-image/pull/6417>`_).
- Use "center" in favor of "centre", and "color" in favor of "colour" gallery examples
  (`#6421 <https://github.com/scikit-image/scikit-image/pull/6421>`_, `#6422 <https://github.com/scikit-image/scikit-image/pull/6422>`_).
- Restrict GitHub Actions permissions to required ones
  (`#6426 <https://github.com/scikit-image/scikit-image/pull/6426>`_, `#6504 <https://github.com/scikit-image/scikit-image/pull/6504>`_).
- Exclude submodules of ``doc.*`` from package install
  (`#6428 <https://github.com/scikit-image/scikit-image/pull/6428>`_).
- Substitute deprecated ``vertices`` with ``simplices`` in ``skimage.transform._geometric``
  (`#6430 <https://github.com/scikit-image/scikit-image/pull/6430>`_).
- Fix minor typo in ``skimage.filters.sato``
  (`#6434 <https://github.com/scikit-image/scikit-image/pull/6434>`_).
- Simplify sort-by-absolute-value in ridge filters
  (`#6440 <https://github.com/scikit-image/scikit-image/pull/6440>`_).
- Removed completed items in ``TODO.txt``
  (`#6442 <https://github.com/scikit-image/scikit-image/pull/6442>`_).
- Fix broken links in SKIP 3
  (`#6445 <https://github.com/scikit-image/scikit-image/pull/6445>`_).
- Remove duplicate import in ``skimage.feature._canny``
  (`#6457 <https://github.com/scikit-image/scikit-image/pull/6457>`_).
- Use ``with open(...) as f`` instead of ``f = open(...)``
  (`#6458 <https://github.com/scikit-image/scikit-image/pull/6458>`_).
- Fix broken link in docstring of ``skimage.filters.sobel``
  (`#6474 <https://github.com/scikit-image/scikit-image/pull/6474>`_).
- Use ``broadcast_to`` instead of ``as_strided`` to generate broadcasted arrays
  (`#6476 <https://github.com/scikit-image/scikit-image/pull/6476>`_).
- Update to Ubuntu LTS version on Actions workflows
  (`#6478 <https://github.com/scikit-image/scikit-image/pull/6478>`_).
- Use ``moving_image`` in docstring of ``skimage.registration._optical_flow._tvl1``
  (`#6480 <https://github.com/scikit-image/scikit-image/pull/6480>`_).
- Use ``matplotlib.colormaps`` instead of deprecated ``cm.get_cmap`` in ``skimage.future.graph.show_rag``
  (`#6483 <https://github.com/scikit-image/scikit-image/pull/6483>`_).
- Use ``pyplot.get_cmap`` for compatiblity with matplotlib 3.3 to 3.6 in in ``skimage.future.graph.show_rag``
  (`#6490 <https://github.com/scikit-image/scikit-image/pull/6490>`_).
- Use context manager when possible
  (`#6484 <https://github.com/scikit-image/scikit-image/pull/6484>`_).
- Replace reference to ``api_changes.rst`` with ``release_dev.rst``
  (`#6495 <https://github.com/scikit-image/scikit-image/pull/6495>`_).
- Add Github actions/stale to label "dormant" issues and PRs
  (`#6506 <https://github.com/scikit-image/scikit-image/pull/6506>`_).
- Clarify header pointing to notes for latest version released
  (`#6508 <https://github.com/scikit-image/scikit-image/pull/6508>`_).
- Update benchmark environment to Python 3.10 and NumPy 1.23
  (`#6511 <https://github.com/scikit-image/scikit-image/pull/6511>`_).
- Relax label name comparison in benchmarks.yaml
  (`#6520 <https://github.com/scikit-image/scikit-image/pull/6520>`_).
- Update ``plot_euler_number.py`` for maplotlib 3.6 compatibility
  (`#6522 <https://github.com/scikit-image/scikit-image/pull/6522>`_).
- Make non-functional change to build.txt to fix cache issue on CircleCI
  (`#6528 <https://github.com/scikit-image/scikit-image/pull/6528>`_).
- Update deprecated field ``license_file`` to ``license_files`` in ``setup.cfg``
  (`#6529 <https://github.com/scikit-image/scikit-image/pull/6529>`_).
- Ignore codespell fixes with git blame
  (`#6539 <https://github.com/scikit-image/scikit-image/pull/6539>`_).
- Update "Mark dormant issues" workflow
  (`#6546 <https://github.com/scikit-image/scikit-image/pull/6546>`_).
- Add missing spaces to error mesage in ``skimage.measure.regionprops``
  (`#6545 <https://github.com/scikit-image/scikit-image/pull/6545>`_).
- Apply codespell to fix common spelling mistakes
  (`#6537 <https://github.com/scikit-image/scikit-image/pull/6537>`_).
- Add missing space in math directive in normalized_mutual_information's docstring
  (`#6549 <https://github.com/scikit-image/scikit-image/pull/6549>`_).
- Add missing option stale-pr-label for "Mark dormant issues" workflow
  (`#6552 <https://github.com/scikit-image/scikit-image/pull/6552>`_).
- Remove FUNDING.yml in preference of org version
  (`#6553 <https://github.com/scikit-image/scikit-image/pull/6553>`_).
- Forward port v0.19.1 and v0.19.2 release notes
  (`#6253 <https://github.com/scikit-image/scikit-image/pull/6253>`_).
- Handle pending changes to ``tifffile.imwrite`` defaults and avoid test warnings
  (`#6460 <https://github.com/scikit-image/scikit-image/pull/6460>`_).
- Replace issue templates with issue forms
  (`#6554 <https://github.com/scikit-image/scikit-image/pull/6554>`_, `#6576 <https://github.com/scikit-image/scikit-image/pull/6576>`_).
- Add linting via pre-commit
  (`#6563 <https://github.com/scikit-image/scikit-image/pull/6563>`_).
- Handle deprecation by updating to ``networkx.to_scipy_sparse_array``
  (`#6564 <https://github.com/scikit-image/scikit-image/pull/6564>`_).
- Update minimum supported numpy, scipy, and networkx
  (`#6385 <https://github.com/scikit-image/scikit-image/pull/6385>`_).
- Add CI tests for Python 3.11
  (`#6566 <https://github.com/scikit-image/scikit-image/pull/6566>`_).
- Fix CI for Scipy 1.9.2
  (`#6567 <https://github.com/scikit-image/scikit-image/pull/6567>`_).
- Apply linting results after enabling pre-commit in CI
  (`#6568 <https://github.com/scikit-image/scikit-image/pull/6568>`_).
- Refactor lazy loading to use stubs & lazy_loader package
  (`#6577 <https://github.com/scikit-image/scikit-image/pull/6577>`_).
- Provide pre-commit PR instructions
  (`#6578 <https://github.com/scikit-image/scikit-image/pull/6578>`_).
- Update sphinx configuration
  (`#6579 <https://github.com/scikit-image/scikit-image/pull/6579>`_).
- Test optional Py 3.10  dependencies on MacOS
  (`#6580 <https://github.com/scikit-image/scikit-image/pull/6580>`_).

.. Add multiscale structural similarity (`#6470 <https://github.com/scikit-image/scikit-image/pull/6470>`_) -> accidental empty merge, continued in #6487

TODO merged in milestone 0.21?
------------------------------
- Fix inpaint_biharmonic for images with Fortran-ordered memory layout (`#6263 <https://github.com/scikit-image/scikit-image/pull/6263>`_)
- Support array-likes consistently in geometric transforms (`#6270 <https://github.com/scikit-image/scikit-image/pull/6270>`_)

Backported 0.19.x
-----------------
- hough_line_peaks fix for corner case with optimal angle=0 (`#6271 <https://github.com/scikit-image/scikit-image/pull/6271>`_)
- Fix for error in 'Using Polar and Log-Polar Transformations for Registration' (#6304) (`#6306 <https://github.com/scikit-image/scikit-image/pull/6306>`_)
- Fix issue with newer versions of matplotlib in manual segmentation (`#6328 <https://github.com/scikit-image/scikit-image/pull/6328>`_)
- warp/rotate: fixed a bug with clipping when cval is not in the input range (`#6335 <https://github.com/scikit-image/scikit-image/pull/6335>`_)
- avoid warnings about change to v3 API from imageio (`#6343 <https://github.com/scikit-image/scikit-image/pull/6343>`_)
- Fix smoothed image computation when mask is None in canny (`#6348 <https://github.com/scikit-image/scikit-image/pull/6348>`_)
- Fix channel_axis default for cycle_spin (`#6352 <https://github.com/scikit-image/scikit-image/pull/6352>`_)
- remove use of deprecated kwargs from `test_tifffile_kwarg_passthrough` (`#6355 <https://github.com/scikit-image/scikit-image/pull/6355>`_)
- In newer PIL, palette may contain <256 entries (`#6405 <https://github.com/scikit-image/scikit-image/pull/6405>`_)
- Fix computation of histogram bins for multichannel integer-valued images (`#6413 <https://github.com/scikit-image/scikit-image/pull/6413>`_)
- Skip tests requiring fetched data (`#6089 <https://github.com/scikit-image/scikit-image/pull/6089>`_)
- Preserve backwards compatibility for `channel_axis` parameter in transform functions (`#6095 <https://github.com/scikit-image/scikit-image/pull/6095>`_)
- restore non-underscore functions in skimage.data (`#6097 <https://github.com/scikit-image/scikit-image/pull/6097>`_)
- forward port of #6098 (fix MacOS arm64 wheels and Windows Python 3.10 AMD64 wheel) (`#6101 <https://github.com/scikit-image/scikit-image/pull/6101>`_)
- make rank filter test comparisons robust across architectures (`#6103 <https://github.com/scikit-image/scikit-image/pull/6103>`_)
- pass a specific random_state into ransac in test_ransac_geometric (`#6105 <https://github.com/scikit-image/scikit-image/pull/6105>`_)
- Add linker flags to strip debug symbols during wheel building (`#6109 <https://github.com/scikit-image/scikit-image/pull/6109>`_)
- relax test condition to make it more robust to variable CI load (`#6114 <https://github.com/scikit-image/scikit-image/pull/6114>`_)
- respect SKIMAGE_TEST_STRICT_WARNINGS_GLOBAL setting in tests.yml (`#6118 <https://github.com/scikit-image/scikit-image/pull/6118>`_)
- bump deprecated Azure windows environment (`#6130 <https://github.com/scikit-image/scikit-image/pull/6130>`_)
- Update user warning message for viewer module. (`#6133 <https://github.com/scikit-image/scikit-image/pull/6133>`_)
- fix phase_cross_correlation typo (`#6139 <https://github.com/scikit-image/scikit-image/pull/6139>`_)
- Fix channel_axis handling in pyramid_gaussian and pyramid_laplace (`#6145 <https://github.com/scikit-image/scikit-image/pull/6145>`_)
- deprecate n_iter_max (should be max_num_iter) (`#6148 <https://github.com/scikit-image/scikit-image/pull/6148>`_)
- specify python version used by mybinder.org for gallery demos (`#6152 <https://github.com/scikit-image/scikit-image/pull/6152>`_)
- Fix unintended change to output dtype of match_histograms (`#6169 <https://github.com/scikit-image/scikit-image/pull/6169>`_)
- Fix decorators warnings stacklevel (`#6183 <https://github.com/scikit-image/scikit-image/pull/6183>`_)
- Fix SIFT wrong octave indices + typo (`#6184 <https://github.com/scikit-image/scikit-image/pull/6184>`_)
- Fix issue6190 - inconsistent default parameters in pyramids.py (`#6191 <https://github.com/scikit-image/scikit-image/pull/6191>`_)
- Always set params to nan when ProjectiveTransform.estimate fails (`#6207 <https://github.com/scikit-image/scikit-image/pull/6207>`_)
- PiecewiseAffineTransform.estimate return should reflect underlying transforms (`#6211 <https://github.com/scikit-image/scikit-image/pull/6211>`_)
- EuclideanTransform.estimate should return False when NaNs are present (`#6214 <https://github.com/scikit-image/scikit-image/pull/6214>`_)
- Allow the output_shape argument to be any iterable for resize and resize_local_mean (`#6219 <https://github.com/scikit-image/scikit-image/pull/6219>`_)
- Update filename in testing instructions. (`#6223 <https://github.com/scikit-image/scikit-image/pull/6223>`_)
- Fix calculation of Z normal in marching cubes (`#6227 <https://github.com/scikit-image/scikit-image/pull/6227>`_)
- Remove redundant testing on Appveyor (`#6229 <https://github.com/scikit-image/scikit-image/pull/6229>`_)
- Update imports/refs from deprecated scipy.ndimage.filters namespace (`#6231 <https://github.com/scikit-image/scikit-image/pull/6231>`_)
- Include Cython sources via package_data (`#6232 <https://github.com/scikit-image/scikit-image/pull/6232>`_)
- DOC: fix SciPy intersphinx (`#6239 <https://github.com/scikit-image/scikit-image/pull/6239>`_)
- Fix bug in SLIC superpixels with `enforce_connectivity=True` and `start_label > 0` (`#6242 <https://github.com/scikit-image/scikit-image/pull/6242>`_)
- Fowardport PR #6249 on branch main (update MacOS libomp installation in wheel building script) (`#6250 <https://github.com/scikit-image/scikit-image/pull/6250>`_)
- Ignore sparse matrix deprecation warning (`#6261 <https://github.com/scikit-image/scikit-image/pull/6261>`_)

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
- kwikwag (kwikwag)
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

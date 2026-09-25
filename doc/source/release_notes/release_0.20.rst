scikit-image 0.20.0 (2023-02-28)
================================

scikit-image is an image processing toolbox built on SciPy that
includes algorithms for segmentation, geometric transformations, color
space manipulation, analysis, filtering, morphology, feature
detection, and more.

For more information, examples, and documentation, please visit our website:
https://scikit-image.org

With this release, many of the functions in ``skimage.measure`` now support
anisotropic images with different voxel spacings.

Many performance improvements were made, such as support for footprint
decomposition in ``skimage.morphology``

Four new gallery examples were added to the documentation, including
the new interactive example "Track solidification of a metallic
alloy".

This release completes the transition to a more flexible
``channel_axis`` parameter for indicating multi-channel images, and
includes several other deprecations that make the API more consistent and
expressive.

Finally, in preparation for the removal of ``distutils`` in the upcoming
Python 3.12 release, we replaced our build system with `meson` and a
static `pyproject.toml` specification.

This release supports Python 3.8–3.11.

New features and improvements
-----------------------------
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
- Add isotropic binary morphological operators ``isotropic_closing``, ``isotropic_dilation``, ``isotropic_erosion``, and ``isotropic_opening`` in ``skimage.morphology``.
  These functions return the same results as their non-isotropic counterparts but perform faster for large circular structuring elements
  (`#6492 <https://github.com/scikit-image/scikit-image/pull/6492>`_).
- Add new colocalization metrics ``pearson_corr_coeff``, ``manders_coloc_coeff``, ``manders_overlap_coeff`` and ``intersection_coeff`` to ``skimage.measure``
  (`#6189 <https://github.com/scikit-image/scikit-image/pull/6189>`_).
- Support the Modified Hausdorff Distance (MHD) metric in ``skimage.metrics.hausdorff_distance`` via the new parameter ``method``.
  The MHD can be more robust against outliers than the directed Hausdorff Distance (HD)
  (`#5581 <https://github.com/scikit-image/scikit-image/pull/5581>`_).
- Add two datasets ``skimage.data.protein_transport`` and ``skimage.data.nickel_solidification``
  (`#6087 <https://github.com/scikit-image/scikit-image/pull/6087>`_).
- Add new parameter ``use_gaussian_derivatives`` to ``skimage.feature.hessian_matrix`` which allows the computation of the Hessian matrix by convolving with Gaussian derivatives
  (`#6149 <https://github.com/scikit-image/scikit-image/pull/6149>`_).
- Add new parameters ``squared_butterworth`` and ``npad`` to ``skimage.filters.butterworth``, which support traditional or squared filtering and edge padding, respectively
  (`#6251 <https://github.com/scikit-image/scikit-image/pull/6251>`_).
- Support construction of a ``skimage.io.ImageCollection`` from a ``load_pattern`` with an arbitrary sequence as long as a matching ``load_func`` is provided
  (`#6276 <https://github.com/scikit-image/scikit-image/pull/6276>`_).
- Add new parameter ``alpha`` to ``skimage.metrics.adapted_rand_error`` allowing control over the weight given to precision and recall
  (`#6472 <https://github.com/scikit-image/scikit-image/pull/6472>`_).
- Add new parameter ``binarize`` to ``skimage.measure.grid_points_in_poly`` to optionally return labels that tell whether a pixel is inside, outside, or on the border of the polygon
  (`#6515 <https://github.com/scikit-image/scikit-image/pull/6515>`_).
- Add new parameter ``include_borders`` to ``skimage.measure.convex_hull_image`` to optionally exclude vertices or edges from the final hull mask
  (`#6515 <https://github.com/scikit-image/scikit-image/pull/6515>`_).
- Add new parameter ``offsets`` to ``skimage.measure.regionprops`` that optionally allows specifying the coordinates of the origin and affects the properties ``coords_scaled`` and ``coords``
  (`#3706 <https://github.com/scikit-image/scikit-image/pull/3706>`_).
- Add new parameter ``disambiguate`` to ``skimage.registration.phase_cross_correlation`` to optionally disambiguate periodic shifts
  (`#6617 <https://github.com/scikit-image/scikit-image/pull/6617>`_).
- Support n-dimensional images in ``skimage.filters.farid`` (Farid & Simoncelli filter)
  (`#6257 <https://github.com/scikit-image/scikit-image/pull/6257>`_).
- Support n-dimensional images in ``skimage.restoration.wiener``
  (`#6454 <https://github.com/scikit-image/scikit-image/pull/6454>`_).
- Support three dimensions for the properties ``rotation`` and ``translation`` in ``skimage.transform.EuclideanTransform`` as well as for ``skimage.transform.SimilarityTransform.scale``
  (`#6367 <https://github.com/scikit-image/scikit-image/pull/6367>`_).
- Allow footprints with non-adjacent pixels as neighbors in ``skimage.morphology.flood_fill``
  (`#6236 <https://github.com/scikit-image/scikit-image/pull/6236>`_).
- Support array-likes consistently in ``AffineTransform``, ``EssentialMatrixTransform``, ``EuclideanTransform``, ``FundamentalMatrixTransform``, ``GeometricTransform``, ``PiecewiseAffineTransform``, ``PolynomialTransform``, ``ProjectiveTransform``, ``SimilarityTransform``, ``estimate_transform``, and ``matrix_transform`` in ``skimage.transform``
  (`#6270 <https://github.com/scikit-image/scikit-image/pull/6270>`_).

Performance
^^^^^^^^^^^
- Improve performance (~2x speedup) of ``skimage.feature.canny`` by porting a part of its implementation to Cython
  (`#6387 <https://github.com/scikit-image/scikit-image/pull/6387>`_).
- Improve performance (~2x speedup) of ``skimage.feature.hessian_matrix_eigvals`` and 2D ``skimage.feature.structure_tensor_eigenvalues``
  (`#6441 <https://github.com/scikit-image/scikit-image/pull/6441>`_).
- Improve performance of ``skimage.measure.moments_central`` by avoiding redundant computations
  (`#6188 <https://github.com/scikit-image/scikit-image/pull/6188>`_).
- Reduce import time of ``skimage.io`` by loading the matplotlib plugin only when required
  (`#6550 <https://github.com/scikit-image/scikit-image/pull/6550>`_).
- Incorporate RANSAC improvements from scikit-learn into ``skimage.measure.ransac`` which decrease the number of iterations
  (`#6046 <https://github.com/scikit-image/scikit-image/pull/6046>`_).
- Improve histogram matching performance on unsigned integer data with ``skimage.exposure.match_histograms``.
  (`#6209 <https://github.com/scikit-image/scikit-image/pull/6209>`_, `#6354 <https://github.com/scikit-image/scikit-image/pull/6354>`_).
- Reduce memory consumption of the ridge filters ``meijering``, ``sato``, ``frangi``, and ``hessian`` in ``skimage.filters``
  (`#6509 <https://github.com/scikit-image/scikit-image/pull/6509>`_).
- Reduce memory consumption of ``blob_dog``, ``blob_log``, and ``blob_doh`` in ``skimage.feature``
  (`#6597 <https://github.com/scikit-image/scikit-image/pull/6597>`_).
- Use minimal required unsigned integer size internally in ``skimage.morphology.reconstruction`` which allows to operate the function with higher precision or on larger arrays.
  Previously, int32 was used.
  (`#6342 <https://github.com/scikit-image/scikit-image/pull/6342>`_).
- Use minimal required unsigned integer size in ``skimage.filters.rank_order`` which allows to operate the function with higher precision or on larger arrays.
  Previously, the returned ``labels`` and ``original_values`` were always of type uint32.
  (`#6342 <https://github.com/scikit-image/scikit-image/pull/6342>`_).

Changes and new deprecations
----------------------------
- Set Python 3.8 as the minimal supported version
  (`#6679 <https://github.com/scikit-image/scikit-image/pull/6679>`_).
- Rewrite ``skimage.filters.meijering``, ``skimage.filters.sato``,
  ``skimage.filters.frangi``, and ``skimage.filters.hessian`` to match the published algorithms more closely.
  This change is backward incompatible and will lead to different output values compared to the previous implementation.
  The Hessian matrix calculation is now done more accurately.
  The filters will now be correctly set to zero whenever one of the Hessian eigenvalues has a sign which is incompatible with a ridge of the desired polarity.
  The gamma constant of the Frangi filter is now set adaptively based on the maximum Hessian norm
  (`#6446 <https://github.com/scikit-image/scikit-image/pull/6446>`_).
- Move functions in ``skimage.future.graph`` to ``skimage.graph``. This affects ``cut_threshold``, ``cut_normalized``, ``merge_hierarchical``, ``rag_mean_color``, ``RAG``, ``show_rag``, and ``rag_boundary``
  (`#6674 <https://github.com/scikit-image/scikit-image/pull/6674>`_).
- Return ``False`` in ``skimage.measure.LineModelND.estimate`` instead of raising an error if the model is under-determined
  (`#6453 <https://github.com/scikit-image/scikit-image/pull/6453>`_).
- Return ``False`` in ``skimage.measure.CircleModel.estimate`` instead of warning if the model is under-determined
  (`#6453 <https://github.com/scikit-image/scikit-image/pull/6453>`_).
- Rename ``skimage.filters.inverse`` to ``skimage.filters.inverse_filter``.
  ``skimage.filters.inverse`` is deprecated and will be removed in the next release
  (`#6418 <https://github.com/scikit-image/scikit-image/pull/6418>`_, `#6701 <https://github.com/scikit-image/scikit-image/pull/6701>`_).
- Update minimal supported dependencies to ``numpy>=1.20``
  (`#6565 <https://github.com/scikit-image/scikit-image/pull/6565>`_).
- Update minimal supported dependencies to ``scipy>=1.8``
  (`#6564 <https://github.com/scikit-image/scikit-image/pull/6564>`_).
- Update minimal supported dependencies to ``networkx>=2.8``
  (`#6564 <https://github.com/scikit-image/scikit-image/pull/6564>`_).
- Update minimal supported dependency to ``pillow>=9.0.1``
  (`#6402 <https://github.com/scikit-image/scikit-image/pull/6402>`_).
- Update minimal supported dependency to ``setuptools 67``
  (`#6754 <https://github.com/scikit-image/scikit-image/pull/6754>`_).
- Update optional, minimal supported dependency to ``matplotlib>=3.3``
  (`#6383 <https://github.com/scikit-image/scikit-image/pull/6383>`_).
- Warn for non-integer image inputs to ``skimage.feature.local_binary_pattern``.
  Applying the function to floating-point images may give unexpected results when small numerical differences between adjacent pixels are present
  (`#6272 <https://github.com/scikit-image/scikit-image/pull/6272>`_).
- Warn if ``skimage.registration.phase_cross_correlation`` returns only the shift vector.
  Starting with the next release this function will always return a tuple of three (shift vector, error, phase difference).
  Use ``return_error="always"`` to silence this warning and switch to this new behavior
  (`#6543 <https://github.com/scikit-image/scikit-image/pull/6543>`_).
- Warn in ``skimage.metrics.structural_similarity``, if ``data_range`` is not specified in case of floating point data
  (`#6612 <https://github.com/scikit-image/scikit-image/pull/6612>`_).
- Automatic detection of the color channel is deprecated in ``skimage.filters.gaussian`` and a warning is emitted if the parameter ``channel_axis`` is not set explicitly
  (`#6583 <https://github.com/scikit-image/scikit-image/pull/6583>`_).

Completed deprecations
----------------------
- Remove ``skimage.viewer`` which was scheduled for removal in the postponed version 1.0
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
- Remove deprecated parameter ``multichannel``; use ``channel_axis`` instead.
  This affects ``skimage.draw.random_shapes``, ``skimage.exposure.match_histograms``, ``skimage.feature.multiscale_basic_features``, ``skimage.feature.hog``, ``skimage.feature.difference_of_gaussians``, ``skimage.filters.unsharp_mask``, and ``skimage.metrics.structural_similarity``.
  In ``skimage.restoration``, this affects ``cycle_spin``, ``denoise_bilateral``, ``denoise_tv_bregman``, ``denoise_tv_chambolle``, ``denoise_wavelet``, ``estimate_sigma``, ``inpaint_biharmonic``, and ``denoise_nl_means``.
  In ``skimage.segmentation``, this affects ``felzenszwalb``, ``random_walker``, and ``slic``.
  In ``skimage.transform``, this affects ``rescale``, ``warp_polar``, ``pyramid_reduce``, ``pyramid_expand``, ``pyramid_gaussian``, and ``pyramid_laplacian``.
  In ``skimage.util``, this affects ``montage`` and ``apply_parallel``
  (`#6583 <https://github.com/scikit-image/scikit-image/pull/6583>`_).
- Remove deprecated parameter ``selem``; use ``footprint`` instead.
  In ``skimage.filters``, this affects ``median``, ``autolevel_percentile``, ``gradient_percentile``, ``mean_percentile``, ``subtract_mean_percentile``, ``enhance_contrast_percentile``, ``percentile``, ``pop_percentile``, ``sum_percentile``, ``threshold_percentile``, ``mean_bilateral``, ``pop_bilateral``, ``sum_bilateral``, ``autolevel``, ``equalize``, ``gradient``, ``maximum``, ``mean``, ``geometric_mean``, ``subtract_mean``, ``median``, ``minimum``, ``modal``, ``enhance_contrast``, ``pop``, ``sum``, ``threshold``, ``noise_filter``, ``entropy``, ``otsu``, ``windowed_histogram``, and ``majority``.
  In ``skimage.morphology``, this affects ``flood_fill``, ``flood``, ``binary_erosion``, ``binary_dilation``, ``binary_opening``, ``binary_closing``, ``h_maxima``, ``h_minima``, ``local_maxima``, ``local_minima``, ``erosion``, ``dilation``, ``opening``, ``closing``, ``white_tophat``, ``black_tophat``, and ``reconstruction``
  (`#6583 <https://github.com/scikit-image/scikit-image/pull/6583>`_).
- Remove deprecated parameter ``max_iter`` from ``skimage.filters.threshold_minimum``, ``skimage.morphology.thin``, and ``skimage.segmentation.chan_vese``;
  use ``max_num_iter`` instead
  (`#6583 <https://github.com/scikit-image/scikit-image/pull/6583>`_).
- Remove deprecated parameter ``max_iterations`` from ``skimage.segmentation.active_contour``;
  use ``max_num_iter`` instead
  (`#6583 <https://github.com/scikit-image/scikit-image/pull/6583>`_).
- Remove deprecated parameter ``input`` from ``skimage.measure.label``;
  use ``label_image`` instead
  (`#6583 <https://github.com/scikit-image/scikit-image/pull/6583>`_).
- Remove deprecated parameter ``coordinates`` from ``skimage.measure.regionprops`` and ``skimage.segmentation.active_contour``
  (`#6583 <https://github.com/scikit-image/scikit-image/pull/6583>`_).
- Remove deprecated parameter ``neighbourhood`` from ``skimage.measure.perimeter``;
  use ``neighborhood`` instead
  (`#6583 <https://github.com/scikit-image/scikit-image/pull/6583>`_).
- Remove deprecated parameters ``height`` and ``width`` from ``skimage.morphology.rectangle``;
  use ``ncols`` and ``nrows`` instead
  (`#6583 <https://github.com/scikit-image/scikit-image/pull/6583>`_).
- Remove deprecated parameter ``in_place`` from ``skimage.morphology.remove_small_objects``, ``skimage.morphology.remove_small_holes``, and ``skimage.segmentation.clear_border``; use ``out`` instead
  (`#6583 <https://github.com/scikit-image/scikit-image/pull/6583>`_).
- Remove deprecated parameter ``iterations`` from ``skimage.restoration.richardson_lucy``, ``skimage.segmentation.morphological_chan_vese``, and ``skimage.segmentation.morphological_geodesic_active_contour``; use ``num_iter`` instead
  (`#6583 <https://github.com/scikit-image/scikit-image/pull/6583>`_).
- Remove support for deprecated keys ``"min_iter"`` and ``"max_iter"`` in ``skimage.restoration.unsupervised_wiener``'s parameter ``user_params``; use ``"min_num_iter"`` and ``"max_num_iter"`` instead
  (`#6583 <https://github.com/scikit-image/scikit-image/pull/6583>`_).
- Remove deprecated functions ``greycomatrix`` and ``greycoprops`` from ``skimage.feature``
  (`#6583 <https://github.com/scikit-image/scikit-image/pull/6583>`_).
- Remove deprecated submodules ``skimage.morphology.grey`` and ``skimage.morphology.greyreconstruct``; use ``skimage.morphology`` instead
  (`#6583 <https://github.com/scikit-image/scikit-image/pull/6583>`_).
- Remove deprecated submodule ``skimage.morphology.selem``; use ``skimage.morphology.footprints`` instead
  (`#6583 <https://github.com/scikit-image/scikit-image/pull/6583>`_).
- Remove deprecated ``skimage.future.graph.ncut`` (it was replaced by ``skimage.graph.cut_normalized``)
  (`#6685 <https://github.com/scikit-image/scikit-image/pull/6685>`_).

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
- Fix ``skimage.restoration.inpaint_biharmonic`` for images with Fortran-ordered memory layout
  (`#6263 <https://github.com/scikit-image/scikit-image/pull/6263>`_).
- Fix automatic detection of the color channel in ``skimage.filters.gaussian`` (this behavior is deprecated, see new deprecations)
  (`#6583 <https://github.com/scikit-image/scikit-image/pull/6583>`_).
- Fix stacklevel of warning in ``skimage.color.lab2rgb``
  (`#6616 <https://github.com/scikit-image/scikit-image/pull/6616>`_).
- Fix the order of return values for ``skimage.feature.hessian_matrix`` and raise an error if ``order='xy'`` is requested for images with more than 2 dimensions
  (`#6624 <https://github.com/scikit-image/scikit-image/pull/6624>`_).
- Fix misleading exception in functions in ``skimage.filters.rank`` that did
  not mention that 2D images are also supported
  (`#6666 <https://github.com/scikit-image/scikit-image/pull/6666>`_).
- Fix in-place merging of wheights in ``skimage.graph.RAG.merge_nodes``
  (`#6692 <https://github.com/scikit-image/scikit-image/pull/6692>`_).
- Fix growing memory error and silence compiler warning in internal ``heappush`` function
  (`#6727 <https://github.com/scikit-image/scikit-image/pull/6727>`_).
- Fix compiliation warning about struct initialization in `Cascade.detect_multi_scale`
  (`#6728 <https://github.com/scikit-image/scikit-image/pull/6728>`_).

Documentation
-------------

New
^^^
- Add gallery example "Decompose flat footprints (structuring elements)"
  (`#6151 <https://github.com/scikit-image/scikit-image/pull/6151>`_).
- Add gallery example "Butterworth Filters" and improve docstring of ``skimage.filters.butterworth``
  (`#6251 <https://github.com/scikit-image/scikit-image/pull/6251>`_).
- Add gallery example "Render text onto an image"
  (`#6431 <https://github.com/scikit-image/scikit-image/pull/6431>`_).
- Add gallery example "Track solidification of a metallic alloy"
  (`#6469 <https://github.com/scikit-image/scikit-image/pull/6469>`_).
- Add gallery example "Colocalization metrics"
  (`#6189 <https://github.com/scikit-image/scikit-image/pull/6189>`_).
- Add support page (``.github/SUPPORT.md``) to help users from GitHub find appropriate support resources
  (`#6171 <https://github.com/scikit-image/scikit-image/pull/6171>`_, `#6575 <https://github.com/scikit-image/scikit-image/pull/6575>`_).
- Add ``CITATION.bib`` to repository to help with citing scikit-image
  (`#6195 <https://github.com/scikit-image/scikit-image/pull/6195>`_).
- Add usage instructions for new Meson-based build system with ``dev.py``
  (`#6600 <https://github.com/scikit-image/scikit-image/pull/6600>`_).

Improved & updated
^^^^^^^^^^^^^^^^^^
- Improve gallery example "Measure perimeters with different estimators"
  (`#6200 <https://github.com/scikit-image/scikit-image/pull/6200>`_, `#6121 <https://github.com/scikit-image/scikit-image/pull/6121>`_).
- Adapt gallery example "Build image pyramids" to more diversified shaped images and downsample factors
  (`#6293 <https://github.com/scikit-image/scikit-image/pull/6293>`_).
- Adapt gallery example "Explore 3D images (of cells)" with interactive slice explorer using plotly
  (`#4953 <https://github.com/scikit-image/scikit-image/pull/4953>`_).
- Clarify meaning of the ``weights`` term and rewrite docstrings of ``skimage.restoration.denoise_tv_bregman`` and ``skimage.restoration.denoise_tv_chambolle``
  (`#6544 <https://github.com/scikit-image/scikit-image/pull/6544>`_).
- Describe the behavior of ``skimage.io.MultiImage`` more precisely in its docstring
  (`#6290 <https://github.com/scikit-image/scikit-image/pull/6290>`_, `#6292 <https://github.com/scikit-image/scikit-image/pull/6292>`_).
- Clarify that the enabled ``watershed_line`` parameter will not catch borders between adjacent marker regions in ``skimage.segmentation.watershed``
  (`#6280 <https://github.com/scikit-image/scikit-image/pull/6280>`_).
- Clarify that ``skimage.morphology.skeletonize`` accepts an ``image`` of any input type
  (`#6322 <https://github.com/scikit-image/scikit-image/pull/6322>`_).
- Use gridded thumbnails in our gallery to demonstrate the different images and datasets available in ``skimage.data``
  (`#6298 <https://github.com/scikit-image/scikit-image/pull/6298>`_, `#6300 <https://github.com/scikit-image/scikit-image/pull/6300>`_, `#6301 <https://github.com/scikit-image/scikit-image/pull/6301>`_).
- Tweak ``balance`` in the docstring example of ``skimage.restoration.wiener`` for a less blurry result
  (`#6265 <https://github.com/scikit-image/scikit-image/pull/6265>`_).
- Document support for Path objects in ``skimage.io.imread`` and ``skimage.io.imsave``
  (`#6361 <https://github.com/scikit-image/scikit-image/pull/6361>`_).
- Improve error message in ``skimage.filters.threshold_multiotsu`` if the discretized image cannot be thresholded
  (`#6375 <https://github.com/scikit-image/scikit-image/pull/6375>`_).
- Show original unlabeled image as well in the gallery example "Expand segmentation labels without overlap"
  (`#6396 <https://github.com/scikit-image/scikit-image/pull/6396>`_).
- Document refactoring of ``grey*`` to ``skimage.feature.graymatrix`` and ``skimage.feature.graycoprops`` in the release 0.19
  (`#6420 <https://github.com/scikit-image/scikit-image/pull/6420>`_).
- Document inclusion criteria for new functionality in core developer guide
  (`#6488 <https://github.com/scikit-image/scikit-image/pull/6488>`_).
- Print the number of segments after applying the Watershed in the gallery example "Comparison of segmentation and superpixel algorithms"
  (`#6535 <https://github.com/scikit-image/scikit-image/pull/6535>`_).
- Replace issue templates with issue forms
  (`#6554 <https://github.com/scikit-image/scikit-image/pull/6554>`_, `#6576 <https://github.com/scikit-image/scikit-image/pull/6576>`_).
- Expand reviewer guidelines in pull request template
  (`#6208 <https://github.com/scikit-image/scikit-image/pull/6208>`_).
- Provide pre-commit PR instructions in pull request template
  (`#6578 <https://github.com/scikit-image/scikit-image/pull/6578>`_).
- Warn about and explain the handling of floating-point data in the docstring of ``skimage.metricts.structural_similarity``
  (`#6595 <https://github.com/scikit-image/scikit-image/pull/6595>`_).
- Fix intensity autoscaling in animated ``imshow`` in gallery example "Measure fluorescence intensity at the nuclear envelope"
  (`#6599 <https://github.com/scikit-image/scikit-image/pull/6599>`_).
- Clarify dependency on ``scikit-image[data]`` and pooch in ``INSTALL.rst``
  (`#6619 <https://github.com/scikit-image/scikit-image/pull/6619>`_).
- Don't use confusing loop in installation instructions for conda
  (`#6672 <https://github.com/scikit-image/scikit-image/pull/6672>`_).
- Document value ranges of L*a*b* and L*Ch in ``lab2xyz``, ``rgb2lab``, ``lab2lch``, and ``lch2lab`` in ``skimage.color``
  (`#6688 <https://github.com/scikit-image/scikit-image/pull/6688>`_, `#6697 <https://github.com/scikit-image/scikit-image/pull/6697>`_, `#6719 <https://github.com/scikit-image/scikit-image/pull/6719>`_).
- Use more consistent style in docstring of ``skimage.feature.local_binary_pattern``
  (`#6736 <https://github.com/scikit-image/scikit-image/pull/6736>`_).

Fixes, spelling & minor tweaks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Remove deprecated reference and use ``skimage.measure.marching_cubes`` in gallery example "Marching Cubes"
  (`#6377 <https://github.com/scikit-image/scikit-image/pull/6377>`_).
- List only the two primary OS-independent methods of installing scikit-image
  (`#6557 <https://github.com/scikit-image/scikit-image/pull/6557>`_, `#6560 <https://github.com/scikit-image/scikit-image/pull/6560>`_).
- Fix description of ``connectivity`` parameter in the docstring of ``skimage.morphology.flood``
  (`#6534 <https://github.com/scikit-image/scikit-image/pull/6534>`_).
- Fix formatting in the docstring of ``skimage.metrics.hausdorff_distance``
  (`#6203 <https://github.com/scikit-image/scikit-image/pull/6203>`_).
- Fix typo in docstring of ``skimage.measure.moments_hu``
  (`#6016 <https://github.com/scikit-image/scikit-image/pull/6016>`_).
- Fix formatting of mode parameter in ``skimage.util.random_noise``
  (`#6532 <https://github.com/scikit-image/scikit-image/pull/6532>`_).
- Fix broken links in SKIP 3
  (`#6445 <https://github.com/scikit-image/scikit-image/pull/6445>`_).
- Fix broken link in docstring of ``skimage.filters.sobel``
  (`#6474 <https://github.com/scikit-image/scikit-image/pull/6474>`_).
- Change "neighbour" to EN-US spelling "neighbor"
  (`#6204 <https://github.com/scikit-image/scikit-image/pull/6204>`_).
- Add missing copyrights to LICENSE.txt and use formatting according to SPDX identifiers
  (`#6419 <https://github.com/scikit-image/scikit-image/pull/6419>`_).
- Include ``skimage.morphology.footprint_from_sequence`` in the public API documentation
  (`#6555 <https://github.com/scikit-image/scikit-image/pull/6555>`_).
- Correct note about return type in the docstring of ``skimage.exposure.rescale_intensity``
  (`#6582 <https://github.com/scikit-image/scikit-image/pull/6582>`_).
- Stop using the ``git://`` connection protocol and remove references to it
  (`#6201 <https://github.com/scikit-image/scikit-image/pull/6201>`_, `#6283 <https://github.com/scikit-image/scikit-image/pull/6283>`_).
- Update scikit-image's mailing addresses to the new domain discuss.scientific-python.org
  (`#6255 <https://github.com/scikit-image/scikit-image/pull/6255>`_).
- Remove references to deprecated mailing list in ``doc/source/user_guide/getting_help.rst``
  (`#6575 <https://github.com/scikit-image/scikit-image/pull/6575>`_).
- Use "center" in favor of "centre", and "color" in favor of "colour" gallery examples
  (`#6421 <https://github.com/scikit-image/scikit-image/pull/6421>`_, `#6422 <https://github.com/scikit-image/scikit-image/pull/6422>`_).
- Replace reference to ``api_changes.rst`` with ``release_dev.rst``
  (`#6495 <https://github.com/scikit-image/scikit-image/pull/6495>`_).
- Clarify header pointing to notes for latest version released
  (`#6508 <https://github.com/scikit-image/scikit-image/pull/6508>`_).
- Add missing spaces to error message in ``skimage.measure.regionprops``
  (`#6545 <https://github.com/scikit-image/scikit-image/pull/6545>`_).
- Apply codespell to fix common spelling mistakes
  (`#6537 <https://github.com/scikit-image/scikit-image/pull/6537>`_).
- Add missing space in math directive in normalized_mutual_information's docstring
  (`#6549 <https://github.com/scikit-image/scikit-image/pull/6549>`_).
- Fix lengths of docstring heading underline in ``skimage.morphology.isotropic_`` functions
  (`#6628 <https://github.com/scikit-image/scikit-image/pull/6628>`_).
- Fix plot order due to duplicate examples with the file name ``plot_thresholding.py``
  (`#6644 <https://github.com/scikit-image/scikit-image/pull/6644>`_).
- Get rid of numpy deprecation warning in gallery example ``plot_equalize``
  (`#6650 <https://github.com/scikit-image/scikit-image/pull/6650>`_).
- Fix swapping of opening and closing in gallery example ``plot_rank_filters``
  (`#6652 <https://github.com/scikit-image/scikit-image/pull/6652>`_).
- Get rid of numpy deprecation warning in gallery example ``in plot_log_gamma.py``
  (`#6655 <https://github.com/scikit-image/scikit-image/pull/6655>`_).
- Remove warnings and unnecessary messages in gallery example "Tinting gray-scale images"
  (`#6656 <https://github.com/scikit-image/scikit-image/pull/6656>`_).
- Update the contribution guide to recommend creating the virtualenv outside the source tree
  (`#6675 <https://github.com/scikit-image/scikit-image/pull/6675>`_).
- Fix typo in docstring of ``skimage.data.coffee``
  (`#6740 <https://github.com/scikit-image/scikit-image/pull/6740>`_).
- Add missing backtick in docstring of ``skimage.graph.merge_nodes``
  (`#6741 <https://github.com/scikit-image/scikit-image/pull/6741>`_).
- Fix typo in ``skimage.metrics.variation_of_information``
  (`#6768 <https://github.com/scikit-image/scikit-image/pull/6768>`_).

Other and development related updates
-------------------------------------

Governance & planning
^^^^^^^^^^^^^^^^^^^^^
- Add draft of SKIP 4 "Transitioning to scikit-image 2.0"
  (`#6339 <https://github.com/scikit-image/scikit-image/pull/6339>`_, `#6353 <https://github.com/scikit-image/scikit-image/pull/6353>`_).

Maintenance
^^^^^^^^^^^
- Prepare release notes for v0.20.0
  (`#6556 <https://github.com/scikit-image/scikit-image/pull/6556>`_, `#6766 <https://github.com/scikit-image/scikit-image/pull/6766>`_).
- Add and test alternative build system based on Meson as an alternative to the deprecated distutils system
  (`#6536 <https://github.com/scikit-image/scikit-image/pull/6536>`_).
- Use ``cnp.float32_t`` and ``cnp.float64_t`` over ``float`` and ``double`` in Cython code
  (`#6303 <https://github.com/scikit-image/scikit-image/pull/6303>`_).
- Move ``skimage/measure/mc_meta`` folder into ``tools/precompute/`` folder to avoid its unnecessary distribution to users
  (`#6294 <https://github.com/scikit-image/scikit-image/pull/6294>`_).
- Remove unused function ``getLutNames`` in ``tools/precompute/mc_meta/createluts.py``
  (`#6294 <https://github.com/scikit-image/scikit-image/pull/6294>`_).
- Point urls for data files to a specific commit
  (`#6297 <https://github.com/scikit-image/scikit-image/pull/6297>`_).
- Drop Codecov badge from project README
  (`#6302 <https://github.com/scikit-image/scikit-image/pull/6302>`_).
- Remove undefined reference to ``'python_to_notebook'`` in ``doc/ext/notebook_doc.py``
  (`#6307 <https://github.com/scikit-image/scikit-image/pull/6307>`_).
- Parameterize tests in ``skimage.measure.tests.test_moments``
  (`#6323 <https://github.com/scikit-image/scikit-image/pull/6323>`_).
- Avoid unnecessary copying in ``skimage.morphology.skeletonize`` and update code style and tests
  (`#6327 <https://github.com/scikit-image/scikit-image/pull/6327>`_).
- Fix typo in ``_probabilistic_hough_line``
  (`#6373 <https://github.com/scikit-image/scikit-image/pull/6373>`_).
- Derive OBJECT_COLUMNS from COL_DTYPES in ``skimage.measure._regionprops``
  (`#6389 <https://github.com/scikit-image/scikit-image/pull/6389>`_).
- Support ``loadtxt`` of NumPy 1.23 with ``skimage/feature/orb_descriptor_positions.txt``
  (`#6400 <https://github.com/scikit-image/scikit-image/pull/6400>`_).
- Exclude pillow 9.1.1 from supported requirements
  (`#6384 <https://github.com/scikit-image/scikit-image/pull/6384>`_).
- Use the same numpy version dependencies for building as used by default
  (`#6409 <https://github.com/scikit-image/scikit-image/pull/6409>`_).
- Forward-port v0.19.1 and v0.19.2 release notes
  (`#6253 <https://github.com/scikit-image/scikit-image/pull/6253>`_).
- Forward-port v0.19.3 release notes
  (`#6416 <https://github.com/scikit-image/scikit-image/pull/6416>`_).
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
- Remove duplicate import in ``skimage.feature._canny``
  (`#6457 <https://github.com/scikit-image/scikit-image/pull/6457>`_).
- Use ``with open(...) as f`` instead of ``f = open(...)``
  (`#6458 <https://github.com/scikit-image/scikit-image/pull/6458>`_).
- Use context manager when possible
  (`#6484 <https://github.com/scikit-image/scikit-image/pull/6484>`_).
- Use ``broadcast_to`` instead of ``as_strided`` to generate broadcasted arrays
  (`#6476 <https://github.com/scikit-image/scikit-image/pull/6476>`_).
- Use ``moving_image`` in docstring of ``skimage.registration._optical_flow._tvl1``
  (`#6480 <https://github.com/scikit-image/scikit-image/pull/6480>`_).
- Use ``pyplot.get_cmap`` instead of deprecated ``cm.get_cmap`` in ``skimage.future.graph.show_rag`` for compatibility with matplotlib 3.3 to 3.6
  (`#6483 <https://github.com/scikit-image/scikit-image/pull/6483>`_, `#6490 <https://github.com/scikit-image/scikit-image/pull/6490>`_).
- Update ``plot_euler_number.py`` for maplotlib 3.6 compatibility
  (`#6522 <https://github.com/scikit-image/scikit-image/pull/6522>`_).
- Make non-functional change to build.txt to fix cache issue on CircleCI
  (`#6528 <https://github.com/scikit-image/scikit-image/pull/6528>`_).
- Update deprecated field ``license_file`` to ``license_files`` in ``setup.cfg``
  (`#6529 <https://github.com/scikit-image/scikit-image/pull/6529>`_).
- Ignore codespell fixes with git blame
  (`#6539 <https://github.com/scikit-image/scikit-image/pull/6539>`_).
- Remove ``FUNDING.yml`` in preference of org version
  (`#6553 <https://github.com/scikit-image/scikit-image/pull/6553>`_).
- Handle pending changes to ``tifffile.imwrite`` defaults and avoid test warnings
  (`#6460 <https://github.com/scikit-image/scikit-image/pull/6460>`_).
- Handle deprecation by updating to ``networkx.to_scipy_sparse_array``
  (`#6564 <https://github.com/scikit-image/scikit-image/pull/6564>`_).
- Update minimum supported numpy, scipy, and networkx
  (`#6385 <https://github.com/scikit-image/scikit-image/pull/6385>`_).
- Apply linting results after enabling pre-commit in CI
  (`#6568 <https://github.com/scikit-image/scikit-image/pull/6568>`_).
- Refactor lazy loading to use stubs & lazy_loader package
  (`#6577 <https://github.com/scikit-image/scikit-image/pull/6577>`_).
- Update sphinx configuration
  (`#6579 <https://github.com/scikit-image/scikit-image/pull/6579>`_).
- Update ``pyproject.toml`` to support Python 3.11 and to fix 32-bit pinned packages on Windows
  (`#6519 <https://github.com/scikit-image/scikit-image/pull/6519>`_).
- Update primary email address in mailmap entry for grlee77
  (`#6639 <https://github.com/scikit-image/scikit-image/pull/6639>`_).
- Handle new warnings introduced in NumPy 1.24
  (`#6637 <https://github.com/scikit-image/scikit-image/pull/6637>`_).
- Remove unnecessary dependency on ninja in ``pyproject.toml``
  (`#6634 <https://github.com/scikit-image/scikit-image/pull/6634>`_).
- Pin to latest meson-python ``>=0.11.0``
  (`#6627 <https://github.com/scikit-image/scikit-image/pull/6627>`_).
- Increase warning stacklevel by 1 in ``skimage.color.lab2xyz``
  (`#6613 <https://github.com/scikit-image/scikit-image/pull/6613>`_).
- Update OpenBLAS to v0.3.17
  (`#6607 <https://github.com/scikit-image/scikit-image/pull/6607>`_, `#6610 <https://github.com/scikit-image/scikit-image/pull/6610>`_).
- Fix Meson build on windows in sync with SciPy
  (`#6609 <https://github.com/scikit-image/scikit-image/pull/6609>`_).
- Set ``check: true`` for ``run_command`` in ``skimage/meson.build``
  (`#6606 <https://github.com/scikit-image/scikit-image/pull/6606>`_).
- Add ``dev.py`` and setup commands
  (`#6600 <https://github.com/scikit-image/scikit-image/pull/6600>`_).
- Organize ``dev.py`` commands into sections
  (`#6629 <https://github.com/scikit-image/scikit-image/pull/6629>`_).
- Remove thumbnail_size in config since sphinx-gallery>=0.9.0
  (`#6647 <https://github.com/scikit-image/scikit-image/pull/6647>`_).
- Add new test cases for ``skimage.transform.resize``
  (`#6669 <https://github.com/scikit-image/scikit-image/pull/6669>`_).
- Use meson-python main branch
  (`#6671 <https://github.com/scikit-image/scikit-image/pull/6671>`_).
- Simplify QhullError import
  (`#6677 <https://github.com/scikit-image/scikit-image/pull/6677>`_).
- Remove old SciPy cruft
  (`#6678 <https://github.com/scikit-image/scikit-image/pull/6678>`_, `#6681 <https://github.com/scikit-image/scikit-image/pull/6681>`_).
- Remove old references to imread package
  (`#6680 <https://github.com/scikit-image/scikit-image/pull/6680>`_).
- Remove pillow cruft (and a few other cleanups)
  (`#6683 <https://github.com/scikit-image/scikit-image/pull/6683>`_).
- Remove leftover ``gtk_plugin.ini``
  (`#6686 <https://github.com/scikit-image/scikit-image/pull/6686>`_).
- Prepare v0.20.0rc0
  (`#6706 <https://github.com/scikit-image/scikit-image/pull/6706>`_).
- Remove pre-release suffix for for Python 3.11
  (`#6709 <https://github.com/scikit-image/scikit-image/pull/6709>`_).
- Loosen tests for SciPy 1.10
  (`#6715 <https://github.com/scikit-image/scikit-image/pull/6715>`_).
- Specify C flag only if supported by compiler
  (`#6716 <https://github.com/scikit-image/scikit-image/pull/6716>`_).
- Extract version info from ``skimage/__init__.py`` in ``skimage/meson.build``
  (`#6723 <https://github.com/scikit-image/scikit-image/pull/6723>`_).
- Fix Cython errors/warnings
  (`#6725 <https://github.com/scikit-image/scikit-image/pull/6725>`_).
- Generate pyproject deps from requirements
  (`#6726 <https://github.com/scikit-image/scikit-image/pull/6726>`_).
- MAINT: Use ``uintptr_t`` to calculate new heap ptr positions
  (`#6734 <https://github.com/scikit-image/scikit-image/pull/6734>`_).
- Bite the bullet: remove distutils and setup.py
  (`#6738 <https://github.com/scikit-image/scikit-image/pull/6738>`_).
- Use meson-python developer version
  (`#6753 <https://github.com/scikit-image/scikit-image/pull/6753>`_).
- Require ``setuptools`` 65.6+
  (`#6751 <https://github.com/scikit-image/scikit-image/pull/6751>`_).
- Remove ``setup.cfg``, use ``pyproject.toml`` instead
  (`#6758 <https://github.com/scikit-image/scikit-image/pull/6758>`_).
- Update ``pyproject.toml`` to use ``meson-python>=0.13.0rc0``
  (`#6759 <https://github.com/scikit-image/scikit-image/pull/6759>`_).

Benchmarks
^^^^^^^^^^
- Add benchmarks for ``morphology.local_maxima``
  (`#3255 <https://github.com/scikit-image/scikit-image/pull/3255>`_).
- Add benchmarks for ``skimage.morphology.reconstruction``
  (`#6342 <https://github.com/scikit-image/scikit-image/pull/6342>`_).
- Update benchmark environment to Python 3.10 and NumPy 1.23
  (`#6511 <https://github.com/scikit-image/scikit-image/pull/6511>`_).

CI & automation
^^^^^^^^^^^^^^^
- Add Github ``actions/stale`` to label "dormant" issues and PRs
  (`#6506 <https://github.com/scikit-image/scikit-image/pull/6506>`_, `#6546 <https://github.com/scikit-image/scikit-image/pull/6546>`_, `#6552 <https://github.com/scikit-image/scikit-image/pull/6552>`_).
- Fix the autogeneration of API docs for lazy loaded subpackages
  (`#6075 <https://github.com/scikit-image/scikit-image/pull/6075>`_).
- Checkout gh-pages with a shallow clone
  (`#6085 <https://github.com/scikit-image/scikit-image/pull/6085>`_).
- Fix dev doc build
  (`#6091 <https://github.com/scikit-image/scikit-image/pull/6091>`_).
- Fix CI by excluding Pillow 9.1.0
  (`#6315 <https://github.com/scikit-image/scikit-image/pull/6315>`_).
- Pin pip pip to <22.1 in ``tools/github/before_install.sh``
  (`#6379 <https://github.com/scikit-image/scikit-image/pull/6379>`_).
- Update GH actions from v2 to v3
  (`#6382 <https://github.com/scikit-image/scikit-image/pull/6382>`_).
- Update to supported CircleCI images
  (`#6401 <https://github.com/scikit-image/scikit-image/pull/6401>`_).
- Use artifact-redirector
  (`#6407 <https://github.com/scikit-image/scikit-image/pull/6407>`_).
- Forward-port gh-6369: Fix windows wheels: use vsdevcmd.bat to make sure rc.exe is on the path
  (`#6417 <https://github.com/scikit-image/scikit-image/pull/6417>`_).
- Restrict GitHub Actions permissions to required ones
  (`#6426 <https://github.com/scikit-image/scikit-image/pull/6426>`_, `#6504 <https://github.com/scikit-image/scikit-image/pull/6504>`_).
- Update to Ubuntu LTS version on Actions workflows
  (`#6478 <https://github.com/scikit-image/scikit-image/pull/6478>`_).
- Relax label name comparison in ``benchmarks.yaml`` workflow
  (`#6520 <https://github.com/scikit-image/scikit-image/pull/6520>`_).
- Add linting via pre-commit
  (`#6563 <https://github.com/scikit-image/scikit-image/pull/6563>`_).
- Add CI tests for Python 3.11
  (`#6566 <https://github.com/scikit-image/scikit-image/pull/6566>`_).
- Fix CI for Scipy 1.9.2
  (`#6567 <https://github.com/scikit-image/scikit-image/pull/6567>`_).
- Test optional Py 3.10  dependencies on MacOS
  (`#6580 <https://github.com/scikit-image/scikit-image/pull/6580>`_).
- Pin setuptools in GHA MacOS workflow and ``azure-pipelines.yml``
  (`#6626 <https://github.com/scikit-image/scikit-image/pull/6626>`_).
- Build Python 3.11 wheels
  (`#6581 <https://github.com/scikit-image/scikit-image/pull/6581>`_).
- Fix doc build on CircleCI and add ccache
  (`#6646 <https://github.com/scikit-image/scikit-image/pull/6646>`_).
- Build wheels on CI via branch rather than tag
  (`#6668 <https://github.com/scikit-image/scikit-image/pull/6668>`_).
- Do not build wheels on pushes to main
  (`#6673 <https://github.com/scikit-image/scikit-image/pull/6673>`_).
- Use ``tools/github/before_install.sh`` wheels workflow
  (`#6718 <https://github.com/scikit-image/scikit-image/pull/6718>`_).
- Use Ruff for linting
  (`#6729 <https://github.com/scikit-image/scikit-image/pull/6729>`_).
- Use test that can fail for sdist
  (`#6731 <https://github.com/scikit-image/scikit-image/pull/6731>`_).
- Fix fstring in ``skimage._shared._warnings.expected_warnings``
  (`#6733 <https://github.com/scikit-image/scikit-image/pull/6733>`_).
- Build macosx/py38 wheel natively
  (`#6743 <https://github.com/scikit-image/scikit-image/pull/6743>`_).
- Remove CircleCI URL check
  (`#6749 <https://github.com/scikit-image/scikit-image/pull/6749>`_).
- CI Set MACOSX_DEPLOYMENT_TARGET=10.9 for Wheels
  (`#6750 <https://github.com/scikit-image/scikit-image/pull/6750>`_).
- Add temporary workaround until new meson-python release
  (`#6757 <https://github.com/scikit-image/scikit-image/pull/6757>`_).
- Update action to use new environment file
  (`#6762 <https://github.com/scikit-image/scikit-image/pull/6762>`_).
- Autogenerate pyproject.toml
  (`#6763 <https://github.com/scikit-image/scikit-image/pull/6763>`_).

71 authors contributed to this release [alphabetical by first name or login]
----------------------------------------------------------------------------
- Adeel Hassan
- Albert Y. Shih
- AleixBP (AleixBP)
- Alex (sashashura)
- Alexandr Kalinin
- Alexandre de Siqueira
- Amin (MOAMSA)
- Antony Lee
- Balint Varga
- Ben Greiner
- bsmietanka (bsmietanka)
- Chris Roat
- Chris Wood
- Daria
- Dave Mellert
- Dudu Lasry
- Elena Pascal
- Eli Schwartz
- Fabian Schneider
- forgeRW (forgeRW)
- Frank A. Krueger
- Gregory Lee
- Gus Becker
- Hande Gözükan
- Jacob Rosenthal
- James Gao
- Jan Kadlec
- Jan-Hendrik Müller
- Jan-Lukas Wynen
- Jarrod Millman
- Jeremy Muhlich
- johnthagen (johnthagen)
- Joshua Newton
- Juan DF
- Juan Nunez-Iglesias
- Judd Storrs
- Larry Bradley
- Lars Grüter
- lihaitao (li1127217ye)
- Lucas Johnson
- Malinda (maldil)
- Marianne Corvellec
- Mark Harfouche
- Martijn Courteaux
- Marvin Albert
- Matthew Brett
- Matthias Bussonnier
- Miles Lucas
- Nathan Chan
- Naveen
- OBgoneSouth (OBgoneSouth)
- Oren Amsalem
- Preston Buscay
- Peter Sobolewski
- Peter Bell
- Ray Bell
- Riadh Fezzani
- Robin Thibaut
- Ross Barnowski
- samtygier (samtygier)
- Sandeep N Menon
- Sanghyeok Hyun
- Sebastian Berg
- Sebastian Wallkötter
- Simon-Martin Schröder
- Stefan van der Walt
- Teemu Kumpumäki
- Thanushi Peiris
- Thomas Voigtmann
- Tim-Oliver Buchholz
- Tyler Reddy

42 reviewers contributed to this release [alphabetical by first name or login]
------------------------------------------------------------------------------
- Abhijeet Parida
- Albert Y. Shih
- Alex (sashashura)
- Alexandre de Siqueira
- Antony Lee
- Ben Greiner
- Carlo Dri
- Chris Roat
- Daniele Nicolodi
- Daria
- Dudu Lasry
- Eli Schwartz
- François Boulogne
- Gregory Lee
- Gus Becker
- Jacob Rosenthal
- James Gao
- Jan-Hendrik Müller
- Jarrod Millman
- Juan DF
- Juan Nunez-Iglesias
- Lars Grüter
- Malinda (maldil)
- Marianne Corvellec
- Mark Harfouche
- Martijn Courteaux
- Marvin Albert
- Matthias Bussonnier
- Oren Amsalem
- Ralf Gommers
- Riadh Fezzani
- Robert Haase
- Robin Thibaut
- Sandeep N Menon
- Sanghyeok Hyun
- Sebastian Berg
- Sebastian Wallkötter
- Simon-Martin Schröder
- Stefan van der Walt
- Thanushi Peiris
- Thomas Voigtmann
- Tim-Oliver Buchholz

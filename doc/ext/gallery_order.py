from os import path
from collections import OrderedDict
from sphinx_gallery.sorting import ExplicitOrder


gallery = OrderedDict({
    '../examples/data': [
        'plot_general.py',
        'plot_scientific.py',
        'plot_specific.py',
    ],
    '../examples/numpy_operations': [
        'plot_camera_numpy.py',
        'plot_view_as_blocks.py',
    ],
    '../examples/color_exposure': [
        'plot_rgb_to_gray.py',
        'plot_tinting_grayscale_images.py',
        'plot_adapt_rgb.py',
        'plot_equalize.py',
        'plot_local_equalize.py',
        'plot_log_gamma.py',
        'plot_regional_maxima.py',
        'plot_ihc_color_separation.py',
    ],
    '../examples/edges': [
        'plot_shapes.py',
        'plot_random_shapes.py',
        'plot_contours.py',
        'plot_polygon.py',
        'plot_convex_hull.py',
        'plot_line_hough_transform.py',
        'plot_circular_elliptical_hough_transform.py',
        'plot_edge_filter.py',
        'plot_canny.py',
        'plot_skeleton.py',
        'plot_active_contours.py',
        'plot_marching_cubes.py',
    ],
    '../examples/transform': [
        'plot_rescale.py',
        'plot_seam_carving.py',
        'plot_edge_modes.py',
        'plot_masked_register_translation.py',
        'plot_swirl.py',
        'plot_pyramid.py',
        'plot_histogram_matching.py',
        'plot_ssim.py',
        'plot_fundamental_matrix.py',
        'plot_ransac.py',
        'plot_matching.py',
        'plot_piecewise_affine.py',
        'plot_radon_transform.py',
        'plot_register_translation.py',
    ],
    '../examples/filters': [
        'plot_rank_mean.py',
        'plot_frangi.py',
        'plot_hysteresis.py',
        'plot_inpaint.py',
        'plot_unsharp_mask.py',
        'plot_deconvolution.py',
        'plot_restoration.py',
        'plot_denoise.py',
        'plot_denoise_wavelet.py',
        'plot_cycle_spinning.py',
        'plot_nonlocal_means.py',
        'plot_entropy.py',
        'plot_phase_unwrap.py',
    ],
    '../examples/features_detection': [
        'plot_brief.py',
        'plot_censure.py',
        'plot_corner.py',
        'plot_daisy.py',
        'plot_orb.py',
        'plot_hog.py',
        'plot_haar.py',
        'plot_blob.py',
        'plot_local_binary_pattern.py',
        'plot_gabor.py',
        'plot_gabors_from_astronaut.py',
        'plot_glcm.py',
        'plot_multiblock_local_binary_pattern.py',
        'plot_holes_and_peaks.py',
        'plot_template.py',
        'plot_windowed_histogram.py',
        'plot_shape_index.py',
    ],
    '../examples/segmentation': [
        'plot_thresholding.py',
        'plot_niblack_sauvola.py',
        'plot_regionprops.py',
        'plot_watershed.py',
        'plot_compact_watershed.py',
        'plot_marked_watershed.py',
        'plot_chan_vese.py',
        'plot_morphsnakes.py',
        'plot_label.py',
        'plot_peak_local_max.py',
        'plot_extrema.py',
        'plot_join_segmentations.py',
        'plot_rag.py',
        'plot_rag_draw.py',
        'plot_rag_mean_color.py',
        'plot_rag_merge.py',
        'plot_rag_boundary.py',
        'plot_boundary_merge.py',
        'plot_ncut.py',
        'plot_random_walker_segmentation.py',
        'plot_segmentations.py',
    ],
    '../examples/xx_applications': [
        'plot_thresholding.py',
        'plot_geometric.py',
        'plot_morphology.py',
        'plot_rank_filters.py',
        'plot_coins_segmentation.py',
        'plot_face_detection.py',
        'plot_haar_extraction_selection_classification.py',
    ]
})

conf_dir = path.dirname(__file__)
examples = [path.normpath(path.join(conf_dir, dir_path, e))
            for dir_path, examples in gallery.items()
            for e in examples]


class SubsectionOrder(ExplicitOrder):
    def __init__(self):
        super().__init__(list(gallery.keys()))


class ExamplesOrder:
    def __init__(self, src_dir):
        self.src_dir = src_dir

    def __call__(self, filename):
        filename = path.normpath(path.join(conf_dir, self.src_dir, filename))
        if filename in examples:
            return examples.index(filename)
        else:
            raise ValueError('{} is not specified in the gallery order'
                             .format(filename))

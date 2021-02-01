"""
=================================================================
Explore execution times and bottlenecks of scikit-image functions
=================================================================


In this example we compare the execution times of scikit-image functions.
For this, we choose a 200x200 image and we apply to this image all functions
which have an API compatible with ``out = function(img)``.

There are several orders of magnitude between the fastest and the slowest
functions. Various factors account for these variations. In particular,
some algorithms are iterative, meaning that the same procedure (requiring
to make some operations for all image pixels) need to be repeated at each
iteration of the algorithm.

In order to visualize if there is a bottleneck in the algorithm, we visualize
the source code of the function, and the total time spent to execute the
function for the test image. For this, we use the line profiler package,
and we display the source code in the hover information of the data points.
"""

import inspect
from timeit import default_timer
import numpy as np
from skimage import (
    exposure,
    feature,
    filters,
    measure,
    metrics,
    morphology,
    registration,
    restoration,
    segmentation,
    transform,
    data,
    color,
    draw
)
import pandas as pd
import plotly
import plotly.express as px
from line_profiler import LineProfiler, show_func
import io
import re

profile = LineProfiler()

l1 = 200

img = np.random.randint(256, size=(l1, l1), dtype=np.uint8)
img_rgb = np.random.randint(256, size=(l1, l1, 3), dtype=np.uint8)
img_binary = data.binary_blobs(length=l1, volume_fraction=0.3).astype(np.uint8)
img2 = data.binary_blobs(length=l1, volume_fraction=0.3, seed=10).astype(np.uint8)

# Parameters needed for some functions
parameters = {
    "match_histograms": dict(reference=img2),
    "cycle_spin": dict(func=restoration.denoise_wavelet, max_shifts=4),
    "gabor": dict(frequency=0.5),
    "denoise_tv_bregman": dict(weight=1.0),
    "apply_hysteresis_threshold": dict(low=0.4, high=0.6),
    "hough_circle": dict(radius=10),
    "rescale": dict(scale=1.1),
    "rotate": dict(angle=10),
    "block_reduce": dict(block_size=(2, 2)),
    "flood": dict(seed_point=(0, 0)),
    "flood_fill": dict(seed_point=(0, 0), new_value=2),
    "join_segmentations": dict(s2=img2),
    "inpaint_biharmonic": dict(mask=img2),
    "contingency_table": dict(im_test=img2),
    "hausdorff_distance": dict(image1=img2),
    "compare_images": dict(image2=img2),
    "mean_squared_error": dict(image1=img2),
    "normalized_root_mse": dict(image_test=img2),
    "peak_signal_noise_ratio": dict(image_test=img2),
    "structural_similarity": dict(im2=img2),
    "variation_of_information": dict(image1=img2),
    "optical_flow_tvl1": dict(moving_image=img2),
    "phase_cross_correlation": dict(moving_image=img2),
    "threshold_local": dict(block_size=l1 // 8 if (l1 // 8) % 2 == 1 else l1 // 8 + 1),
    "downscale_local_mean": dict(factors=(2,) * img.ndim),
    "difference_of_gaussians": dict(low_sigma=1),
    "find_contours": dict(level=0.5),
    "h_maxima": dict(h=10),
    "h_minima": dict(h=10),
}

need_binary_image = [
    "convex_hull_object",
    "convex_hull_image",
    "hausdorff_distance",
    "remove_small_holes",
    "remove_small_objects",
]

need_rgb_image = ["quickshift", "rgb2lab", "rgb2xyz", "xyz2lab"]

# Functions for which the API is not compatible
skip_functions = [
    "integrate",
    "hough_circle_peaks",
    "hough_line_peaks",
    "ransac",
    "window",
    "hough_ellipse",
    "view_as_blocks",
    "view_as_windows",
    "apply_parallel",
    "regular_grid",
    "regular_seeds",
    "estimate_transform",
    "matrix_transform",
    "draw_haar_like_feature",
    "corner_subpix",
    "calibrate_denoiser",
    "ball",
    "cube",
    "diamond",
    "disk",
    "octagon",
    "octahedron",
    "rectangle",
    "square",
    "star",
    "hessian_matrix_eigvals",
    "hessian_matrix_det",
    "structure_tensor_eigvals",
]

# This function is too slow for our demo
slow_functions = ["inpaint_biharmonic"]


def only_one_nondefault(args):
    """
    Returns True if the function has only one non-keyword parameter,
    False otherwise.
    """
    defaults = 0 if args.defaults is None else len(args.defaults)
    if len(args.args) >= 1 and (len(args.args) - defaults <= 1):
        return True
    else:
        return False


def _strip_docstring(func_str, max_len=60):
    """remove docstring from code block so that is does not overflow the plotly
    hover.
    """
    line_number = len([m.start() for m in re.finditer("\n", func_str)])
    if line_number < max_len:
        res = func_str
    else:
        try:
            open_docstring, end_docstring = [
                m.start() for m in re.finditer('"""', func_str)
            ]
            res = func_str[:open_docstring] + func_str[end_docstring + 3 :]
        except ValueError:
            res = func_str
    return res


def run_benchmark(
    img,
    img_binary,
    img_rgb,
    module_list=[
        exposure,
        feature,
        filters,
        measure,
        metrics,
        morphology,
        registration,
        restoration,
        segmentation,
        transform,
        color,
        draw
    ],
    skip_functions=[],
):
    times = {}

    functions = []
    for submodule in module_list:
        functions += inspect.getmembers(submodule, inspect.isfunction)
    non_tested_functions = []

    for function in functions:
        args = inspect.getfullargspec(function[1])
        only_one_argument = only_one_nondefault(args)
        if function[0] in skip_functions:
            continue
        if only_one_argument or function[0] in parameters:
            params = parameters[function[0]] if function[0] in parameters else {}
            try:
                if function[0] in need_binary_image:
                    im = img_binary
                elif function[0] in need_rgb_image:
                    im = img_rgb
                else:
                    im = img
                profile.add_function(function[1])
                start = default_timer()
                profile.runcall(function[1], im, **params)
                end = default_timer()
                times[function[0]] = end - start
            except:
                non_tested_functions.append(function[0])
        else:
            non_tested_functions.append(function[0])
    return times, non_tested_functions



module_list = [
    exposure,
    feature,
    filters,
    measure,
    metrics,
    morphology,
    registration,
    restoration,
    segmentation,
    transform,
    color,
    draw
]

times, non_tested_functions = run_benchmark(
    img, img_binary, img_rgb, skip_functions=skip_functions + slow_functions
)
function_names = sorted(times, key=times.get)
sorted_times = sorted(times.values())

# ----------------------- Print results -------------------------
df = []

for submodule in module_list:
    for func in inspect.getmembers(submodule, inspect.isfunction):
        if func[0] in times:
            df.append(
                {
                    "module": submodule.__name__,
                    "function": func[0],
                    "time": times[func[0]],
                }
            )

# ------------------- Retrieve results from line profiler --------
line_stats = profile.get_stats()
df = pd.DataFrame(df)
df = df.sort_values(by=["time"])
df["timings"] = ""
for (fn, lineno, name), timings in line_stats.timings.items():
    output = io.StringIO()
    show_func(
        fn,
        lineno,
        name,
        line_stats.timings[fn, lineno, name],
        line_stats.unit,
        stream=output,
    )
    dump = _strip_docstring(output.getvalue(), max_len=20).replace("\n", "<br>")
    df.loc[df.function == name, "timings"] = dump

# ----------------  Display results -----------------------------
df = df.sort_values(by=['module', 'time'])
for mod_name in df.module.unique():
    df.loc[df.module == mod_name, 'perf_index'] = np.arange(len(df[df.module == mod_name]), dtype=np.uint8)
fig = px.scatter(
            df,
            x="perf_index",
            y="time",
            color="module",
            facet_col='module',
            facet_col_wrap=3,
            log_y=True,
            hover_data=["function", "timings"],
            template="presentation",
            labels={'perf_index': '', 'module':''}
        )
fig.update_layout(
    title_text=f"Execution time for a {l1}x{l1} image",
    hoverlabel_align="left",
    hovermode="closest",
    showlegend=False,
)
fig.update_xaxes(matches=None)
plotly.io.show(fig)


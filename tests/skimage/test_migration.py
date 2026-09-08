"""Test migration module"""

import sys
import inspect
import subprocess
import warnings
import re
from textwrap import dedent
from pathlib import Path
from pprint import pformat


import numpy as np
import pytest

from _skimage2._shared._dependency_checks import is_wasm

from skimage._migration import (
    Skimage2Migration,
    ski2_migration_decorator,
    _select_blocks,
    _public_skimage_api_names,
)


def test_select_blocks_simple():
    doc = dedent("""\
    Always.
    <!--- cond-start: a -->
    a content
    <!--- cond-end -->
    <!--- cond-start: b -->
    b content
    <!--- cond-end -->
    <!--- cond-start: a -->
    a content 2
    <!--- cond-end -->
    """)

    result = _select_blocks(doc, context_name="a")
    assert result == dedent("""\
    Always.
    a content
    a content 2
    """)

    result = _select_blocks(doc, context_name="b")
    assert result == dedent("""\
    Always.
    b content
    """)

    result = _select_blocks(doc, context_name="x")
    assert result == "Always.\n"


EXAMPLE_INPUT = """\
Replace all calls to ``%(qname_old)s`` with ``%(qname_new)s``.

.. code-block:: python
    :linenos:
    :emphasize-lines: 1

    a = 1
    a

<!--- cond-start: warning -->
Only in warning

.. code-block:: python
    print('foo')

<!--- cond-end -->
  .. code-block:: python

    b = 2
    b

<!--- cond-start: doc -->
Only in doc

Examples
--------

>>> import skimage as ski1
>>> import _skimage2 as ski2
>>> res1 = ski1.somemod.somefunc(10, 11)
>>> res2 = ski2.somemod.somefunc(10, 11)
>>> assert res1 == res2

Some background on the changes.
<!--- cond-end -->
"""

EXAMPLE_WARN = """\
Replace all calls to `%(qname_old)s` with `%(qname_new)s`.

    a = 1
    a

Only in warning

    print('foo')

    b = 2
    b

See %(migration_url)s#%(qname_old_anchor)s
""".strip()

EXAMPLE_DOC = """\
Replace all calls to ``%(qname_old)s`` with ``%(qname_new)s``.

.. code-block:: python
    :linenos:
    :emphasize-lines: 1

    a = 1
    a

  .. code-block:: python

    b = 2
    b

Only in doc

Examples
--------

>>> import skimage as ski1
>>> import _skimage2 as ski2
>>> res1 = ski1.somemod.somefunc(10, 11)
>>> res2 = ski2.somemod.somefunc(10, 11)
>>> assert res1 == res2

Some background on the changes.
""".strip()


MIGRATION_URL = 'https://some.site/doc/migration.html'


def func(a, b):
    return a * b


class KlassDocOnly:
    def __init__(self):
        pass


class KlassWithWarning:
    def __init__(self):
        pass


def test_skimage2migration_parsing():
    migration_dec = Skimage2Migration(MIGRATION_URL)
    warn_msg, doc = migration_dec._parse_migration_doc(EXAMPLE_INPUT)
    assert warn_msg == EXAMPLE_WARN
    assert doc == EXAMPLE_DOC


_func_qname_old = f'{func.__module__}.{func.__qualname__}'
_anchor = _func_qname_old.replace('.', '-').replace('_', '-')
warn_msg, doc = Skimage2Migration(MIGRATION_URL)._filled_docs(
    EXAMPLE_INPUT,
    dict(
        qname_old=_func_qname_old,
        qname_new=_func_qname_old,
        migration_url=MIGRATION_URL,
        qname_old_anchor=_anchor,
    ),
)


def test_skimage2migration_decoration_interpolation():
    migration_dec = Skimage2Migration(MIGRATION_URL)
    dfunc = migration_dec(EXAMPLE_INPUT, qname_old="tests.skimage.test_migration.func")(
        func
    )

    docs = migration_dec.migration_docs
    assert docs == {_func_qname_old: doc}
    assert dfunc is not func

    from skimage.util import PendingSkimage2Change

    with pytest.warns(PendingSkimage2Change) as record:
        assert dfunc(2, 4) == 8

    assert len(record) == 1
    assert record[0].message.args[0] == warn_msg

    # Specify canonical location.
    migration_dec(EXAMPLE_INPUT, qname_old='skimage.bar.func')(func)
    assert docs['skimage.bar.func'].startswith(
        'Replace all calls to ``skimage.bar.func`` with ``skimage2.bar.func``.'
    )
    # And skimage2 location.
    migration_dec(
        EXAMPLE_INPUT, qname_old='skimage.bar.func', qname_new='skimage2.bun.biz'
    )(func)
    assert docs['skimage.bar.func'].startswith(
        'Replace all calls to ``skimage.bar.func`` with ``skimage2.bun.biz``.'
    )


def test_skimage2migration_dedent():
    # Test text dedented.
    migration_dec = Skimage2Migration(MIGRATION_URL)
    dfunc = migration_dec(EXAMPLE_INPUT, qname_old="tests.skimage.test_migration.func")(
        func
    )

    from skimage.util import PendingSkimage2Change

    # Warning and doc nevertheless stays the same.
    assert migration_dec.migration_docs == {_func_qname_old: doc}
    assert dfunc is not func
    with pytest.warns(PendingSkimage2Change) as record:
        assert dfunc(2, 4) == 8

    assert len(record) == 1
    assert record[0].message.args[0] == warn_msg


def test_peak_local_max():
    from skimage.feature import peak_local_max
    from skimage.util import PendingSkimage2Change

    assert peak_local_max is not inspect.unwrap(peak_local_max)
    assert 'skimage.feature.peak_local_max' in ski2_migration_decorator.migration_docs

    img = np.zeros((10, 10))

    with pytest.warns(
        PendingSkimage2Change,
        match=r'`skimage.feature.peak_local_max` is deprecated in favor of',
    ):
        peak_local_max(img)


def test_skimage2migration_no_warning_is_identity_decorator():
    migration_dec = Skimage2Migration(MIGRATION_URL)
    doc_only = dedent("""\
    <!--- cond-start: doc -->
    Doc only.
    <!--- cond-end -->
    """)
    dfunc = migration_dec(doc_only, qname_old="tests.skimage.test_migration.func")(func)

    assert dfunc is func
    assert getattr(func, '__wrapped__', func) is func
    assert migration_dec.migration_docs == {_func_qname_old: 'Doc only.'}

    # No warning when no warning message in migration docstring.
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter('always')
        assert dfunc(2, 4) == 8
    assert len(record) == 0


def test_skimage2_becomes_skimage():
    # At least for now, skimage.filters.frangi is defined in _skimage2, but it
    # should be detected as being in skimage, due to public skimage API.
    from skimage.filters import frangi

    assert "_skimage2" in frangi.__module__
    assert _public_skimage_api_names(frangi) == [
        'skimage.filters.frangi',
        'skimage.filters.ridges.frangi',
    ]


@pytest.mark.parametrize(
    "module,name",
    [
        ("skimage.morphology", "max_tree"),
        ("skimage.util", "lookfor"),
        ("skimage.util", "apply_parallel"),
    ],
)
@pytest.mark.skipif(is_wasm, reason="emscripten does not support processes")
def test_skimage_module_func_name_clashes(module, name):
    # Depending on how `max_tree` or `lookfor` are first imported, lazy_loader
    # may return the module or the function of the same name. Test that the
    # function is always returned.
    cmd = (
        f"from {module}.{name} import {name}; "
        f"import skimage; "
        f"print(callable({module}.{name}))"
    )
    result = subprocess.run(
        [sys.executable, "-c", cmd],
        capture_output=True,
        text=True,
    )
    assert result.stdout == "True\n"


def test_skimage2migration_comment_check():
    migration_dec = Skimage2Migration(MIGRATION_URL)

    doc = EXAMPLE_INPUT + '\n\nA <!-- marker'
    with pytest.raises(
        ValueError, match=r"Remaining <!-- marker in warning of `foo\.bar`;"
    ):
        migration_dec._parse_migration_doc(doc, 'foo.bar')


@pytest.mark.thread_unsafe("Warning filter state is not thread safe")
def test_skimage2migration_classes():
    migration_dec = Skimage2Migration(MIGRATION_URL)
    doc_only_qname = 'tests.skimage.test_migration.KlassDocOnly'
    warn_qname = 'tests.skimage.test_migration.KlassWithWarning'

    dklass = migration_dec(
        dedent(
            """<!--- cond-start: doc -->
        Doc only.
        <!--- cond-end -->
        """
        ),
        qname_old=doc_only_qname,
    )(KlassDocOnly)

    assert dklass is KlassDocOnly

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter('always')
        dklass()
    assert len(record) == 0

    dklass = migration_dec("Basic warning", qname_old=warn_qname)(KlassWithWarning)

    assert dklass is KlassWithWarning
    assert inspect.isclass(dklass)

    from skimage.util import PendingSkimage2Change

    with pytest.warns(PendingSkimage2Change, match='Basic warning') as record:
        dklass()
    assert len(record) == 1

    with pytest.warns(PendingSkimage2Change, match='Basic warning'):
        dklass()


def test_skimage2migration_trainable_segmenter_remains_class():
    from skimage.future.trainable_segmentation import TrainableSegmenter
    from skimage.util import PendingSkimage2Change

    class _Clf:
        """Dummy classifier for TrainableSegmenter"""

        def fit(self, *args, **kwargs):
            pass

        def predict(self, *args, **kwargs):
            return []

    assert inspect.isclass(TrainableSegmenter)

    with pytest.warns(PendingSkimage2Change):
        segmenter = TrainableSegmenter(clf=_Clf())

    assert hasattr(segmenter, 'fit')
    assert hasattr(segmenter, 'predict')


# The file tree in the installed v0.26.0 `skimage` package.
# Generated on 'v0.26.0' (ee0a7a3ebd9ac8c2602f40e55bc015a3c8a81ae8) by running
# `test_skimage_file_tree` (see `skimage_file_tree.txt` in its `tmp_dir`).
SKIMAGE_FILE_TREE = {
    '__init__',
    '_build_utils/__init__',
    '_build_utils/copyfiles',
    '_build_utils/cythoner',
    '_build_utils/gcc_build_bitness',
    '_build_utils/tempita',
    '_build_utils/version',
    '_doctest_adapters',
    '_shared/__init__',
    '_shared/_dependency_checks',
    '_shared/_geometry',
    '_shared/_tempfile',
    '_shared/_warnings',
    '_shared/compat',
    '_shared/coord',
    '_shared/dtype',
    '_shared/fast_exp',
    '_shared/filters',
    '_shared/geometry',
    '_shared/interpolation',
    '_shared/tester',
    '_shared/testing',
    '_shared/transform',
    '_shared/utils',
    '_shared/version_requirements',
    '_vendored/__init__',
    '_vendored/numpy_lookfor',
    'color/__init__',
    'color/adapt_rgb',
    'color/colorconv',
    'color/colorlabel',
    'color/delta_e',
    'color/rgb_colors',
    'conftest',
    'data/__init__',
    'data/_binary_blobs',
    'data/_fetchers',
    'data/_registry',
    'draw/__init__',
    'draw/_draw',
    'draw/_polygon2mask',
    'draw/_random_shapes',
    'draw/draw',
    'draw/draw3d',
    'draw/draw_nd',
    'exposure/__init__',
    'exposure/_adapthist',
    'exposure/exposure',
    'exposure/histogram_matching',
    'feature/__init__',
    'feature/_basic_features',
    'feature/_canny',
    'feature/_canny_cy',
    'feature/_cascade',
    'feature/_daisy',
    'feature/_fisher_vector',
    'feature/_haar',
    'feature/_hessian_det_appx_pythran',
    'feature/_hog',
    'feature/_hoghistogram',
    'feature/_orb_descriptor_positions',
    'feature/_sift',
    'feature/_texture',
    'feature/blob',
    'feature/brief',
    'feature/brief_pythran',
    'feature/censure',
    'feature/censure_cy',
    'feature/corner',
    'feature/corner_cy',
    'feature/haar',
    'feature/match',
    'feature/orb',
    'feature/orb_cy',
    'feature/peak',
    'feature/sift',
    'feature/template',
    'feature/texture',
    'feature/util',
    'filters/__init__',
    'filters/_fft_based',
    'filters/_gabor',
    'filters/_gaussian',
    'filters/_median',
    'filters/_multiotsu',
    'filters/_rank_order',
    'filters/_sparse',
    'filters/_unsharp_mask',
    'filters/_window',
    'filters/edges',
    'filters/lpi_filter',
    'filters/rank/__init__',
    'filters/rank/_percentile',
    'filters/rank/bilateral',
    'filters/rank/bilateral_cy',
    'filters/rank/core_cy',
    'filters/rank/core_cy_3d',
    'filters/rank/generic',
    'filters/rank/generic_cy',
    'filters/rank/percentile_cy',
    'filters/ridges',
    'filters/thresholding',
    'future/__init__',
    'future/manual_segmentation',
    'future/trainable_segmentation',
    'graph/__init__',
    'graph/_graph',
    'graph/_graph_cut',
    'graph/_graph_merge',
    'graph/_mcp',
    'graph/_ncut',
    'graph/_ncut_cy',
    'graph/_rag',
    'graph/_spath',
    'graph/heap',
    'graph/mcp',
    'graph/spath',
    'io/__init__',
    'io/_image_stack',
    'io/_io',
    'io/_plugins/__init__',
    'io/_plugins/fits_plugin',
    'io/_plugins/gdal_plugin',
    'io/_plugins/imageio_plugin',
    'io/_plugins/imread_plugin',
    'io/_plugins/matplotlib_plugin',
    'io/_plugins/pil_plugin',
    'io/_plugins/simpleitk_plugin',
    'io/_plugins/tifffile_plugin',
    'io/collection',
    'io/manage_plugins',
    'io/sift',
    'io/util',
    'measure/__init__',
    'measure/_blur_effect',
    'measure/_ccomp',
    'measure/_colocalization',
    'measure/_find_contours',
    'measure/_find_contours_cy',
    'measure/_label',
    'measure/_marching_cubes_lewiner',
    'measure/_marching_cubes_lewiner_cy',
    'measure/_marching_cubes_lewiner_luts',
    'measure/_moments',
    'measure/_moments_analytical',
    'measure/_moments_cy',
    'measure/_pnpoly',
    'measure/_polygon',
    'measure/_regionprops',
    'measure/_regionprops_utils',
    'measure/block',
    'measure/entropy',
    'measure/fit',
    'measure/pnpoly',
    'measure/profile',
    'metrics/__init__',
    'metrics/_adapted_rand_error',
    'metrics/_contingency_table',
    'metrics/_structural_similarity',
    'metrics/_variation_of_information',
    'metrics/set_metrics',
    'metrics/simple_metrics',
    'morphology/__init__',
    'morphology/_convex_hull',
    'morphology/_extrema_cy',
    'morphology/_flood_fill',
    'morphology/_flood_fill_cy',
    'morphology/_grayreconstruct',
    'morphology/_max_tree',
    'morphology/_misc_cy',
    'morphology/_skeletonize',
    'morphology/_skeletonize_various_cy',
    'morphology/_util',
    'morphology/binary',
    'morphology/convex_hull',
    'morphology/extrema',
    'morphology/footprints',
    'morphology/gray',
    'morphology/grayreconstruct',
    'morphology/isotropic',
    'morphology/max_tree',
    'morphology/misc',
    'registration/__init__',
    'registration/_masked_phase_cross_correlation',
    'registration/_optical_flow',
    'registration/_optical_flow_utils',
    'registration/_phase_cross_correlation',
    'restoration/__init__',
    'restoration/_cycle_spin',
    'restoration/_denoise',
    'restoration/_denoise_cy',
    'restoration/_inpaint',
    'restoration/_nl_means_denoising',
    'restoration/_rolling_ball',
    'restoration/_rolling_ball_cy',
    'restoration/_unwrap_1d',
    'restoration/_unwrap_2d',
    'restoration/_unwrap_3d',
    'restoration/deconvolution',
    'restoration/inpaint',
    'restoration/j_invariant',
    'restoration/non_local_means',
    'restoration/uft',
    'restoration/unwrap',
    'segmentation/__init__',
    'segmentation/_chan_vese',
    'segmentation/_clear_border',
    'segmentation/_expand_labels',
    'segmentation/_felzenszwalb',
    'segmentation/_felzenszwalb_cy',
    'segmentation/_join',
    'segmentation/_quickshift',
    'segmentation/_quickshift_cy',
    'segmentation/_slic',
    'segmentation/_watershed',
    'segmentation/_watershed_cy',
    'segmentation/active_contour_model',
    'segmentation/boundaries',
    'segmentation/morphsnakes',
    'segmentation/random_walker_segmentation',
    'segmentation/slic_superpixels',
    'transform/__init__',
    'transform/_geometric',
    'transform/_hough_transform',
    'transform/_radon_transform',
    'transform/_thin_plate_splines',
    'transform/_warps',
    'transform/_warps_cy',
    'transform/finite_radon_transform',
    'transform/hough_transform',
    'transform/integral',
    'transform/pyramids',
    'transform/radon_transform',
    'util/__init__',
    'util/_backends',
    'util/_invert',
    'util/_label',
    'util/_map_array',
    'util/_montage',
    'util/_regular_grid',
    'util/_remap',
    'util/_slice_along_axes',
    'util/apply_parallel',
    'util/arraycrop',
    'util/compare',
    'util/dtype',
    'util/lookfor',
    'util/noise',
    'util/shape',
    'util/unique',
}


# Define reviewed exceptions. These are ignored in `test_skimage_file_tree`
# when comparing `SKIMAGE_FILE_TREE` with the current file tree.
SKIMAGE_FILE_TREE_IGNORE = {
    # _build_utils were never installed were only available at build-time
    '_build_utils/__init__',
    '_build_utils/copyfiles',
    '_build_utils/cythoner',
    '_build_utils/gcc_build_bitness',
    '_build_utils/tempita',
    '_build_utils/version',
    # Added after v0.26.0:
    '_migration',
    # Nowhere used on GitHub
    '_vendored/__init__',  # also quite young
    '_vendored/numpy_lookfor',
}


# Compiled to modules with a different name
SKIMAGE_COMPILED_FILES = {
    'feature/_hessian_det_appx_pythran': 'feature/_hessian_det_appx',
    'feature/brief_pythran': 'feature/brief_cy',
}


def _walk_python_files(package_path, *, root=None):
    """Walk Python modules in a given package.

    Parameters
    ----------
    package_path : Path
        Directory of a Python package.
    root : Path, optional
        The root of the Python package. Defaults to `package_path`.

    Yields
    ------
    key : str
        A key describing the Python file / submodule relative to the given
        `root` directory. Suffixes are not included. For example,
        "io/collection" or "graph/heap".
    path : Path
        The full path to the file.
    """
    package_path = Path(package_path)
    assert package_path.is_dir()

    if root is None:
        root = package_path
    assert root.is_dir()

    for path in sorted(package_path.glob("*")):
        if path.is_dir():
            yield from _walk_python_files(path, root=root)
            continue
        assert path.is_file()
        if path.suffix in (".py", ".pyx"):
            key = path.relative_to(root).parts[:-1]
            key = key + (path.stem,)
            key = "/".join(key)
            yield key, path


def test_skimage_file_tree(tmp_path):
    import skimage

    skimage_path = Path(skimage.__file__).parent
    skimage_file_tree = {key for key, _ in _walk_python_files(skimage_path)}

    actual = skimage_file_tree - SKIMAGE_FILE_TREE_IGNORE
    expected = SKIMAGE_FILE_TREE - SKIMAGE_FILE_TREE_IGNORE

    for name, rename in SKIMAGE_COMPILED_FILES.items():
        if name in expected:
            expected.remove(name)
            expected.add(rename)

    with (tmp_path / "skimage_file_tree.txt").open("w") as f:
        f.writelines(pformat(skimage_file_tree))
    with (tmp_path / "actual.txt").open("w") as f:
        f.writelines(pformat(actual))
    with (tmp_path / "expected.txt").open("w") as f:
        f.writelines(pformat(expected))

    assert actual == expected


# Generated on 'main' (27041bd05f38facf02570a6bdb9aa4010f2d68b5)
SKIMAGE_API = {
    'skimage.color:ahx_from_rgb',
    'skimage.color:bex_from_rgb',
    'skimage.color:bpx_from_rgb',
    'skimage.color:bro_from_rgb',
    'skimage.color:color_dict',
    'skimage.color:combine_stains(stains, conv_matrix, *, channel_axis=-1)',
    'skimage.color:convert_colorspace(arr, fromspace, tospace, *, channel_axis=-1)',
    'skimage.color:deltaE_cie76(lab1, lab2, channel_axis=-1)',
    'skimage.color:deltaE_ciede2000(lab1, lab2, kL=1, kC=1, kH=1, *, channel_axis=-1)',
    'skimage.color:deltaE_ciede94(lab1, lab2, kH=1, kC=1, kL=1, k1=0.045, '
    'k2=0.015, *, channel_axis=-1)',
    'skimage.color:deltaE_cmc(lab1, lab2, kL=1, kC=1, *, channel_axis=-1)',
    'skimage.color:fgx_from_rgb',
    'skimage.color:gdx_from_rgb',
    'skimage.color:gray2rgb(image, *, channel_axis=-1)',
    'skimage.color:gray2rgba(image, alpha=None, *, channel_axis=-1)',
    'skimage.color:hax_from_rgb',
    'skimage.color:hdx_from_rgb',
    'skimage.color:hed2rgb(hed, *, channel_axis=-1)',
    'skimage.color:hed_from_rgb',
    'skimage.color:hpx_from_rgb',
    'skimage.color:hsv2rgb(hsv, *, channel_axis=-1)',
    'skimage.color:lab2lch(lab, *, channel_axis=-1)',
    "skimage.color:lab2rgb(lab, illuminant='D65', observer='2', *, channel_axis=-1)",
    "skimage.color:lab2xyz(lab, illuminant='D65', observer='2', *, channel_axis=-1)",
    'skimage.color:label2rgb(label, image=None, colors=None, alpha=0.3, '
    "bg_label=0, bg_color=(0, 0, 0), image_alpha=1, kind='overlay', *, "
    'saturation=0, channel_axis=-1)',
    'skimage.color:lch2lab(lch, *, channel_axis=-1)',
    'skimage.color:luv2rgb(luv, *, channel_axis=-1)',
    "skimage.color:luv2xyz(luv, illuminant='D65', observer='2', *, channel_axis=-1)",
    'skimage.color:rbd_from_rgb',
    'skimage.color:rgb2gray(rgb, *, channel_axis=-1)',
    'skimage.color:rgb2hed(rgb, *, channel_axis=-1)',
    'skimage.color:rgb2hsv(rgb, *, channel_axis=-1)',
    "skimage.color:rgb2lab(rgb, illuminant='D65', observer='2', *, channel_axis=-1)",
    'skimage.color:rgb2luv(rgb, *, channel_axis=-1)',
    'skimage.color:rgb2rgbcie(rgb, *, channel_axis=-1)',
    'skimage.color:rgb2xyz(rgb, *, channel_axis=-1)',
    'skimage.color:rgb2ycbcr(rgb, *, channel_axis=-1)',
    'skimage.color:rgb2ydbdr(rgb, *, channel_axis=-1)',
    'skimage.color:rgb2yiq(rgb, *, channel_axis=-1)',
    'skimage.color:rgb2ypbpr(rgb, *, channel_axis=-1)',
    'skimage.color:rgb2yuv(rgb, *, channel_axis=-1)',
    'skimage.color:rgb_from_ahx',
    'skimage.color:rgb_from_bex',
    'skimage.color:rgb_from_bpx',
    'skimage.color:rgb_from_bro',
    'skimage.color:rgb_from_fgx',
    'skimage.color:rgb_from_gdx',
    'skimage.color:rgb_from_hax',
    'skimage.color:rgb_from_hdx',
    'skimage.color:rgb_from_hed',
    'skimage.color:rgb_from_hpx',
    'skimage.color:rgb_from_rbd',
    'skimage.color:rgba2rgb(rgba, background=(1, 1, 1), *, channel_axis=-1)',
    'skimage.color:rgbcie2rgb(rgbcie, *, channel_axis=-1)',
    'skimage.color:separate_stains(rgb, conv_matrix, *, channel_axis=-1)',
    "skimage.color:xyz2lab(xyz, illuminant='D65', observer='2', *, channel_axis=-1)",
    "skimage.color:xyz2luv(xyz, illuminant='D65', observer='2', *, channel_axis=-1)",
    'skimage.color:xyz2rgb(xyz, *, channel_axis=-1)',
    'skimage.color:xyz_tristimulus_values(*, illuminant, observer, dtype=<class '
    "'float'>)",
    'skimage.color:ycbcr2rgb(ycbcr, *, channel_axis=-1)',
    'skimage.color:ydbdr2rgb(ydbdr, *, channel_axis=-1)',
    'skimage.color:yiq2rgb(yiq, *, channel_axis=-1)',
    'skimage.color:ypbpr2rgb(ypbpr, *, channel_axis=-1)',
    'skimage.color:yuv2rgb(yuv, *, channel_axis=-1)',
    'skimage.data:astronaut()',
    'skimage.data:binary_blobs(length=512, blob_size_fraction=0.1, n_dim=2, '
    "volume_fraction=0.5, rng=None, *, boundary_mode='nearest')",
    'skimage.data:brain()',
    'skimage.data:brick()',
    'skimage.data:camera()',
    'skimage.data:cat()',
    'skimage.data:cell()',
    'skimage.data:cells3d()',
    'skimage.data:checkerboard()',
    'skimage.data:chelsea()',
    'skimage.data:clock()',
    'skimage.data:coffee()',
    'skimage.data:coins()',
    'skimage.data:colorwheel()',
    'skimage.data:data_dir',
    'skimage.data:legacy_data_dir',
    'skimage.data:download_all(directory=None)',
    'skimage.data:eagle()',
    "skimage.data:file_hash(fname, alg='sha256')",
    'skimage.data:grass()',
    'skimage.data:gravel()',
    'skimage.data:horse()',
    'skimage.data:hubble_deep_field()',
    'skimage.data:human_mitosis()',
    'skimage.data:immunohistochemistry()',
    'skimage.data:kidney()',
    'skimage.data:lbp_frontal_face_cascade_filename()',
    'skimage.data:lfw_subset()',
    'skimage.data:lily()',
    'skimage.data:logo()',
    'skimage.data:microaneurysms()',
    'skimage.data:moon()',
    'skimage.data:nickel_solidification()',
    'skimage.data:page()',
    'skimage.data:palisades_of_vogt()',
    'skimage.data:protein_transport()',
    'skimage.data:retina()',
    'skimage.data:rocket()',
    'skimage.data:shepp_logan_phantom()',
    'skimage.data:skin()',
    'skimage.data:stereo_motorcycle()',
    'skimage.data:text()',
    'skimage.data:vortex()',
    'skimage.draw:_bezier_segment',
    'skimage.draw:bezier_curve(r0, c0, r1, c1, r2, c2, weight, shape=None)',
    "skimage.draw:circle_perimeter(r, c, radius, method='bresenham', shape=None)",
    'skimage.draw:circle_perimeter_aa(r, c, radius, shape=None)',
    'skimage.draw:disk(center, radius, *, shape=None)',
    'skimage.draw:ellipse(r, c, r_radius, c_radius, shape=None, rotation=0.0)',
    'skimage.draw:ellipse_perimeter(r, c, r_radius, c_radius, orientation=0, '
    'shape=None)',
    'skimage.draw:ellipsoid(a, b, c, spacing=(1.0, 1.0, 1.0), levelset=False)',
    'skimage.draw:ellipsoid_stats(a, b, c)',
    'skimage.draw:line(r0, c0, r1, c1)',
    'skimage.draw:line_aa(r0, c0, r1, c1)',
    'skimage.draw:line_nd(start, stop, *, endpoint=False, integer=True)',
    'skimage.draw:polygon(r, c, shape=None)',
    'skimage.draw:polygon2mask(image_shape, polygon)',
    'skimage.draw:polygon_perimeter(r, c, shape=None, clip=False)',
    'skimage.draw:random_shapes(image_shape, max_shapes, min_shapes=1, '
    'min_size=2, max_size=None, num_channels=3, shape=None, intensity_range=None, '
    'allow_overlap=False, num_trials=100, rng=None, *, channel_axis=-1)',
    'skimage.draw:rectangle(start, end=None, extent=None, shape=None)',
    'skimage.draw:rectangle_perimeter(start, end=None, extent=None, shape=None, '
    'clip=False)',
    'skimage.draw:set_color(image, coords, color, alpha=1)',
    'skimage.exposure:adjust_gamma(image, gamma=1, gain=1)',
    'skimage.exposure:adjust_log(image, gain=1, inv=False)',
    'skimage.exposure:adjust_sigmoid(image, cutoff=0.5, gain=10, inv=False)',
    'skimage.exposure:cumulative_distribution(image, nbins=256)',
    'skimage.exposure:equalize_adapthist(image, kernel_size=None, '
    'clip_limit=0.01, nbins=256)',
    'skimage.exposure:equalize_hist(image, nbins=256, mask=None)',
    "skimage.exposure:histogram(image, nbins=256, source_range='image', "
    'normalize=False, *, channel_axis=None)',
    'skimage.exposure:is_low_contrast(image, fraction_threshold=0.05, '
    "lower_percentile=1, upper_percentile=99, method='linear')",
    'skimage.exposure:match_histograms(image, reference, *, channel_axis=None)',
    "skimage.exposure:rescale_intensity(image, in_range='image', out_range='dtype')",
    'skimage.feature:BRIEF',
    'skimage.feature:CENSURE',
    'skimage.feature:Cascade',
    'skimage.feature:ORB',
    'skimage.feature:SIFT',
    'skimage.feature:blob_dog(image, min_sigma=1, max_sigma=50, sigma_ratio=1.6, '
    'threshold=0.5, overlap=0.5, *, threshold_rel=<DEPRECATED>, '
    "exclude_border=False, prescale='legacy')",
    'skimage.feature:blob_doh(image, min_sigma=1, max_sigma=30, num_sigma=10, '
    'threshold=0.01, overlap=0.5, log_scale=False, *, threshold_rel=<DEPRECATED>, '
    "prescale='legacy')",
    'skimage.feature:blob_log(image, min_sigma=1, max_sigma=50, num_sigma=10, '
    'threshold=0.2, overlap=0.5, log_scale=False, *, threshold_rel=<DEPRECATED>, '
    "exclude_border=False, prescale='legacy')",
    'skimage.feature:canny(image, sigma=1.0, low_threshold=None, '
    "high_threshold=None, mask=None, use_quantiles=False, *, mode='constant', "
    'cval=0.0)',
    'skimage.feature:corner_fast(image, n=12, threshold=0.15)',
    'skimage.feature:corner_foerstner(image, sigma=1)',
    "skimage.feature:corner_harris(image, method='k', k=0.05, eps=1e-06, sigma=1)",
    "skimage.feature:corner_kitchen_rosenfeld(image, mode='constant', cval=0)",
    'skimage.feature:corner_moravec(image, window_size=1)',
    'skimage.feature:corner_orientations(image, corners, mask)',
    'skimage.feature:corner_peaks(image, min_distance=1, threshold_abs=None, '
    'threshold_rel=None, exclude_border=True, indices=True, num_peaks=inf, '
    'footprint=None, labels=None, *, num_peaks_per_label=inf, p_norm=inf)',
    'skimage.feature:corner_shi_tomasi(image, sigma=1)',
    'skimage.feature:corner_subpix(image, corners, window_size=11, alpha=0.99)',
    'skimage.feature:daisy(image, step=4, radius=15, rings=3, histograms=8, '
    "orientations=8, normalization='l1', sigmas=None, ring_radii=None, "
    'visualize=False)',
    'skimage.feature:draw_haar_like_feature(image, r, c, width, height, '
    'feature_coord, color_positive_block=(1.0, 0.0, 0.0), '
    'color_negative_block=(0.0, 1.0, 0.0), alpha=0.5, max_n_features=None, '
    'rng=None)',
    'skimage.feature:draw_multiblock_lbp(image, r, c, width, height, lbp_code=0, '
    'color_greater_block=(1, 1, 1), color_less_block=(0, 0.69, 0.96), alpha=0.5)',
    'skimage.feature:fisher_vector(descriptors, gmm, *, improved=False, alpha=0.5)',
    'skimage.feature:graycomatrix(image, distances, angles, levels=None, '
    'symmetric=False, normed=False)',
    "skimage.feature:graycoprops(P, prop='contrast')",
    'skimage.feature:haar_like_feature(int_image, r, c, width, height, '
    'feature_type=None, feature_coord=None)',
    'skimage.feature:haar_like_feature_coord(width, height, feature_type=None)',
    "skimage.feature:hessian_matrix(image, sigma=1, mode='constant', cval=0, "
    "order='rc', use_gaussian_derivatives=None)",
    'skimage.feature:hessian_matrix_det(image, sigma=1, approximate=True)',
    'skimage.feature:hessian_matrix_eigvals(H_elems)',
    'skimage.feature:hog(image, orientations=9, pixels_per_cell=(8, 8), '
    "cells_per_block=(3, 3), block_norm='L2-Hys', visualize=False, "
    'transform_sqrt=False, feature_vector=True, *, channel_axis=None)',
    'skimage.feature:learn_gmm(descriptors, *, n_modes=32, gm_args=None)',
    "skimage.feature:local_binary_pattern(image, P, R, method='default')",
    'skimage.feature:match_descriptors(descriptors1, descriptors2, metric=None, '
    'p=2, max_distance=inf, cross_check=True, max_ratio=1.0)',
    'skimage.feature:match_template(image, template, pad_input=False, '
    "mode='constant', constant_values=0)",
    'skimage.feature:multiblock_lbp(int_image, r, c, width, height)',
    'skimage.feature:multiscale_basic_features(image, intensity=True, edges=True, '
    'texture=True, sigma_min=0.5, sigma_max=16, num_sigma=None, workers=None, *, '
    'channel_axis=None)',
    'skimage.feature:peak_local_max(image, min_distance=1, threshold_abs=None, '
    'threshold_rel=None, exclude_border=True, num_peaks=None, footprint=None, '
    'labels=None, num_peaks_per_label=None, p_norm=inf)',
    'skimage.feature:plot_matched_features(image0, image1, *, keypoints0, '
    "keypoints1, matches, ax, keypoints_color='k', matches_color=None, "
    "only_matches=False, alignment='horizontal')",
    "skimage.feature:shape_index(image, sigma=1, mode='constant', cval=0)",
    "skimage.feature:structure_tensor(image, sigma=1, mode='constant', cval=0, "
    "order='rc')",
    'skimage.feature:structure_tensor_eigenvalues(A_elems)',
    'skimage.filters.rank:autolevel(image, footprint, out=None, mask=None, '
    'shift_x=0, shift_y=0, shift_z=0)',
    'skimage.filters.rank:autolevel_percentile(image, footprint, out=None, '
    'mask=None, shift_x=0, shift_y=0, p0=0, p1=1)',
    'skimage.filters.rank:enhance_contrast(image, footprint, out=None, mask=None, '
    'shift_x=0, shift_y=0, shift_z=0)',
    'skimage.filters.rank:enhance_contrast_percentile(image, footprint, out=None, '
    'mask=None, shift_x=0, shift_y=0, p0=0, p1=1)',
    'skimage.filters.rank:entropy(image, footprint, out=None, mask=None, '
    'shift_x=0, shift_y=0, shift_z=0)',
    'skimage.filters.rank:equalize(image, footprint, out=None, mask=None, '
    'shift_x=0, shift_y=0, shift_z=0)',
    'skimage.filters.rank:geometric_mean(image, footprint, out=None, mask=None, '
    'shift_x=0, shift_y=0, shift_z=0)',
    'skimage.filters.rank:gradient(image, footprint, out=None, mask=None, '
    'shift_x=0, shift_y=0, shift_z=0)',
    'skimage.filters.rank:gradient_percentile(image, footprint, out=None, '
    'mask=None, shift_x=0, shift_y=0, p0=0, p1=1)',
    'skimage.filters.rank:majority(image, footprint, *, out=None, mask=None, '
    'shift_x=0, shift_y=0, shift_z=0)',
    'skimage.filters.rank:maximum(image, footprint, out=None, mask=None, '
    'shift_x=0, shift_y=0, shift_z=0)',
    'skimage.filters.rank:mean(image, footprint, out=None, mask=None, shift_x=0, '
    'shift_y=0, shift_z=0)',
    'skimage.filters.rank:mean_bilateral(image, footprint, out=None, mask=None, '
    'shift_x=0, shift_y=0, s0=10, s1=10)',
    'skimage.filters.rank:mean_percentile(image, footprint, out=None, mask=None, '
    'shift_x=0, shift_y=0, p0=0, p1=1)',
    'skimage.filters.rank:median(image, footprint=None, out=None, mask=None, '
    'shift_x=0, shift_y=0, shift_z=0)',
    'skimage.filters.rank:minimum(image, footprint, out=None, mask=None, '
    'shift_x=0, shift_y=0, shift_z=0)',
    'skimage.filters.rank:modal(image, footprint, out=None, mask=None, shift_x=0, '
    'shift_y=0, shift_z=0)',
    'skimage.filters.rank:noise_filter(image, footprint, out=None, mask=None, '
    'shift_x=0, shift_y=0, shift_z=0)',
    'skimage.filters.rank:otsu(image, footprint, out=None, mask=None, shift_x=0, '
    'shift_y=0, shift_z=0)',
    'skimage.filters.rank:percentile(image, footprint, out=None, mask=None, '
    'shift_x=0, shift_y=0, p0=0)',
    'skimage.filters.rank:pop(image, footprint, out=None, mask=None, shift_x=0, '
    'shift_y=0, shift_z=0)',
    'skimage.filters.rank:pop_bilateral(image, footprint, out=None, mask=None, '
    'shift_x=0, shift_y=0, s0=10, s1=10)',
    'skimage.filters.rank:pop_percentile(image, footprint, out=None, mask=None, '
    'shift_x=0, shift_y=0, p0=0, p1=1)',
    'skimage.filters.rank:subtract_mean(image, footprint, out=None, mask=None, '
    'shift_x=0, shift_y=0, shift_z=0)',
    'skimage.filters.rank:subtract_mean_percentile(image, footprint, out=None, '
    'mask=None, shift_x=0, shift_y=0, p0=0, p1=1)',
    'skimage.filters.rank:sum(image, footprint, out=None, mask=None, shift_x=0, '
    'shift_y=0, shift_z=0)',
    'skimage.filters.rank:sum_bilateral(image, footprint, out=None, mask=None, '
    'shift_x=0, shift_y=0, s0=10, s1=10)',
    'skimage.filters.rank:sum_percentile(image, footprint, out=None, mask=None, '
    'shift_x=0, shift_y=0, p0=0, p1=1)',
    'skimage.filters.rank:threshold(image, footprint, out=None, mask=None, '
    'shift_x=0, shift_y=0, shift_z=0)',
    'skimage.filters.rank:threshold_percentile(image, footprint, out=None, '
    'mask=None, shift_x=0, shift_y=0, p0=0)',
    'skimage.filters.rank:windowed_histogram(image, footprint, out=None, '
    'mask=None, shift_x=0, shift_y=0, n_bins=None)',
    'skimage.filters:LPIFilter2D',
    'skimage.filters:apply_hysteresis_threshold(image, low, high)',
    'skimage.filters:butterworth(image, cutoff_frequency_ratio=0.005, '
    'high_pass=True, order=2.0, channel_axis=None, *, squared_butterworth=True, '
    'npad=0)',
    "skimage.filters:correlate_sparse(image, kernel, mode='reflect')",
    'skimage.filters:difference_of_gaussians(image, low_sigma, high_sigma=None, '
    "*, mode='nearest', cval=0, channel_axis=None, truncate=4.0)",
    "skimage.filters:farid(image, mask=None, *, axis=None, mode='reflect', cval=0.0)",
    'skimage.filters:farid_h(image, *, mask=None)',
    'skimage.filters:farid_v(image, *, mask=None)',
    'skimage.filters:filter_forward(data, impulse_response=None, '
    'filter_params=None, predefined_filter=None)',
    'skimage.filters:filter_inverse(data, impulse_response=None, '
    'filter_params=None, max_gain=2, predefined_filter=None)',
    'skimage.filters:frangi(image, sigmas=range(1, 10, 2), scale_range=None, '
    'scale_step=None, alpha=0.5, beta=0.5, gamma=None, black_ridges=True, '
    "mode='reflect', cval=0)",
    'skimage.filters:gabor(image, frequency, theta=0, bandwidth=1, sigma_x=None, '
    "sigma_y=None, n_stds=3, offset=0, mode='reflect', cval=0)",
    'skimage.filters:gabor_kernel(frequency, theta=0, bandwidth=1, sigma_x=None, '
    "sigma_y=None, n_stds=3, offset=0, dtype=<class 'numpy.complex128'>)",
    "skimage.filters:gaussian(image, sigma=1.0, *, mode='nearest', cval=0, "
    'preserve_range=False, truncate=4.0, channel_axis=None, out=None)',
    'skimage.filters:hessian(image, sigmas=range(1, 10, 2), scale_range=None, '
    'scale_step=None, alpha=0.5, beta=0.5, gamma=15, black_ridges=True, '
    "mode='reflect', cval=0)",
    'skimage.filters:laplace(image, ksize=3, mask=None)',
    "skimage.filters:median(image, footprint=None, out=None, mode='nearest', "
    "cval=0.0, behavior='ndimage')",
    'skimage.filters:meijering(image, sigmas=range(1, 10, 2), alpha=None, '
    "black_ridges=True, mode='reflect', cval=0)",
    "skimage.filters:prewitt(image, mask=None, *, axis=None, mode='reflect', cval=0.0)",
    'skimage.filters:prewitt_h(image, mask=None)',
    'skimage.filters:prewitt_v(image, mask=None)',
    'skimage.filters:rank_order(image)',
    'skimage.filters:roberts(image, mask=None)',
    'skimage.filters:roberts_neg_diag(image, mask=None)',
    'skimage.filters:roberts_pos_diag(image, mask=None)',
    'skimage.filters:sato(image, sigmas=range(1, 10, 2), black_ridges=True, '
    "mode='reflect', cval=0)",
    "skimage.filters:scharr(image, mask=None, *, axis=None, mode='reflect', cval=0.0)",
    'skimage.filters:scharr_h(image, mask=None)',
    'skimage.filters:scharr_v(image, mask=None)',
    "skimage.filters:sobel(image, mask=None, *, axis=None, mode='reflect', cval=0.0)",
    'skimage.filters:sobel_h(image, mask=None)',
    'skimage.filters:sobel_v(image, mask=None)',
    'skimage.filters:threshold_isodata(image=None, nbins=256, return_all=False, '
    '*, hist=None)',
    'skimage.filters:threshold_li(image, *, tolerance=None, initial_guess=None, '
    'iter_callback=None)',
    "skimage.filters:threshold_local(image, block_size=3, method='gaussian', "
    "offset=0, mode='reflect', param=None, cval=0)",
    'skimage.filters:threshold_mean(image)',
    'skimage.filters:threshold_minimum(image=None, nbins=256, max_num_iter=10000, '
    '*, hist=None)',
    'skimage.filters:threshold_multiotsu(image=None, classes=3, nbins=256, *, '
    'hist=None)',
    'skimage.filters:threshold_niblack(image, window_size=15, k=0.2)',
    'skimage.filters:threshold_otsu(image=None, nbins=256, *, hist=None)',
    'skimage.filters:threshold_sauvola(image, window_size=15, k=0.2, r=None)',
    'skimage.filters:threshold_triangle(image, nbins=256)',
    'skimage.filters:threshold_yen(image=None, nbins=256, *, hist=None)',
    'skimage.filters:try_all_threshold(image, figsize=(8, 5), verbose=True)',
    'skimage.filters:unsharp_mask(image, radius=1.0, amount=1.0, '
    'preserve_range=False, *, channel_axis=None)',
    'skimage.filters:wiener(data, impulse_response=None, filter_params=None, '
    'K=0.25, predefined_filter=None)',
    'skimage.filters:window(window_type, shape, warp_kwargs=None)',
    'skimage.future:TrainableSegmenter',
    'skimage.future:fit_segmenter(labels, features, clf)',
    'skimage.future:manual_lasso_segmentation(image, alpha=0.4, return_all=False)',
    'skimage.future:manual_polygon_segmentation(image, alpha=0.4, return_all=False)',
    'skimage.future:predict_segmenter(features, clf)',
    'skimage.graph:MCP',
    'skimage.graph:MCP_Connect',
    'skimage.graph:MCP_Flexible',
    'skimage.graph:MCP_Geometric',
    'skimage.graph:RAG',
    'skimage.graph:central_pixel(graph, nodes=None, shape=None, partition_size=100)',
    'skimage.graph:cut_normalized(labels, rag, thresh=0.001, num_cuts=10, '
    'in_place=True, max_edge=1.0, *, rng=None)',
    'skimage.graph:cut_threshold(labels, rag, thresh, in_place=True)',
    'skimage.graph:merge_hierarchical(labels, rag, thresh, rag_copy, '
    'in_place_merge, merge_func, weight_func)',
    'skimage.graph:pixel_graph(image, *, mask=None, edge_function=None, '
    "connectivity=1, spacing=None, sparse_type='matrix')",
    'skimage.graph:rag_boundary(labels, edge_map, connectivity=2)',
    "skimage.graph:rag_mean_color(image, labels, connectivity=2, mode='distance', "
    'sigma=255.0)',
    'skimage.graph:route_through_array(array, start, end, fully_connected=True, '
    'geometric=True)',
    'skimage.graph:shortest_path(arr, reach=1, axis=-1, output_indexlist=False)',
    "skimage.graph:show_rag(labels, rag, image, border_color='black', "
    "edge_width=1.5, edge_cmap='magma', img_cmap='bone', in_place=True, ax=None)",
    'skimage.io:ImageCollection',
    'skimage.io:MultiImage',
    'skimage.io:concatenate_images(ic)',
    'skimage.io:imread(fname, as_gray=False, plugin=<DEPRECATED>, **plugin_args)',
    'skimage.io:imread_collection(load_pattern, conserve_memory=True, '
    'plugin=<DEPRECATED>, **plugin_args)',
    'skimage.io:imread_collection_wrapper(imread)',
    'skimage.io:imsave(fname, arr, plugin=<DEPRECATED>, *, check_contrast=True, '
    '**plugin_args)',
    'skimage.io:load_sift(f)',
    'skimage.io:load_surf(f)',
    'skimage.io:pop()',
    'skimage.io:push(img)',
    'skimage.measure:CircleModel',
    'skimage.measure:EllipseModel',
    'skimage.measure:LineModelND',
    'skimage.measure:RansacModelProtocol',
    'skimage.measure:approximate_polygon(coords, tolerance)',
    'skimage.measure:block_reduce(image, block_size=2, func=<function sum at '
    '0x...>, cval=0, func_kwargs=None)',
    'skimage.measure:blur_effect(image, h_size=11, channel_axis=None, '
    'reduce_func=<function max at 0x...>)',
    'skimage.measure:centroid(image, *, spacing=None)',
    'skimage.measure:euler_number(image, connectivity=None)',
    "skimage.measure:find_contours(image, level=None, fully_connected='low', "
    "positive_orientation='low', *, mask=None)",
    'skimage.measure:grid_points_in_poly(shape, verts, binarize=True)',
    'skimage.measure:inertia_tensor(image, mu=None, *, spacing=None)',
    'skimage.measure:inertia_tensor_eigvals(image, mu=None, T=None, *, spacing=None)',
    'skimage.measure:intersection_coeff(image0_mask, image1_mask, mask=None)',
    'skimage.measure:label(label_image, background=None, return_num=False, '
    'connectivity=None)',
    'skimage.measure:manders_coloc_coeff(image0, image1_mask, mask=None)',
    'skimage.measure:manders_overlap_coeff(image0, image1, mask=None)',
    'skimage.measure:marching_cubes(volume, level=None, *, spacing=(1.0, 1.0, '
    "1.0), gradient_direction='descent', step_size=1, allow_degenerate=True, "
    "method='lewiner', mask=None)",
    'skimage.measure:mesh_surface_area(verts, faces)',
    'skimage.measure:moments(image, order=3, *, spacing=None)',
    'skimage.measure:moments_central(image, center=None, order=3, *, '
    'spacing=None, **kwargs)',
    'skimage.measure:moments_coords(coords, order=3)',
    'skimage.measure:moments_coords_central(coords, center=None, order=3)',
    'skimage.measure:moments_hu(nu)',
    'skimage.measure:moments_normalized(mu, order=3, spacing=None)',
    'skimage.measure:pearson_corr_coeff(image0, image1, mask=None)',
    'skimage.measure:perimeter(image, neighborhood=4)',
    'skimage.measure:perimeter_crofton(image, directions=4)',
    'skimage.measure:points_in_poly(points, verts)',
    'skimage.measure:profile_line(image, src, dst, linewidth=1, order=None, '
    "mode='reflect', cval=0.0, *, reduce_func=<function mean at 0x...>)",
    'skimage.measure:ransac(data, model_class, min_samples, residual_threshold, '
    'is_data_valid=None, is_model_valid=None, max_trials=100, '
    'stop_sample_num=inf, stop_residuals_sum=0, stop_probability=1, rng=None, '
    'initial_inliers=None, model_kwargs=None)',
    'skimage.measure:regionprops(label_image, intensity_image=None, cache=True, '
    '*, extra_properties=None, spacing=None, offset=None)',
    'skimage.measure:regionprops_table(label_image, intensity_image=None, '
    "properties=('label', 'bbox'), *, cache=True, separator='-', "
    'extra_properties=None, spacing=None)',
    'skimage.measure:shannon_entropy(image, base=2)',
    'skimage.measure:subdivide_polygon(coords, degree=2, preserve_ends=False)',
    'skimage.metrics:adapted_rand_error(image_true=None, image_test=None, *, '
    'table=None, ignore_labels=(0,), alpha=0.5)',
    'skimage.metrics:contingency_table(im_true, im_test, *, ignore_labels=None, '
    "normalize=False, sparse_type='matrix')",
    "skimage.metrics:hausdorff_distance(image0, image1, method='standard')",
    'skimage.metrics:hausdorff_pair(image0, image1)',
    'skimage.metrics:mean_squared_error(image0, image1)',
    'skimage.metrics:normalized_mutual_information(image0, image1, *, bins=100)',
    'skimage.metrics:normalized_root_mse(image_true, image_test, *, '
    "normalization='euclidean')",
    'skimage.metrics:peak_signal_noise_ratio(image_true, image_test, *, '
    'data_range=None)',
    'skimage.metrics:structural_similarity(im1, im2, *, win_size=None, '
    'gradient=False, data_range=None, channel_axis=None, gaussian_weights=False, '
    'full=False, **kwargs)',
    'skimage.metrics:variation_of_information(image0=None, image1=None, *, '
    'table=None, ignore_labels=())',
    'skimage.morphology:area_closing(image, area_threshold=64, connectivity=1, '
    'parent=None, tree_traverser=None)',
    'skimage.morphology:area_opening(image, area_threshold=64, connectivity=1, '
    'parent=None, tree_traverser=None)',
    "skimage.morphology:ball(radius, dtype=<class 'numpy.uint8'>, *, "
    'strict_radius=True, decomposition=None)',
    'skimage.morphology:black_tophat(image, footprint=None, out=None, *, '
    "mode='reflect', cval=0.0)",
    'skimage.morphology:closing(image, footprint=None, out=None, *, '
    "mode='reflect', cval=0.0)",
    'skimage.morphology:convex_hull_image(image, offset_coordinates=True, '
    'tolerance=1e-10, include_borders=True)',
    'skimage.morphology:convex_hull_object(image, *, connectivity=2)',
    'skimage.morphology:diameter_closing(image, diameter_threshold=8, '
    'connectivity=1, parent=None, tree_traverser=None)',
    'skimage.morphology:diameter_opening(image, diameter_threshold=8, '
    'connectivity=1, parent=None, tree_traverser=None)',
    "skimage.morphology:diamond(radius, dtype=<class 'numpy.uint8'>, *, "
    'decomposition=None)',
    'skimage.morphology:dilation(image, footprint=None, out=None, *, '
    "mode='reflect', cval=0.0)",
    "skimage.morphology:disk(radius, dtype=<class 'numpy.uint8'>, *, "
    'strict_radius=True, decomposition=None)',
    "skimage.morphology:ellipse(width, height, dtype=<class 'numpy.uint8'>, *, "
    'decomposition=None)',
    'skimage.morphology:erosion(image, footprint=None, out=None, *, '
    "mode='reflect', cval=0.0)",
    'skimage.morphology:flood(image, seed_point, *, footprint=None, '
    'connectivity=None, tolerance=None)',
    'skimage.morphology:flood_fill(image, seed_point, new_value, *, '
    'footprint=None, connectivity=None, tolerance=None, in_place=False)',
    'skimage.morphology:footprint_from_sequence(footprints)',
    'skimage.morphology:footprint_rectangle(shape, *, dtype=<class '
    "'numpy.uint8'>, decomposition=None)",
    'skimage.morphology:h_maxima(image, h, footprint=None)',
    'skimage.morphology:h_minima(image, h, footprint=None)',
    'skimage.morphology:isotropic_closing(image, radius, out=None, spacing=None)',
    'skimage.morphology:isotropic_dilation(image, radius, out=None, spacing=None)',
    'skimage.morphology:isotropic_erosion(image, radius, out=None, spacing=None)',
    'skimage.morphology:isotropic_opening(image, radius, out=None, spacing=None)',
    'skimage.morphology:label(label_image, background=None, return_num=False, '
    'connectivity=None)',
    'skimage.morphology:local_maxima(image, footprint=None, connectivity=None, '
    'indices=False, allow_borders=True)',
    'skimage.morphology:local_minima(image, footprint=None, connectivity=None, '
    'indices=False, allow_borders=True)',
    'skimage.morphology:max_tree(image, connectivity=1)',
    'skimage.morphology:max_tree_local_maxima(image, connectivity=1, parent=None, '
    'tree_traverser=None)',
    'skimage.morphology:medial_axis(image, mask=None, return_distance=False, *, '
    'rng=None)',
    'skimage.morphology:mirror_footprint(footprint)',
    "skimage.morphology:octagon(m, n, dtype=<class 'numpy.uint8'>, *, "
    'decomposition=None)',
    "skimage.morphology:octahedron(radius, dtype=<class 'numpy.uint8'>, *, "
    'decomposition=None)',
    'skimage.morphology:opening(image, footprint=None, out=None, *, '
    "mode='reflect', cval=0.0)",
    'skimage.morphology:pad_footprint(footprint, *, pad_end=True)',
    "skimage.morphology:reconstruction(seed, mask, method='dilation', "
    'footprint=None, offset=None)',
    'skimage.morphology:remove_objects_by_distance(label_image, min_distance, *, '
    'priority=None, p_norm=2, spacing=None, out=None)',
    'skimage.morphology:remove_small_holes(ar, area_threshold=<DEPRECATED>, '
    'connectivity=1, *, max_size=63, out=None)',
    'skimage.morphology:remove_small_objects(ar, min_size=<DEPRECATED>, '
    'connectivity=1, *, max_size=63, out=None)',
    'skimage.morphology:skeletonize(image, *, method=None)',
    "skimage.morphology:star(a, dtype=<class 'numpy.uint8'>)",
    'skimage.morphology:thin(image, max_num_iter=None)',
    'skimage.morphology:white_tophat(image, footprint=None, out=None, *, '
    "mode='reflect', cval=0.0)",
    'skimage.registration:optical_flow_ilk(reference_image, moving_image, *, '
    'radius=7, num_warp=10, gaussian=False, prefilter=False, dtype=<class '
    "'numpy.float32'>)",
    'skimage.registration:optical_flow_tvl1(reference_image, moving_image, *, '
    'attachment=15, tightness=0.3, num_warp=5, num_iter=10, tol=0.0001, '
    "prefilter=False, dtype=<class 'numpy.float32'>)",
    'skimage.registration:phase_cross_correlation(reference_image, moving_image, '
    "*, upsample_factor=1, space='real', disambiguate=False, reference_mask=None, "
    "moving_mask=None, overlap_ratio=0.3, normalization='phase')",
    'skimage.restoration:ball_kernel(radius, ndim)',
    'skimage.restoration:calibrate_denoiser(image, denoise_function, '
    'denoise_parameters, *, stride=4, approximate_loss=True, extra_output=False)',
    'skimage.restoration:cycle_spin(x, func, max_shifts, shift_steps=1, '
    'num_workers=<DEPRECATED>, func_kw=None, *, workers=None, channel_axis=None)',
    'skimage.restoration:denoise_bilateral(image, win_size=None, '
    "sigma_color=None, sigma_spatial=1, bins=10000, mode='constant', cval=0, *, "
    'channel_axis=None)',
    'skimage.restoration:denoise_invariant(image, denoise_function, *, stride=4, '
    'masks=None, denoiser_kwargs=None)',
    'skimage.restoration:denoise_nl_means(image, patch_size=7, patch_distance=11, '
    'h=0.1, fast_mode=True, sigma=0.0, *, preserve_range=False, '
    'channel_axis=None)',
    'skimage.restoration:denoise_tv_bregman(image, weight=5.0, max_num_iter=100, '
    'eps=0.001, isotropic=True, *, channel_axis=None)',
    'skimage.restoration:denoise_tv_chambolle(image, weight=0.1, eps=0.0002, '
    'max_num_iter=200, *, channel_axis=None)',
    "skimage.restoration:denoise_wavelet(image, sigma=None, wavelet='db1', "
    "mode='soft', wavelet_levels=None, convert2ycbcr=False, method='BayesShrink', "
    'rescale_sigma=True, *, channel_axis=None)',
    'skimage.restoration:ellipsoid_kernel(shape, intensity)',
    'skimage.restoration:estimate_sigma(image, average_sigmas=False, *, '
    'channel_axis=None)',
    'skimage.restoration:inpaint_biharmonic(image, mask, *, '
    'split_into_regions=False, channel_axis=None)',
    'skimage.restoration:richardson_lucy(image, psf, num_iter=50, clip=True, '
    'filter_epsilon=None)',
    'skimage.restoration:rolling_ball(image, *, radius=100, kernel=None, '
    'nansafe=False, num_threads=<DEPRECATED>, workers=None)',
    'skimage.restoration:unsupervised_wiener(image, psf, reg=None, '
    'user_params=None, is_real=True, clip=True, *, rng=None)',
    'skimage.restoration:unwrap_phase(image, wrap_around=False, rng=None)',
    'skimage.restoration:wiener(image, psf, balance, reg=None, is_real=True, '
    'clip=True)',
    'skimage.segmentation:active_contour(image, snake, alpha=0.01, beta=0.1, '
    'w_line=0.0, w_edge=1, gamma=0.01, max_px_move=1.0, max_num_iter=2500, '
    "convergence=0.1, *, boundary_condition='periodic')",
    'skimage.segmentation:chan_vese(image, mu=0.25, lambda1=1.0, lambda2=1.0, '
    "tol=0.001, max_num_iter=500, dt=0.5, init_level_set='checkerboard', "
    'extended_output=False)',
    'skimage.segmentation:checkerboard_level_set(image_shape, square_size=5)',
    'skimage.segmentation:clear_border(labels, buffer_size=0, bgval=0, mask=None, '
    '*, out=None)',
    'skimage.segmentation:disk_level_set(image_shape, *, center=None, radius=None)',
    'skimage.segmentation:expand_labels(label_image, distance=1, spacing=1)',
    'skimage.segmentation:felzenszwalb(image, scale=1, sigma=0.8, min_size=20, *, '
    'channel_axis=-1)',
    'skimage.segmentation:find_boundaries(label_img, connectivity=1, '
    "mode='thick', background=0)",
    'skimage.segmentation:flood(image, seed_point, *, footprint=None, '
    'connectivity=None, tolerance=None)',
    'skimage.segmentation:flood_fill(image, seed_point, new_value, *, '
    'footprint=None, connectivity=None, tolerance=None, in_place=False)',
    'skimage.segmentation:inverse_gaussian_gradient(image, alpha=100.0, sigma=5.0)',
    'skimage.segmentation:join_segmentations(s1, s2, return_mapping: bool = False)',
    'skimage.segmentation:mark_boundaries(image, label_img, color=(1, 1, 0), '
    "outline_color=None, mode='outer', background_label=0)",
    'skimage.segmentation:morphological_chan_vese(image, num_iter, '
    "init_level_set='checkerboard', smoothing=1, lambda1=1, lambda2=1, "
    'iter_callback=<function <lambda> at 0x...>)',
    'skimage.segmentation:morphological_geodesic_active_contour(gimage, num_iter, '
    "init_level_set='disk', smoothing=1, threshold='auto', balloon=0, "
    'iter_callback=<function <lambda> at 0x...>)',
    'skimage.segmentation:quickshift(image, ratio=1.0, kernel_size=5, '
    'max_dist=10, return_tree=False, sigma=0, convert2lab=True, rng=42, *, '
    'channel_axis=-1)',
    "skimage.segmentation:random_walker(data, labels, beta=130, mode='cg_j', "
    'tol=0.001, copy=True, return_full_prob=False, spacing=None, *, '
    'prob_tol=0.001, channel_axis=None)',
    'skimage.segmentation:relabel_sequential(label_field, offset=1)',
    'skimage.segmentation:slic(image, n_segments=100, compactness=10.0, '
    'max_num_iter=10, sigma=0, spacing=None, convert2lab=None, '
    'enforce_connectivity=True, min_size_factor=0.5, max_size_factor=3, '
    'slic_zero=False, start_label=1, mask=None, *, channel_axis=-1)',
    'skimage.segmentation:watershed(image, markers=None, connectivity=1, '
    'offset=None, mask=None, compactness=0, watershed_line=False)',
    'skimage.transform:AffineTransform',
    'skimage.transform:EssentialMatrixTransform',
    'skimage.transform:EuclideanTransform',
    'skimage.transform:FundamentalMatrixTransform',
    'skimage.transform:PiecewiseAffineTransform',
    'skimage.transform:PolynomialTransform',
    'skimage.transform:ProjectiveTransform',
    'skimage.transform:SimilarityTransform',
    'skimage.transform:ThinPlateSplineTransform',
    'skimage.transform:downscale_local_mean(image, factors, cval=0, clip=True)',
    'skimage.transform:estimate_transform(ttype, src, dst, *args, **kwargs)',
    'skimage.transform:frt2(a)',
    'skimage.transform:hough_circle(image, radius, normalize=True, full_output=False)',
    'skimage.transform:hough_circle_peaks(hspaces, radii, min_xdistance=1, '
    'min_ydistance=1, threshold=None, num_peaks=inf, total_num_peaks=inf, '
    'normalize=False)',
    'skimage.transform:hough_ellipse(image, threshold=4, accuracy=1, min_size=4, '
    'max_size=None)',
    'skimage.transform:hough_line(image, theta=None)',
    'skimage.transform:hough_line_peaks(hspace, angles, dists, min_distance=9, '
    'min_angle=10, threshold=None, num_peaks=inf)',
    'skimage.transform:ifrt2(a)',
    'skimage.transform:integral_image(image, *, dtype=None)',
    'skimage.transform:integrate(ii, start, end)',
    'skimage.transform:iradon(radon_image, theta=None, output_size=None, '
    "filter_name='ramp', interpolation='linear', circle=True, "
    'preserve_range=True)',
    'skimage.transform:iradon_sart(radon_image, theta=None, image=None, '
    'projection_shifts=None, clip=None, relaxation=0.15, dtype=None)',
    'skimage.transform:matrix_transform(coords, matrix)',
    'skimage.transform:order_angles_golden_ratio(theta)',
    'skimage.transform:probabilistic_hough_line(image, threshold=10, '
    'line_length=50, line_gap=10, theta=None, rng=None)',
    'skimage.transform:pyramid_expand(image, upscale=2, sigma=None, order=1, '
    "mode='reflect', cval=0, preserve_range=False, *, channel_axis=None)",
    'skimage.transform:pyramid_gaussian(image, max_layer=-1, downscale=2, '
    "sigma=None, order=1, mode='reflect', cval=0, preserve_range=False, *, "
    'channel_axis=None)',
    'skimage.transform:pyramid_laplacian(image, max_layer=-1, downscale=2, '
    "sigma=None, order=1, mode='reflect', cval=0, preserve_range=False, *, "
    'channel_axis=None)',
    'skimage.transform:pyramid_reduce(image, downscale=2, sigma=None, order=1, '
    "mode='reflect', cval=0, preserve_range=False, *, channel_axis=None)",
    'skimage.transform:radon(image, theta=None, circle=True, *, preserve_range=False)',
    "skimage.transform:rescale(image, scale, order=None, mode='reflect', cval=0, "
    'clip=True, preserve_range=False, anti_aliasing=None, '
    'anti_aliasing_sigma=None, *, channel_axis=None)',
    "skimage.transform:resize(image, output_shape, order=None, mode='reflect', "
    'cval=0, clip=True, preserve_range=False, anti_aliasing=None, '
    'anti_aliasing_sigma=None)',
    'skimage.transform:resize_local_mean(image, output_shape, grid_mode=True, '
    'preserve_range=False, *, channel_axis=None)',
    'skimage.transform:rotate(image, angle, resize=False, center=None, '
    "order=None, mode='constant', cval=0, clip=True, preserve_range=False)",
    'skimage.transform:swirl(image, center=None, strength=1, radius=100, '
    "rotation=0, output_shape=None, order=None, mode='reflect', cval=0, "
    'clip=True, preserve_range=False)',
    'skimage.transform:warp(image, inverse_map, map_args=None, output_shape=None, '
    "order=None, mode='constant', cval=0.0, clip=True, preserve_range=False)",
    'skimage.transform:warp_coords(coord_map, shape, dtype=<class \'numpy.float64\'>)',
    'skimage.transform:warp_polar(image, center=None, *, radius=None, '
    "output_shape=None, scaling='linear', channel_axis=None, **kwargs)",
    'skimage.util:FailedEstimationAccessError',
    'skimage.util:PendingSkimage2Change',
    'skimage.util:apply_parallel(function, array, chunks=None, depth=0, '
    'mode=None, extra_arguments=(), extra_keywords=None, *, dtype=None, '
    'compute=None, channel_axis=None)',
    "skimage.util:compare_images(image0, image1, *, method='diff', n_tiles=(8, 8))",
    "skimage.util:crop(ar, crop_width, copy=False, order='K')",
    'skimage.util:dtype_limits(image, clip_negative=False)',
    'skimage.util:img_as_bool(image, force_copy=False)',
    'skimage.util:img_as_float(image, force_copy=False)',
    'skimage.util:img_as_float32(image, force_copy=False)',
    'skimage.util:img_as_float64(image, force_copy=False)',
    'skimage.util:img_as_int(image, force_copy=False)',
    'skimage.util:img_as_ubyte(image, force_copy=False)',
    'skimage.util:img_as_uint(image, force_copy=False)',
    'skimage.util:invert(image, signed_float=False)',
    'skimage.util:label_points(coords, output_shape)',
    'skimage.util:lookfor(what)',
    'skimage.util:map_array(input_arr, input_vals, output_vals, out=None)',
    "skimage.util:montage(arr_in, fill='mean', rescale_intensity=False, "
    'grid_shape=None, padding_width=0, *, channel_axis=None)',
    "skimage.util:random_noise(image, mode='gaussian', rng=None, clip=True, **kwargs)",
    'skimage.util:regular_grid(ar_shape, n_points)',
    "skimage.util:regular_seeds(ar_shape, n_points, dtype=<class 'int'>)",
    'skimage.util:slice_along_axes(image, slices, axes=None, copy=False)',
    'skimage.util:unique_rows(ar)',
    'skimage.util:view_as_blocks(arr_in, block_shape)',
    'skimage.util:view_as_windows(arr_in, window_shape, step=1)',
    'skimage:__version__',
}


def _walk_api_via_dunder_all(module):
    """Walk the API tree that is defined by ``__all__`` attributes in modules.

    Parameters
    ----------
    module : ModuleType
        The root of the API tree to walk.

    Yields
    ------
    qualname : str
        The qualified name of the API object. The module boundary is defined
        with ":" instead of ".", for example "skimage.data:logo()".
    obj : Any
        The API object.
    """
    for name in sorted(getattr(module, "__all__", [])):
        obj = getattr(module, name)
        if inspect.ismodule(obj):
            yield from _walk_api_via_dunder_all(obj)
        else:
            qualname = f"{module.__name__}:{name}"
            yield qualname, obj


def test_public_skimage_api(tmp_path):
    import skimage

    # Capitalization and length may vary with platform
    re_hex_id = re.compile(r"at 0x[0-9a-fA-F]{1,16}>")

    skimage_api = set()
    for qualname, obj in _walk_api_via_dunder_all(skimage):
        assert "._" not in qualname

        key = qualname

        if inspect.isfunction(obj):
            # For functions, include signature in key
            signature = inspect.signature(obj)
            # Strip unique id for objects that include it in representation
            signature = re_hex_id.sub("at 0x...>", str(signature))
            key = f"{key}{signature}"

        # TODO Record signature of methods too

        skimage_api.add(key)

    # Save results to `tmp_dir` for easier "diffing". Generate diff with:
    # git diff --no-index -U0 expected.txt actual.txt
    with (tmp_path / "actual.txt").open("w") as f:
        f.writelines(pformat(skimage_api))
    with (tmp_path / "expected.txt").open("w") as f:
        f.writelines(pformat(SKIMAGE_API))

    assert skimage_api == SKIMAGE_API


@pytest.mark.skipif(
    sys.flags.optimize >= 2, reason="docstrings unavailable with PYTHONOPTIMIZE=2"
)
def test_skimage_api_docstring_imports():
    # Check that wrappers in skimage don't mention skimage2.
    # Only checks objects whose "__module__" attribute starts with "skimage."!
    import skimage

    exceptions = {
        "skimage.io:imread_collection_wrapper",  # No docstring in v0.26
    }
    for qualname, obj in _walk_api_via_dunder_all(skimage):
        if qualname in exceptions:
            continue
        if not getattr(obj, "__module__", "").startswith("skimage."):
            continue
        assert obj.__doc__
        assert "import skimage2" not in obj.__doc__
        assert "import _skimage2" not in obj.__doc__
        assert "from skimage2" not in obj.__doc__
        assert "from _skimage2" not in obj.__doc__

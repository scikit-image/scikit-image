"""
Functions for converting between color spaces.

The "central" color space in this module is RGB, more specifically the linear
sRGB color space using D65 as a white-point [1]_.  This represents a
standard monitor (w/o gamma correction). For a good FAQ on color spaces see
[2]_.

The API consists of functions to convert to and from RGB as defined above, as
well as a generic function to convert to and from any supported color space
(which is done through RGB in most cases).


Supported color spaces
----------------------
* RGB : Red Green Blue.
        Here the sRGB standard [1]_.
* HSV : Hue, Saturation, Value.
        Uniquely defined when related to sRGB [3]_.
* RGB CIE : Red Green Blue.
        The original RGB CIE standard from 1931 [4]_. Primary colors are 700 nm
        (red), 546.1 nm (blue) and 435.8 nm (green).
* XYZ CIE : XYZ
        Derived from the RGB CIE color space. Chosen such that
        ``x == y == z == 1/3`` at the whitepoint, and all color matching
        functions are greater than zero everywhere.
* LAB CIE : Lightness, a, b
        Colorspace derived from XYZ CIE that is intended to be more
        perceptually uniform
* LUV CIE : Lightness, u, v
        Colorspace derived from XYZ CIE that is intended to be more
        perceptually uniform
* LCH CIE : Lightness, Chroma, Hue
        Defined in terms of LAB CIE.  C and H are the polar representation of
        a and b.  The polar angle C is defined to be on ``(0, 2*pi)``

:author: Nicolas Pinto (rgb2hsv)
:author: Ralf Gommers (hsv2rgb)
:author: Travis Oliphant (XYZ and RGB CIE functions)
:author: Matt Terry (lab2lch)
:author: Alex Izvorski (yuv2rgb, rgb2yuv and related)

:license: modified BSD

References
----------
.. [1] Official specification of sRGB, IEC 61966-2-1:1999.
.. [2] http://www.poynton.com/ColorFAQ.html
.. [3] https://en.wikipedia.org/wiki/HSL_and_HSV
.. [4] https://en.wikipedia.org/wiki/CIE_1931_color_space

"""

from _skimage2.color.colorconv import (
    ahx_from_rgb as ahx_from_rgb,
    bex_from_rgb as bex_from_rgb,
    bpx_from_rgb as bpx_from_rgb,
    bro_from_rgb as bro_from_rgb,
    cie_primaries as cie_primaries,
    combine_stains as combine_stains,
    convert_colorspace as convert_colorspace,
    fgx_from_rgb as fgx_from_rgb,
    gdx_from_rgb as gdx_from_rgb,
    gray2rgb as gray2rgb,
    gray2rgba as gray2rgba,
    gray_from_rgb as gray_from_rgb,
    hax_from_rgb as hax_from_rgb,
    hdx_from_rgb as hdx_from_rgb,
    hed2rgb as hed2rgb,
    hed_from_rgb as hed_from_rgb,
    hpx_from_rgb as hpx_from_rgb,
    hsv2rgb as hsv2rgb,
    lab2lch as lab2lch,
    lab2rgb as lab2rgb,
    lab2xyz as lab2xyz,
    lab_ref_white as lab_ref_white,
    lch2lab as lch2lab,
    luv2rgb as luv2rgb,
    luv2xyz as luv2xyz,
    rbd_from_rgb as rbd_from_rgb,
    rgb2gray as rgb2gray,
    rgb2hed as rgb2hed,
    rgb2hsv as rgb2hsv,
    rgb2lab as rgb2lab,
    rgb2luv as rgb2luv,
    rgb2rgbcie as rgb2rgbcie,
    rgb2xyz as rgb2xyz,
    rgb2ycbcr as rgb2ycbcr,
    rgb2ydbdr as rgb2ydbdr,
    rgb2yiq as rgb2yiq,
    rgb2ypbpr as rgb2ypbpr,
    rgb2yuv as rgb2yuv,
    rgb_from_ahx as rgb_from_ahx,
    rgb_from_bex as rgb_from_bex,
    rgb_from_bpx as rgb_from_bpx,
    rgb_from_bro as rgb_from_bro,
    rgb_from_fgx as rgb_from_fgx,
    rgb_from_gdx as rgb_from_gdx,
    rgb_from_hax as rgb_from_hax,
    rgb_from_hdx as rgb_from_hdx,
    rgb_from_hed as rgb_from_hed,
    rgb_from_hpx as rgb_from_hpx,
    rgb_from_rbd as rgb_from_rbd,
    rgb_from_rgbcie as rgb_from_rgbcie,
    rgb_from_xyz as rgb_from_xyz,
    rgb_from_ycbcr as rgb_from_ycbcr,
    rgb_from_ydbdr as rgb_from_ydbdr,
    rgb_from_yiq as rgb_from_yiq,
    rgb_from_ypbpr as rgb_from_ypbpr,
    rgb_from_yuv as rgb_from_yuv,
    rgba2rgb as rgba2rgb,
    rgbcie2rgb as rgbcie2rgb,
    rgbcie_from_rgb as rgbcie_from_rgb,
    rgbcie_from_xyz as rgbcie_from_xyz,
    sb_primaries as sb_primaries,
    separate_stains as separate_stains,
    xyz2lab as xyz2lab,
    xyz2luv as xyz2luv,
    xyz2rgb as xyz2rgb,
    xyz_from_rgb as xyz_from_rgb,
    xyz_from_rgbcie as xyz_from_rgbcie,
    xyz_tristimulus_values as xyz_tristimulus_values,
    ycbcr2rgb as ycbcr2rgb,
    ycbcr_from_rgb as ycbcr_from_rgb,
    ydbdr2rgb as ydbdr2rgb,
    ydbdr_from_rgb as ydbdr_from_rgb,
    yiq2rgb as yiq2rgb,
    yiq_from_rgb as yiq_from_rgb,
    ypbpr2rgb as ypbpr2rgb,
    ypbpr_from_rgb as ypbpr_from_rgb,
    yuv2rgb as yuv2rgb,
    yuv_from_rgb as yuv_from_rgb,
)  # noqa: F401

__all__ = [
    'ahx_from_rgb',
    'bex_from_rgb',
    'bpx_from_rgb',
    'bro_from_rgb',
    'cie_primaries',
    'combine_stains',
    'convert_colorspace',
    'fgx_from_rgb',
    'gdx_from_rgb',
    'gray2rgb',
    'gray2rgba',
    'gray_from_rgb',
    'hax_from_rgb',
    'hdx_from_rgb',
    'hed2rgb',
    'hed_from_rgb',
    'hpx_from_rgb',
    'hsv2rgb',
    'lab2lch',
    'lab2rgb',
    'lab2xyz',
    'lab_ref_white',
    'lch2lab',
    'luv2rgb',
    'luv2xyz',
    'rbd_from_rgb',
    'rgb2gray',
    'rgb2hed',
    'rgb2hsv',
    'rgb2lab',
    'rgb2luv',
    'rgb2rgbcie',
    'rgb2xyz',
    'rgb2ycbcr',
    'rgb2ydbdr',
    'rgb2yiq',
    'rgb2ypbpr',
    'rgb2yuv',
    'rgb_from_ahx',
    'rgb_from_bex',
    'rgb_from_bpx',
    'rgb_from_bro',
    'rgb_from_fgx',
    'rgb_from_gdx',
    'rgb_from_hax',
    'rgb_from_hdx',
    'rgb_from_hed',
    'rgb_from_hpx',
    'rgb_from_rbd',
    'rgb_from_rgbcie',
    'rgb_from_xyz',
    'rgb_from_ycbcr',
    'rgb_from_ydbdr',
    'rgb_from_yiq',
    'rgb_from_ypbpr',
    'rgb_from_yuv',
    'rgba2rgb',
    'rgbcie2rgb',
    'rgbcie_from_rgb',
    'rgbcie_from_xyz',
    'sb_primaries',
    'separate_stains',
    'xyz2lab',
    'xyz2luv',
    'xyz2rgb',
    'xyz_from_rgb',
    'xyz_from_rgbcie',
    'xyz_tristimulus_values',
    'ycbcr2rgb',
    'ycbcr_from_rgb',
    'ydbdr2rgb',
    'ydbdr_from_rgb',
    'yiq2rgb',
    'yiq_from_rgb',
    'ypbpr2rgb',
    'ypbpr_from_rgb',
    'yuv2rgb',
    'yuv_from_rgb',
]

from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())

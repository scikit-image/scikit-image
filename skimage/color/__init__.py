"""Utilities for color conversion, color labeling, and color difference
calculations.
"""

from .colorconv import (convert_colorspace,
                        xyz_tristimulus_values,
                        rgba2rgb,
                        rgb2hsv,
                        hsv2rgb,
                        rgb2xyz,
                        xyz2rgb,
                        rgb2rgbcie,
                        rgbcie2rgb,
                        rgb2gray,
                        gray2rgb,
                        gray2rgba,
                        xyz2lab,
                        lab2xyz,
                        lab2rgb,
                        rgb2lab,
                        xyz2luv,
                        luv2xyz,
                        luv2rgb,
                        rgb2luv,
                        rgb2hed,
                        hed2rgb,
                        lab2lch,
                        lch2lab,
                        rgb2yuv,
                        yuv2rgb,
                        rgb2yiq,
                        yiq2rgb,
                        rgb2ypbpr,
                        ypbpr2rgb,
                        rgb2ycbcr,
                        ycbcr2rgb,
                        rgb2ydbdr,
                        ydbdr2rgb,
                        separate_stains,
                        combine_stains,
                        rgb_from_hed,
                        hed_from_rgb,
                        rgb_from_hdx,
                        hdx_from_rgb,
                        rgb_from_fgx,
                        fgx_from_rgb,
                        rgb_from_bex,
                        bex_from_rgb,
                        rgb_from_rbd,
                        rbd_from_rgb,
                        rgb_from_gdx,
                        gdx_from_rgb,
                        rgb_from_hax,
                        hax_from_rgb,
                        rgb_from_bro,
                        bro_from_rgb,
                        rgb_from_bpx,
                        bpx_from_rgb,
                        rgb_from_ahx,
                        ahx_from_rgb,
                        rgb_from_hpx,
                        hpx_from_rgb)

from .colorlabel import color_dict, label2rgb

from .bayer2rgb import bayer2rgb

from .delta_e import (deltaE_cie76,
                      deltaE_ciede94,
                      deltaE_ciede2000,
                      deltaE_cmc,
                      )


__all__ = ['convert_colorspace',
           'xyz_tristimulus_values',
           'rgba2rgb',
           'rgb2hsv',
           'hsv2rgb',
           'rgb2xyz',
           'xyz2rgb',
           'rgb2rgbcie',
           'rgbcie2rgb',
           'rgb2gray',
           'gray2rgb',
           'gray2rgba',
           'xyz2lab',
           'lab2xyz',
           'lab2rgb',
           'rgb2lab',
           'rgb2hed',
           'hed2rgb',
           'lab2lch',
           'lch2lab',
           'rgb2yuv',
           'yuv2rgb',
           'rgb2yiq',
           'yiq2rgb',
           'rgb2ypbpr',
           'ypbpr2rgb',
           'rgb2ycbcr',
           'ycbcr2rgb',
           'rgb2ydbdr',
           'ydbdr2rgb',
           'separate_stains',
           'combine_stains',
           'rgb_from_hed',
           'hed_from_rgb',
           'rgb_from_hdx',
           'hdx_from_rgb',
           'rgb_from_fgx',
           'fgx_from_rgb',
           'rgb_from_bex',
           'bex_from_rgb',
           'rgb_from_rbd',
           'rbd_from_rgb',
           'rgb_from_gdx',
           'gdx_from_rgb',
           'rgb_from_hax',
           'hax_from_rgb',
           'rgb_from_bro',
           'bro_from_rgb',
           'rgb_from_bpx',
           'bpx_from_rgb',
           'rgb_from_ahx',
           'ahx_from_rgb',
           'rgb_from_hpx',
           'hpx_from_rgb',
           'color_dict',
           'label2rgb',
           'deltaE_cie76',
           'deltaE_ciede94',
           'deltaE_ciede2000',
           'deltaE_cmc',
           ]

import numpy as np
from skimage.measure import divide_selfoverlapping

from skimage._shared.testing import assert_array_equal


# A self-overlapping polygon that can be divided into three
# non self-overlapping polygons.
sample_cw_overlap = np.array([[200, 271], [251, 267], [312, 267], [381, 269],
                              [425, 271], [471, 321], [483, 367], [474, 416],
                              [436, 478], [370, 510], [279, 512], [188, 504],
                              [91, 470], [36, 414], [0, 303], [0, 206],
                              [31, 134], [98, 75], [193, 31], [293, 0],
                              [391, 1], [461, 51], [502, 111], [512, 210],
                              [446, 252], [422, 271], [381, 269], [309, 271],
                              [254, 265], [199, 269], [169, 248], [146, 277],
                              [168, 307]])

# A self-overlapping polygon with traversion direction given in counter
# clockwise direction.
sample_ccw_overlap = np.array([[168, 307], [146, 277], [169, 248], [199, 269],
                               [254, 265], [309, 271], [381, 269], [422, 271],
                               [446, 252], [512, 210], [502, 111], [461,  51],
                               [391, 1], [293, 0], [193, 31], [98,  75],
                               [31, 134], [0, 206], [0, 303], [36, 414],
                               [91, 470], [188, 504], [279, 512], [370, 510],
                               [436, 478], [474, 416], [483, 367], [471, 321],
                               [425, 271], [381, 269], [312, 267], [251, 267],
                               [200, 271]])

# A non self-overlapping polygon that cannot be divided into non
# self-overlapping polygons.
sample_cw_no_overlap = np.array([[280, 512], [0, 325], [243, 0], [512, 242]])

# A non self-overlapping polygon with traversion direction given in counter
# clockwise direction.
sample_ccw_no_overlap = np.array([[512, 242], [243, 0], [0, 325], [280, 512]])

# A polygon that self-intersects, but is not self-overlapping.
sample_cw_self_inter = np.array([[473, 272], [202, 512], [290, 508], [0, 221],
                                 [0, 336], [310, 0], [180, 19], [512, 304],
                                 [512, 199]])

# A polygon that self-intersects, but is not self-overlapping, and its
# vertices are in given counter clockwise direction.
sample_ccw_self_inter = np.array([[512, 199], [512, 304], [180, 19], [310, 0],
                                  [0, 336], [0, 221], [290, 508], [202, 512],
                                  [473, 272]])

# A polygon that self-intersects, is self-overlapping, and its
# vertices are in given counter clockwise direction.
sample_cw_self_inter_overlap = np.array([[376, 0], [439, 21], [480, 70],
                                         [512, 191], [512, 242], [501, 308],
                                         [463, 407], [422, 455], [370, 480],
                                         [230, 512], [176, 478], [141, 439],
                                         [98, 350], [45, 322], [9, 360],
                                         [0, 418], [16, 454], [76, 396],
                                         [116, 305], [136, 240], [175, 207],
                                         [212, 202], [258, 206], [290, 229],
                                         [325, 245], [379, 234], [384, 203],
                                         [372, 191], [317, 201], [272, 230],
                                         [229, 246], [157, 234], [134, 193],
                                         [105, 129], [76, 85], [45, 58],
                                         [28, 98], [20, 139], [72, 145],
                                         [121, 113], [168, 67], [237, 21],
                                         [326, 3]])

# A polygon that self-intersects, and is self-overlapping.
sample_ccw_self_inter_overlap = np.array([[326, 3], [237, 21], [168, 67],
                                          [121, 113], [72, 145], [20, 139],
                                          [28, 98], [45, 58], [76, 85],
                                          [105, 129], [134, 193], [157, 234],
                                          [229, 246], [272, 230], [317, 201],
                                          [372, 191], [384, 203], [379, 234],
                                          [325, 245], [290, 229], [258, 206],
                                          [212, 202], [175, 207], [136, 240],
                                          [116, 305], [76, 396], [16, 454],
                                          [0, 418], [9, 360], [45, 322],
                                          [98, 350], [141, 439], [176, 478],
                                          [230, 512], [370, 480], [422, 455],
                                          [463, 407], [501, 308], [512, 242],
                                          [512, 191], [480, 70], [439, 21],
                                          [376, 0]])


def test_overlap():
    sub_polys = divide_selfoverlapping(sample_cw_overlap)
    assert len(sub_polys) == 3

    sub_polys = divide_selfoverlapping(sample_ccw_overlap)
    assert len(sub_polys) == 3


def test_no_overlap():
    # No self-overlapping polygons are retured with no change
    sub_polys = divide_selfoverlapping(sample_cw_no_overlap)
    assert len(sub_polys) == 1
    assert_array_equal(sub_polys[0], sample_cw_no_overlap)

    sub_polys = divide_selfoverlapping(sample_ccw_no_overlap)
    assert len(sub_polys) == 1
    assert_array_equal(sub_polys[0], sample_ccw_no_overlap)


def test_no_overlap_self_intersect():
    sub_polys = divide_selfoverlapping(sample_cw_self_inter)
    assert len(sub_polys) == 1
    assert_array_equal(sub_polys[0], sample_cw_self_inter)

    sub_polys = divide_selfoverlapping(sample_ccw_self_inter)
    assert len(sub_polys) == 1
    assert_array_equal(sub_polys[0], sample_ccw_self_inter)


def test_overlap_self_intersect():
    sub_polys = divide_selfoverlapping(sample_cw_self_inter_overlap)
    assert len(sub_polys) == 1
    assert_array_equal(sub_polys[0], sample_cw_self_inter_overlap)

    sub_polys = divide_selfoverlapping(sample_ccw_self_inter_overlap)
    assert len(sub_polys) == 1
    assert_array_equal(sub_polys[0], sample_ccw_self_inter_overlap)

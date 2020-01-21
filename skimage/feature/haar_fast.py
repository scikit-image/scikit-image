# Copyright (c) 2020, Amazon.com, Inc
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
# * Neither the name of the copyright holders nor the names of any
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
#
# File name: _haar_fast.py
# Contributor: Amazon Lab126 Multimedia team
# Author: WeiChung Chang - cweichun@amazon.com
# Date created: 01/04/2020
#

import numpy as np
import bisect
import sys

FEATURE_TYPE = ('type-2-x', 'type-2-y',
                'type-3-x', 'type-3-y',
                'type-4')

FEATURE_PARAM_TABLE = {
    # feature  : (direction, ratio)
    'type-2-x' : ('h', 2),
    'type-3-x' : ('h', 3),
    'type-2-y' : ('v', 2),
    'type-3-y' : ('v', 3),
    # we assemble type-4 by type-2-x
    'type-4' : ('d', 2),
}


def pad_zeros_to_imgs(imgs):
    lengh = len(imgs.shape)
    if lengh == 3:  # "NHW"
        pad_width = ((0, 0), (1, 0), (1, 0))
    elif lengh == 4:  # 'NHWC'
        pad_width = ((0, 0), (1, 0), (1, 0), (0, 0))
    else:  # 'HW'
        if lengh != 2:
            raise RuntimeError('unsupported dim')
        pad_width = ((1, 0), (1, 0))
    return np.pad(array=imgs, pad_width=pad_width, mode='constant',
                  constant_values=0)


def integral_array_horizontal(imgs):
    lengh = len(imgs.shape)
    r = np.zeros(imgs.shape)
    if lengh == 3 or lengh == 4:  # "NHW" or "NHWC"
        r = np.cumsum(imgs, axis=2)
    else:  # 'HW'
        if lengh != 2:
            raise RuntimeError('unsupported dim')
        r = np.cumsum(imgs, axis=1)
    return r


def integral_array_vertical(imgs):
    lengh = len(imgs.shape)
    r = np.zeros(imgs.shape)
    if lengh == 3 or lengh == 4:  # "NHW" or "NHWC"
        r = np.cumsum(imgs, axis=1)
    else:  # 'HW'
        if lengh != 2:
            raise RuntimeError('unsupported dim')
        r = np.cumsum(imgs, axis=0)

    return r


def integral_array(imgs):
    lengh = len(imgs.shape)
    r = np.zeros(imgs.shape)
    if lengh == 3 or lengh == 4:  # "NHW" or "NHWC"
        r = np.cumsum(imgs, axis=1)
        r = np.cumsum(r,    axis=2)
    else:  # 'HW'
        if lengh != 2:
            raise RuntimeError('unsupported dim')
        r = np.cumsum(imgs, axis=0)
        r = np.cumsum(r,    axis=1)
    return r


class SegmentTreeNode():

    def __init__(self, label):
        self.__children = []
        self.__start = 0
        self.__length = 0
        self.__end = 0
        self.__id = label
        self.__parent = None
        self.__range = (-1, -1)

    def is_root(self):
        return (self.__parent is None)

    def is_leaf(self):
        return (len(self.__children) == 0)

    def is_segment_node(self):
        return (self.is_leaf() and self.__length != 0)

    def insert(self, node):
        if isinstance(node, SegmentTreeNode) is False:
            raise RuntimeError(
                '"%s" is not an instance of SegmentTreeNode' % type(node))
        if self.is_segment_node() is True:
            raise RuntimeError('Cannot insert to segment node')

        node.__parent = self
        if len(self.__children) > 0:
            node.__start = node.__end = self.__children[-1].__end
        else:
            node.__start = node.__end = 0
        node.__range = (node.__start, node.__end)
        self.__children.append(node)

    def set_length(self, length):
        if length <= 0:
            raise RuntimeError('length %d should > 0' % length)

        # update self
        self.__length = length
        self.__end = self.__start + self.__length
        self.__range = (self.__start, self.__end)

        # update ancestor
        now = self
        while now.__parent is not None:
            now = now.__parent
            now.__length += length
            now.__end = now.__start + now.__length
            now.__range = (now.__start, now.__end)

    def get_pattern_by_index(self, index: int):
        res = []
        offset = self.__lookup_recursive(index, res)
        dct = {item[0] :	 item[1] for item in res}
        return {'pattern' : dct, 'offset' : offset}

    def dump(self):
        print("".ljust(self.__get_level() * 4),
              'name = ', self.__id,
              " range = [%7d %7d) " % (self.__start, self.__end),
              ' , ', ' length = %7d ' % self.__length)

    def get_range(self):
        return self.__range

    def DFS(self):
        self.dump()
        for i in self.__children:
            i.DFS()

    def go_down(self):
        if self.is_leaf() is True:
            raise RuntimeError('We are in leaf node')
        return self.__children[-1]

    def find_node_by_id(self, id_):
        # level order search
        to_check = []
        to_check.append(self)
        while len(to_check) != 0:
            if to_check[0].__id == id_:
                return to_check[0]
            for c in to_check[0].__children:
                to_check.append(c)
            to_check.pop(0)

    def go_up(self):
        if self.is_root() is True:
            raise RuntimeError('We are in root node')
        return self.__parent

    def __lookup(self, index: int):
        keys = [child.__range for child in self.__children]
        pos = bisect.bisect_right(keys, (index, sys.maxsize)) - 1
        return self.__children[pos]

    def __lookup_recursive(self, index: int, res):
        res.append(self.__id)
        index = index - self.__range[0]
        if len(self.__children) != 0:  # non-leaf, do search recursively
            item = self.__lookup(index)
            return item.__lookup_recursive(index, res)
        else:
            return index

    def __get_level(self):
        level = 0
        now = self
        while now.__parent is not None:
            level += 1
            now = now.__parent
        return level


class HaarFeatureMap():
    LEVEL_TO_NAME = {
        0 : "root",
        1 : "base",
        2 : "major",
        3 : "assemble"
    }
    NAME_TO_LEVEL = {v: k for k, v in LEVEL_TO_NAME.items()}
    LEAF_NODE_LEVEL = NAME_TO_LEVEL["assemble"]
    STORE_RESULT = True

    def __init__(self, h, w):
        self.__stree = SegmentTreeNode(('root', (h, w)))
        self.__features = []
        self.__level = 0
        self.__node_ptr = self.__stree
        return

    def enter_next_level(self, pattern):
        if self.__level == self.LEAF_NODE_LEVEL:
            raise RuntimeError("We are at leaf now")

        self.__node_ptr.insert(
            SegmentTreeNode((self.LEVEL_TO_NAME[self.__level + 1], pattern)))
        self.__node_ptr = self.__node_ptr.go_down()
        self.__level += 1
        return

    def back_to_previous_level(self, pattern):
        if self.__level == 0:
            raise RuntimeError("We are at root now!")
        self.__node_ptr = self.__node_ptr.go_up()
        self.__level -= 1
        return

    # input fmap is in the shape of (N, FH, FW)
    def add_feature_map(self, fmap):
        if self.__level != self.LEAF_NODE_LEVEL:
            raise RuntimeError("we can only insert fmap at assemble layer")

        flattened = fmap.reshape(fmap.shape[0], -1)
        if self.STORE_RESULT is True:
            self.__features.append(flattened)
        self.__node_ptr.set_length(flattened.shape[-1])
        return

    def __cal_type_2_x_coord(coord, assem_pat):
        wu = assem_pat[1] // 2
        w = assem_pat[1]
        h = assem_pat[0]

        b1_lt = [coord[0], coord[1]]
        b1_rb = [coord[0] + h - 1, coord[1] + wu - 1]
        b2_lt = [coord[0], coord[1] + wu]
        b2_rb = [coord[0] + h - 1, coord[1] + w - 1]
        return [[b1_lt, b1_rb], [b2_lt, b2_rb]]

    def __cal_type_3_x_coord(coord, assem_pat):
        wu = assem_pat[1] // 3
        w = assem_pat[1]
        h = assem_pat[0]

        b1_lt = [coord[0], coord[1]]
        b1_rb = [coord[0] + h - 1, coord[1] + wu - 1]
        b2_lt = [coord[0], coord[1] + wu]
        b2_rb = [coord[0] + h - 1, coord[1] + (2 * wu) - 1]
        b3_lt = [coord[0], coord[1] + (2 * wu)]
        b3_rb = [coord[0] + h - 1, coord[1] + w - 1]
        return [[b1_lt, b1_rb], [b2_lt, b2_rb], [b3_lt, b3_rb]]

    def __cal_type_2_y_coord(coord, assem_pat):
        hu = assem_pat[0] // 2
        h = assem_pat[0]
        w = assem_pat[1]

        b1_lt = [coord[0], coord[1]]
        b1_rb = [coord[0] + hu - 1, coord[1] + w - 1]
        b2_lt = [coord[0] + hu, coord[1]]
        b2_rb = [coord[0] + h - 1, coord[1] + w - 1]
        return [[b1_lt, b1_rb], [b2_lt, b2_rb]]

    def __cal_type_3_y_coord(coord, assem_pat):
        hu = assem_pat[0] // 3
        h = assem_pat[0]
        w = assem_pat[1]

        b1_lt = [coord[0], coord[1]]
        b1_rb = [coord[0] + hu - 1, coord[1] + w - 1]
        b2_lt = [coord[0] + hu, coord[1]]
        b2_rb = [coord[0] + (2 * hu) - 1, coord[1] + w - 1]
        b3_lt = [coord[0] + (2 * hu), scoord[1]]
        b3_rb = [coord[0] + h - 1, coord[1] + w - 1]
        return [[b1_lt, b1_rb], [b2_lt, b2_rb], [b3_lt, b3_rb]]

    def __cal_type_4_coord(coord, assem_pat):
        wu = assem_pat[1] // 2
        hu = assem_pat[0] // 2
        w = assem_pat[1]
        h = assem_pat[0]

        b1_lt = [coord[0], coord[1]]
        b1_rb = [coord[0] + hu - 1, coord[1] + wu - 1]

        b2_lt = [coord[0], coord[1] + wu]
        b2_rb = [coord[0] + hu - 1, coord[1] + w - 1]

        b3_lt = [coord[0] + hu, coord[1]]
        b3_rb = [coord[0] + h - 1, coord[1] + wu - 1]

        b4_lt = [coord[0] + hu, coord[1] + wu]
        b4_rb = [coord[0] + h - 1, coord[1] + w - 1]
        return [[b1_lt, b1_rb], [b2_lt, b2_rb], [b4_lt, b4_rb], [b3_lt, b3_rb]]

    switch = {
        'type-2-x' : __cal_type_2_x_coord,
        'type-3-x' : __cal_type_3_x_coord,
        'type-2-y' : __cal_type_2_y_coord,
        'type-3-y' : __cal_type_3_y_coord,
        'type-4'   : __cal_type_4_coord
    }

    def get_feature(self):
        if isinstance(self.__features, list) is True:
            self.finalize()
        return self.__features

    def query_by_index(self, index):
        if isinstance(self.__features, list) is True:
            self.finalize()
        return self.__features[:, index]

    def query_by_type(self, type_):
        if isinstance(self.__features, list) is True:
            self.finalize()
        node = self.__stree.find_node_by_id(('base', type_))
        rg = node.get_range()
        return self.__features[:, rg[0]:rg[1]]

    def to_feature_coord(self, index):
        if isinstance(self.__features, list) is True:
            self.finalize()
        result = self.__stree.get_pattern_by_index(index)

        offset = result['offset']
        pattern = result['pattern']

        h = pattern['root'][0]
        w = pattern['root'][1]
        ah = pattern['assemble'][0]
        aw = pattern['assemble'][1]
        fw = w - aw + 1
        fh = h - ah + 1

        coord = (offset // fw, offset % fw)

        output = np.empty((1,), dtype=object)
        feature_coord = \
            self.switch[pattern['base']](coord, pattern['assemble'])
        output[0] = feature_coord

        return output

    def to_feature_type(self, index):
        result = self.__stree.get_pattern_by_index(index)
        pattern = result['pattern']
        return pattern['base']

    def dump(self):
        self.__stree.DFS()

    def finalize(self):
        if len(self.__features) > 0:
            self.__features = np.hstack(self.__features)
        else:
            self.__features = None
        return self.__features


class HaarCalculator():
    switch = {
        'type-2-x' : (1, 2),
        'type-3-x' : (1, 3),
        'type-2-y' : (2, 1),
        'type-3-y' : (3, 1),
        'type-4'   : (1, 2)
    }

    def __init__(self, raw_imgs):
        if (len(raw_imgs.shape) != 3):
            raise ValueError('Images must be in the foramt of [N, H, W]!')
        self.raw_images = raw_imgs
        self.segmented_array = \
            HaarFeatureMap(raw_imgs.shape[1], raw_imgs.shape[2])
        self.__h = raw_imgs.shape[1]
        self.__w = raw_imgs.shape[2]
        self.__pad_imgs = pad_zeros_to_imgs(raw_imgs)  # (N, H, W)
        self.__pad_int_image_h = integral_array_horizontal(self.__pad_imgs)
        self.__pad_int_image_v = integral_array_vertical(self.__pad_imgs)
        self.__pad_h = self.__pad_imgs.shape[1]
        self.__pad_w = self.__pad_imgs.shape[2]

    def generate_major_pattern(self, type_):
        if type_ not in self.switch:
            raise RuntimeError('unknown feature type "%s"' % type_)
        fh = self.switch[type_][0]
        fw = self.switch[type_][1]

        if self.switch[type_][0] != 1:
            h_dim = (np.arange((self.__h // fh) + 1)[1:]) * fh
        else:
            h_dim = np.array([1])

        if self.switch[type_][1] != 1:
            w_dim = (np.arange((self.__w // fw) + 1)[1:]) * fw
        else:
            w_dim = np.array([1])

        major_pattern = \
            np.transpose(
               [np.tile(h_dim, len(w_dim)),
                np.repeat(w_dim, len(h_dim))])
        return major_pattern

    def caculate_all_fmap(self):
        for ftype in FEATURE_TYPE:
            self.__caculate_base_pattern_fmap(ftype)
        return

    def caculate_fmap_from_type(self, feature_type):
        self.__caculate_base_pattern_fmap(feature_type)
        return

    def decorator(f):
        def __wrapper(*args, **kwargs):
            self = args[0]
            self.segmented_array.enter_next_level(args[-1])
            result = f(*args, **kwargs)
            self.segmented_array.back_to_previous_level(args[-1])
            return result
        return __wrapper

    @decorator
    def __caculate_base_pattern_fmap(self, ftype):
        major_patterns = self.generate_major_pattern(ftype)
        for major_pattern in major_patterns:
            self.__caculate_major_pattern_fmap(ftype, tuple(major_pattern))
        return

    @decorator
    def __caculate_major_pattern_fmap(self, ftype, major_pattern):
        direction, base_length = FEATURE_PARAM_TABLE[ftype]
        major_fmap = self.__caculate_major_pattern_fmap_worker(
            ftype, tuple(major_pattern))
        self.__assemble_fmap(major_fmap, direction, major_pattern)
        return

    @decorator
    def _update_assemble_fmap(self, fmap, assemble_pattern):
        self.segmented_array.add_feature_map(fmap)
        return

    def __caculate_major_pattern_fmap_worker(self, ftype, major_pattern):
        direction, base_length = FEATURE_PARAM_TABLE[ftype]

        if direction == 'h' or direction == 'd':
            basic_dim = (1, major_pattern[1] // base_length)
        else:
            basic_dim = (major_pattern[0] // base_length, 1)

        h_l = self.__pad_h - basic_dim[0]
        w_l = self.__pad_w - basic_dim[1]
        if direction == 'h' or direction == 'd':
            basic_fmap =
            self.__pad_int_image_h[:, 1:, basic_dim[1]:] -
            self.__pad_int_image_h[:, 1:, 0:w_l]
        else:
            basic_fmap =
            self.__pad_int_image_v[:, basic_dim[0]:, 1:] -
            self.__pad_int_image_v[:, 0:h_l, 1:]

        if base_length == 2:
            h_l = basic_fmap.shape[1] - basic_dim[0]
            w_l = basic_fmap.shape[2] - basic_dim[1]
        else:
            h_l = basic_fmap.shape[1] - (2 * basic_dim[0])
            w_l = basic_fmap.shape[2] - (2 * basic_dim[1])

        if direction == 'h' or direction == 'd':
            if base_length == 2:
                fmap = basic_fmap[:, :, basic_dim[1]:] -
                basic_fmap[:, :, 0:w_l]
            else:
                fmap = basic_fmap[:, :, basic_dim[1]*2:] -
                (basic_fmap[:, :, basic_dim[1]:basic_dim[1]+w_l]) +
                basic_fmap[:, :, 0:w_l]
                fmap = fmap * -1
        else:
            if base_length == 2:
                fmap = basic_fmap[:, basic_dim[0]:, :] -
                basic_fmap[:, 0:h_l, :]
            else:
                fmap = basic_fmap[:, basic_dim[0]*2:, :] -
                (basic_fmap[:, basic_dim[0]:basic_dim[0]+h_l, :]) +
                basic_fmap[:, 0:h_l, :]
                fmap = fmap * -1
        return fmap

    def __assemble_fmap(self, major_fmap, dir_, major_pattern):
        n_ = major_fmap.shape[0]
        h_ = major_fmap.shape[1]
        w_ = major_fmap.shape[2]

        if dir_ == 'v':
            self._update_assemble_fmap(major_fmap, (major_pattern[0], 1))

            minor_dim = w_
            if minor_dim < 2:
                return
            last_fmap = major_fmap
            for md in range(2, minor_dim+1):
                shift = md - 1
                len_ = w_ - shift
                now = major_fmap[:, :, shift:] + last_fmap[:, :, :len_]
                last_fmap = now

                self._update_assemble_fmap(now, (major_pattern[0], md))
        elif dir_ == 'h':
            self._update_assemble_fmap(major_fmap, (1, major_pattern[1]))

            minor_dim = h_
            if minor_dim < 2:
                return
            last_fmap = major_fmap
            for md in range(2, minor_dim+1):
                shift = md - 1
                len_ = h_ - shift
                now = major_fmap[:, shift:, :] + last_fmap[:, :len_, :]
                last_fmap = now

                self._update_assemble_fmap(now, (md, major_pattern[1]))
        else:
            minor_dim = h_
            if minor_dim < 2:
                return
            last_fmap = major_fmap

            shift = 1
            for md in range(1, (minor_dim//2)+1):
                shift = md
                base_len = h_ - shift
                len_ = last_fmap.shape[1] - shift
                now = last_fmap[:, :len_, :] - last_fmap[:, shift:, :]

                last_fmap = last_fmap[:, :base_len, :] +
                major_fmap[:, shift:, :]
                self._update_assemble_fmap(now, (md * 2, major_pattern[1]))
        return

    def dump(self):
        self.segmented_array.dump()
        r = self.segmented_array.finalize()

    def to_feature_coord(self, index):
        return self.segmented_array.to_feature_coord(index)


class DirectHaarCalculator():
    switch = {
        'type-2-x' : (1, 2),
        'type-3-x' : (1, 3),
        'type-2-y' : (2, 1),
        'type-3-y' : (3, 1),
        'type-4'   : (1, 2)
    }

    def __init__(self, raw_imgs):
        if (len(raw_imgs.shape) != 3):
            raw_imgs = raw_imgs.reshape(
                1, raw_imgs.shape[0], raw_imgs.shape[1])
        self.raw_images = raw_imgs

        self.__h = raw_imgs.shape[1]
        self.__w = raw_imgs.shape[2]
        self.__pad_imgs = pad_zeros_to_imgs(raw_imgs)
        self.__int_imgs = integral_array(self.__pad_imgs)
        self.__pad_h = self.__pad_imgs.shape[1]
        self.__pad_w = self.__pad_imgs.shape[2]
        return

    def __sanity_check(self, type_, pattern):
        if ((type_ == 'type-3-y' and (pattern[0] % 3 != 0)) or
                (type_ == 'type-3-x' and (pattern[1] % 3 != 0)) or
                (type_ == 'type-2-y' and (pattern[0] % 2 != 0)) or
                (type_ == 'type-4' and (pattern[0] % 2 != 0)) or
                (type_ == 'type-2-x' and (pattern[1] % 2 != 0)) or
                (type_ == 'type-4' and (pattern[1] % 2 != 0))):
            raise ValueError('input pattern error', pattern)

    def calculate_assemble_pattern(self, type_, pattern):
        self.__sanity_check(type_, pattern)
        if pattern[0] > self.__h or pattern[1] > self.__w:
            return None
        if type_ == 'type-2-y':
            res = self.__calculate_worker(
                self.__vertical2_pattern_to_coordinate(pattern))
        elif type_ == 'type-2-x':
            res = self.__calculate_worker(
                self.__horizontal2_pattern_to_coordinate(pattern))
        elif type_ == 'type-3-y':
            res = self.__calculate_worker(
                self.__vertical3_pattern_to_coordinate(pattern))
        elif type_ == 'type-3-x':
            res = self.__calculate_worker(
                self.__horizontal3_pattern_to_coordinate(pattern))
        elif type_ == 'type-4':
            pattern = (pattern[0] // 2, pattern[1])
            res = self.__calculate_worker(
                self.__horizontal2_pattern_to_coordinate(pattern))
            res = self.__gen_diagonal(res, pattern)
        return res

    def __calculate_worker(self, params):
        coords = params[0]
        range_ = params[1]
        res = np.zeros((self.__int_imgs.shape[0], range_[0], range_[1]))

        for item in coords:
            res += (item[0] *
                    self.__int_imgs[:, item[1]:item[2], item[3]:item[4]])
        return res

    def __vertical2_pattern_to_coordinate(self, pattern):
        dim_h = pattern[0]
        dim_w = pattern[1]
        hh = dim_h//2
        range_h = self.__pad_h - dim_h
        range_w = self.__pad_w - dim_w

        p_rb = (1, dim_h, self.__pad_h, dim_w, self.__pad_w)
        p_rt = (-2, hh, (hh + range_h), dim_w, self.__pad_w)
        p_lb = (-1, dim_h, self.__pad_h, 0, range_w)
        p_lt = (2, hh, (hh + range_h), 0, range_w)

        n_rt = (1, 0, range_h, dim_w, self.__pad_w)
        n_lt = (-1, 0, range_h, 0, range_w)

        return [p_rb, p_rt, p_lb, p_lt, n_rt, n_lt], (range_h, range_w)

    def __horizontal2_pattern_to_coordinate(self, pattern):
        dim_h = pattern[0]
        dim_w = pattern[1]
        hw = dim_w//2
        range_h = self.__pad_h - dim_h
        range_w = self.__pad_w - dim_w

        p_rb = (1, dim_h, self.__pad_h, dim_w, self.__pad_w)
        p_rt = (-1, 0, range_h, dim_w, self.__pad_w)
        p_lb = (-2, dim_h, self.__pad_h, hw, (hw + range_w))
        p_lt = (2, 0, range_h, hw, (hw + range_w))

        n_lb = (-1, 0, range_h, 0, range_w)
        n_lt = (1, dim_h, self.__pad_h, 0, range_w)

        return [p_rb, p_rt, p_lb, p_lt, n_lb, n_lt], (range_h, range_w)

    def __vertical3_pattern_to_coordinate(self, pattern):
        dim_h = pattern[0]
        dim_w = pattern[1]
        th = dim_h//3
        range_h = self.__pad_h - dim_h
        range_w = self.__pad_w - dim_w

        p_rb_b = (-1, dim_h, self.__pad_h,   dim_w, self.__pad_w)
        p_rt_b = (2, th*2, (th*2+range_h), dim_w, self.__pad_w)
        p_lb_b = (1, dim_h, self.__pad_h, 0, range_w)
        p_lt_b = (-2, th*2,  (th*2+range_h), 0, range_w)

        n_rt = (-2, th, (th+range_h), dim_w, self.__pad_w)
        n_lt = (2, th, (th+range_h), 0, range_w)

        p_rt_t = (1, 0, range_h, dim_w, self.__pad_w)
        p_lt_t = (-1, 0, range_h, 0, range_w)

        return [p_rb_b, p_rt_b, p_lb_b, p_lt_b, n_rt, n_lt, p_rt_t, p_lt_t],
        (range_h, range_w)

    def __horizontal3_pattern_to_coordinate(self, pattern):
        dim_h = pattern[0]
        dim_w = pattern[1]
        tw = dim_w//3
        range_h = self.__pad_h - dim_h
        range_w = self.__pad_w - dim_w

        p_rb_r = (-1, dim_h, self.__pad_h, dim_w, self.__pad_w)
        p_rt_r = (1, 0, range_h, dim_w, self.__pad_w)
        p_lb_r = (2, dim_h, self.__pad_h, (tw * 2), (tw * 2 + range_w))
        p_lt_r = (-2, 0, range_h, (tw * 2), (tw * 2 + range_w))

        n_lb = (-2, dim_h, self.__pad_h, tw, (tw + range_w))
        n_lt = (2, 0, range_h, tw, (tw + range_w))

        p_lt_l = (-1, 0, range_h, 0, range_w)
        p_lb_l = (1, dim_h, self.__pad_h, 0, range_w)

        return [p_rb_r, p_rt_r, p_lb_r, p_lt_r, n_lb, n_lt, p_lt_l, p_lb_l],
        (range_h, range_w)

    def __gen_diagonal(self, hfmap, pattern):
        fh = pattern[0]
        fw = pattern[1]
        d = hfmap.shape[1] - fh
        fdig = []
        if (d > 0):
            fdig = (hfmap[:, 0:d, :] - hfmap[:, fh:, :])
            fdig = (hfmap[:, 0:d, :] -
                    hfmap[:, fh:, :]).reshape(self.__pad_imgs.shape[0], -1)
        return fdig

    def __get_all_assemble_pattern(self, type_):
        if type_ not in self.switch:
            raise RuntimeError('unknown feature type "%s"' % type_)
        fh = self.switch[type_][0]
        fw = self.switch[type_][1]

        if self.switch[type_][0] != 1:
            h_dim = (np.arange((self.__h // fh) + 1)[1:]) * fh
        else:
            h_dim = np.array(range(1, self.__h + 1))

        if self.switch[type_][1] != 1:
            w_dim = (np.arange((self.__w // fw) + 1)[1:]) * fw
        else:
            w_dim = np.array(range(1, self.__w + 1))

        if type_ == 'type-4':
            h_dim = h_dim[1::2]
        patterns = np.transpose(
            [np.tile(h_dim, len(w_dim)), np.repeat(w_dim, len(h_dim))])
        return patterns

    def calculate_pattern_by_type(self, type_, flatten=False):
        res = []
        ps = self.__get_all_assemble_pattern(type_)
        for p in ps:
            r = self.calculate_assemble_pattern(type_, p)
            if flatten is True:
                r = r.reshape(r.shape[0], -1)
            res.append(r)
        if flatten is True:
            res = np.hstack(res)
        return res

    def calculate_all_pattern(self, flatten=False):
        res = []
        for type_, _ in self.switch.items():
            ps = self.__get_all_assemble_pattern(type_)
            for p in ps:
                r = self.calculate_assemble_pattern(type_, p)
                if flatten is True:
                    r = r.reshape(r.shape[0], -1)
                res.append(r)
        if flatten is True:
            res = np.hstack(res)
        return res


def haar_like_feature_fast(images, feature_type=None, pattern=None):
    if pattern is None:
        calculator = HaarCalculator(images)
        if feature_type is None:
            calculator.caculate_all_fmap()
        else:
            for type_ in feature_type:
                calculator.caculate_fmap_from_type(type_)
        result = calculator.segmented_array.get_feature()
        return result
    else:
        calculator = DirectHaarCalculator(images)
        result = calculator.calculate(feature_type, pattern)
    return result

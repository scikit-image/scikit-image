#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for novice submoudle.

Authors
-------
- tests were written by Michael Hansen, 2013

:license: modified BSD
"""

import os
from numpy.testing import TestCase, assert_equal, assert_raises
from skimage import novice, data_dir

class TestNovice(TestCase):
    sample_path = os.path.join(data_dir, 'elephant.png')
    small_sample_path = os.path.join(data_dir, 'block.png')

    def test_pic_info(self):
        pic = novice.open(self.sample_path)
        assert_equal(pic.format, 'PNG')
        assert_equal(pic.path, os.path.abspath(self.sample_path))
        assert_equal(pic.size, (665, 500))
        assert_equal(pic.width, 665)
        assert_equal(pic.height, 500)
        assert_equal(pic.modified, False)
        assert_equal(pic.inflation, 1)

        num_pixels = sum(1 for p in pic)
        assert_equal(num_pixels, pic.width * pic.height)

    def test_modify(self):
        pic = novice.open(self.small_sample_path)
        assert_equal(pic.modified, False)

        for p in pic:
            if p.x < (pic.width / 2):
                p.red /= 2
                p.green /= 2
                p.blue /= 2

        for p in pic:
            if p.x < (pic.width / 2):
                assert_equal(p.red <= 128, True)
                assert_equal(p.green <= 128, True)
                assert_equal(p.blue <= 128, True)

        s = pic.size
        pic.size = (pic.width / 2, pic.height / 2)
        assert_equal(pic.size, (s[0] / 2, s[1] / 2))

        assert_equal(pic.modified, True)
        assert_equal(pic.path, None)

        mod_path = "{0}.jpg".format(os.path.splitext(self.sample_path)[0])
        pic.save(mod_path)

        assert_equal(pic.modified, False)
        assert_equal(pic.path, os.path.abspath(mod_path))
        assert_equal(pic.format, "JPEG")
        os.unlink(mod_path)

    def test_indexing(self):
        pic = novice.open(self.small_sample_path)

        # Slicing
        pic[0:5, 0:5] = (0, 0, 0)
        for p in pic:
            if (p.x < 5) and (p.y < 5):
                assert_equal(p.rgb, (0, 0, 0))
                assert_equal(p.red, 0)
                assert_equal(p.green, 0)
                assert_equal(p.blue, 0)

        # Outside bounds
        assert_raises(IndexError, lambda: pic[pic.width, pic.height])

        # Negative indexing
        pic[-1, -1] = (0, 0, 0)
        assert_equal(pic[pic.width - 1, pic.height - 1].rgb, (0, 0, 0))

        # Stepping (checkerboard)
        pic[:, :] = (0, 0, 0)
        pic[::2, ::2] = (255, 255, 255)

        for p in pic:
            if (p.x % 2 == 0) and (p.y % 2 == 0):
                assert_equal(p.rgb, (255, 255, 255))
            else:
                assert_equal(p.rgb, (0, 0, 0))

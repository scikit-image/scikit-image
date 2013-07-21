#!/usr/bin/env python
# encoding: utf-8
# filename: profile.py

import pstats, cProfile

from test import start

cProfile.runctx("start()", globals(), locals(), "Profile.prof")

s = pstats.Stats("Profile.prof")
s.strip_dirs().sort_stats("time").print_stats(20)

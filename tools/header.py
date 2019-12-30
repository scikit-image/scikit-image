#!/usr/bin/env python

import sys

screen_width = 50

print('*' * screen_width)

if len(sys.argv) > 1:
    header = ' '.join(sys.argv[1:])
    header = header.replace('.', ' ')
    print('*', header.center(screen_width - 4), '*')
    print('*' * screen_width)


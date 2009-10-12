#!/bin/bash

cython opencv_backend.pyx && cython opencv_cv.pyx && gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing -I/usr/include/python2.6 -o opencv_backend.so opencv_backend.c && gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing -I/usr/include/python2.6 -o opencv_cv.so opencv_cv.c

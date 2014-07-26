#!/usr/bin/env python
from __future__ import print_function
from skimage import io

for key in (io.manage_plugins.plugin_store):
    print(key,"->",io.manage_plugins.plugin_store[key])
    
io.use_plugin('freeimage')

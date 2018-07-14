This directory contains meta-files related to the Lewiner marching cubes
algorithm. These are not used by the algorithm, but can be convenient
for development/maintenance:
    
* MarchingCubes.cpp - the original algorithm, this is ported to Cython
* LookupTable.h - the original LUTs, these are ported to Python
* createluts.py - scrip to generate Python luts from the .h file
* visual_test.py - script to compare visual results of marchingcubes algorithms

import sys
import base64

import numpy as np

if sys.version_info[0] == 3:
    base64decode = base64.decodebytes
else:
    base64decode = base64.decodestring

from . import _marching_cubes_lewiner_luts as mcluts
from . import _marching_cubes_lewiner_cy



def marching_cubes_lewiner(volume, level=None, spacing=(1., 1., 1.),
                           step_size=1, use_classic=False):
    """
    Lewiner marching cubes algorithm to find surfaces in 3d volumetric data
    
    In contrast to ``marching_cubes()``, this algorithm resolves
    ambiguities and guarantees topologically correct results.
    
    Parameters
    ----------
    volume : (M, N, P) array (the data is internally converted to
        float32 if necessary)
    level : float
        Contour value to search for isosurfaces in `volume`. If not
        given or None, the average of the min and max of vol is used.
    spacing : length-3 tuple of floats
        Voxel spacing in spatial dimensions corresponding to numpy array
        indexing dimensions (M, N, P) as in `volume`.
    step_size : int
        Step size in voxels. Default 1. Larger steps yield faster but
        coarser results. The result will always be topologically correct
        though.
    use_classic : bool
        If given and True, the classic marching cubes by Lorensen (1987)
        is used. This option is included for reference purposes. Note
        that this algorithm has ambiguities and is not guaranteed to
        produce a topologically correct result.
    
    Returns
    -------
    verts : (V, 3) array
        Spatial coordinates for V unique mesh vertices. Coordinate order
        matches input `volume` (M, N, P).
    faces : (F, 3) array
        Define triangular faces via referencing vertex indices from ``verts``.
        This algorithm specifically outputs triangles, so each face has
        exactly three indices.
    normals : (V, 3) array
        The normal direction at each vertex, as calculated from the
        data.
    values : (V, ) array
        Gives a measure for the maximum value of the data in the local region
        near each vertex. This can be used by visualization tools to apply
        a colormap to the mesh.
    
    Notes about the algorithm
    -------------------------
    
    This is an implementation of:
        
        Efficient implementation of Marching Cubes' cases with
        topological guarantees. Thomas Lewiner, Helio Lopes, Antonio
        Wilson Vieira and Geovan Tavares. Journal of Graphics Tools
        8(2): pp. 1-15 (december 2003)
    
    The algorithm is an improved version of Chernyaev's Marching Cubes 33
    algorithm, originally written in C++. It is an efficient algorithm
    that relies on heavy use of lookup tables to handle the many different 
    cases. This keeps the algorithm relatively easy. The current algorithm
    is a port of Lewiner's algorithm and written in Cython.
    """ 
    
    # Check volume and ensure its in the format that the alg needs
    if not isinstance(volume, np.ndarray) or (volume.ndim != 3):
        raise ValueError('Input volume should be a 3D numpy array.')
    if volume.shape[0] < 2 or volume.shape[1] < 2 or volume.shape[2] < 2:
        raise ValueError("Input array must be at least 2x2x2.")
    volume = np.array(volume, dtype=np.float32, order="C", copy=False)
    
    # Check/convert other inputs:
    # level
    if level is None:
        level = 0.5 * (volume.min() + volume.max())
    else:
        level = float(level)
        if level < volume.min() or level > volume.max():
            raise ValueError("Surface level must be within volume data range.")
    # spacing
    if len(spacing) != 3:
        raise ValueError("`spacing` must consist of three floats.")
    # step_size
    step_size = int(step_size)
    if step_size < 1:
        raise ValueError('step_size must be at least one.')
    # use_classic
    use_classic = bool(use_classic)
    
    # Get LutProvider class (reuse if possible) 
    L = _getMCLuts()
    
    # Apply algorithm
    func = _marching_cubes_lewiner_cy.marching_cubes
    vertices, faces , normals, values = func(volume, level, L, step_size, use_classic)
    
    if not len(vertices):
        raise RuntimeError('No surface found at the given iso value.')
    
    # Output in z-y-x order, as is common in skimage
    vertices = np.fliplr(vertices)
    normals = np.fliplr(normals)
    
    # Finishing touches to output
    faces.shape = -1, 3
    if spacing != (1, 1, 1):
        vertices = vertices * np.r_[spacing]
    
    return vertices, faces, normals, values


def _toArray(args):
    shape, text = args
    byts = base64decode(text.encode('utf-8'))
    ar = np.frombuffer(byts, dtype='int8')
    ar.shape = shape
    return ar


# Map an edge-index to two relative pixel positions. The ege index 
# represents a point that lies somewhere in between these pixels.
# Linear interpolation should be used to determine where it is exactly.
#   0
# 3   1   ->  0x
#   2         xx
EDGETORELATIVEPOSX = np.array([ [0,1],[1,1],[1,0],[0,0], [0,1],[1,1],[1,0],[0,0], [0,0],[1,1],[1,1],[0,0] ], 'int8')
EDGETORELATIVEPOSY = np.array([ [0,0],[0,1],[1,1],[1,0], [0,0],[0,1],[1,1],[1,0], [0,0],[0,0],[1,1],[1,1] ], 'int8')
EDGETORELATIVEPOSZ = np.array([ [0,0],[0,0],[0,0],[0,0], [1,1],[1,1],[1,1],[1,1], [0,1],[0,1],[0,1],[0,1] ], 'int8')


def _getMCLuts():
    """ Kind of lazy obtaining of the luts.
    """ 
    if not hasattr(mcluts, 'THE_LUTS'):
        
        mcluts.THE_LUTS = _marching_cubes_lewiner_cy.LutProvider(
                EDGETORELATIVEPOSX, EDGETORELATIVEPOSY, EDGETORELATIVEPOSZ, 
                
                _toArray(mcluts.CASESCLASSIC), _toArray(mcluts.CASES),
                
                _toArray(mcluts.TILING1), _toArray(mcluts.TILING2), _toArray(mcluts.TILING3_1), _toArray(mcluts.TILING3_2), 
                _toArray(mcluts.TILING4_1), _toArray(mcluts.TILING4_2), _toArray(mcluts.TILING5), _toArray(mcluts.TILING6_1_1),
                _toArray(mcluts.TILING6_1_2), _toArray(mcluts.TILING6_2), _toArray(mcluts.TILING7_1), 
                _toArray(mcluts.TILING7_2), _toArray(mcluts.TILING7_3), _toArray(mcluts.TILING7_4_1), 
                _toArray(mcluts.TILING7_4_2), _toArray(mcluts.TILING8), _toArray(mcluts.TILING9), 
                _toArray(mcluts.TILING10_1_1), _toArray(mcluts.TILING10_1_1_), _toArray(mcluts.TILING10_1_2), 
                _toArray(mcluts.TILING10_2), _toArray(mcluts.TILING10_2_), _toArray(mcluts.TILING11), 
                _toArray(mcluts.TILING12_1_1), _toArray(mcluts.TILING12_1_1_), _toArray(mcluts.TILING12_1_2), 
                _toArray(mcluts.TILING12_2), _toArray(mcluts.TILING12_2_), _toArray(mcluts.TILING13_1), 
                _toArray(mcluts.TILING13_1_), _toArray(mcluts.TILING13_2), _toArray(mcluts.TILING13_2_), 
                _toArray(mcluts.TILING13_3), _toArray(mcluts.TILING13_3_), _toArray(mcluts.TILING13_4), 
                _toArray(mcluts.TILING13_5_1), _toArray(mcluts.TILING13_5_2), _toArray(mcluts.TILING14),
                
                _toArray(mcluts.TEST3), _toArray(mcluts.TEST4), _toArray(mcluts.TEST6), 
                _toArray(mcluts.TEST7), _toArray(mcluts.TEST10), _toArray(mcluts.TEST12), 
                _toArray(mcluts.TEST13), _toArray(mcluts.SUBCONFIG13),
                )
    
    return mcluts.THE_LUTS


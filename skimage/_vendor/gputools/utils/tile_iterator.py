"""


mweigert@mpi-cbg.de

"""
from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
from itertools import product

def tile_iterator(im,
                 blocksize = (64, 64),
                 padsize = (64,64),
                 mode = "constant",
                 verbose = False):
    """

    iterates over padded tiles of an ND image 
    while keeping track of the slice positions

    Example:
    --------
    im = np.zeros((200,200))
    res = np.empty_like(im)

    for padded_tile, s_src, s_dest in tile_iterator(im,
                              blocksize=(128, 128),
                              padsize = (64,64),
                              mode = "wrap"):

        #do something with the tile, e.g. a convolution
        res_padded = np.mean(padded_tile)*np.ones_like(padded_tile)

        # reassemble the result at the correct position
        res[s_src] = res_padded[s_dest]



    Parameters
    ----------
    im: ndarray
        the input data (arbitrary dimension)
    blocksize:
        the dimension of the blocks to split into
        e.g. (nz, ny, nx) for a 3d image
    padsize:
        the size of left and right pad for each dimension
    mode:
        padding mode, like numpy.pad
        e.g. "wrap", "constant"...

    Returns
    -------
        tile, slice_src, slice_dest

        tile[slice_dest] is the tile in im[slice_src]

    """

    if not(im.ndim == len(blocksize) ==len(padsize)):
        raise ValueError("im.ndim (%s) != len(blocksize) (%s) != len(padsize) (%s)"
                         %(im.ndim , len(blocksize) , len(padsize)))

    subgrids = tuple([int(np.ceil(1.*n/b)) for n,b in zip(im.shape, blocksize)])


    #if the image dimension are not divible by the blocksize, pad it accordingly
    pad_mismatch = tuple([(s*b-n) for n,s, b in zip(im.shape,subgrids,blocksize)])

    if verbose:
        print("tile padding... ")

    im_pad = np.pad(im,[(p,p+pm) for pm,p in zip(pad_mismatch,padsize)], mode = mode)

    # iterates over cartesian product of subgrids
    for i,index in enumerate(product(*[range(sg) for sg in subgrids])):
        # dest[s_output] is where we will write to
        s_input = tuple([slice(i*b,(i+1)*b) for i,b in zip(index, blocksize)])

        s_output = tuple([slice(p,b+p-pm*(i==s-1)) for b,pm,p,i,s in zip(blocksize,pad_mismatch,padsize, index, subgrids)])

        s_padinput = tuple([slice(i*b,(i+1)*b+2*p) for i,b,p in zip(index, blocksize, padsize)])
        padded_block = im_pad[s_padinput]

        yield padded_block, s_input, s_output


if __name__ == '__main__':


    # simple test


    for n in [1,2,3]:
        print("n = %s"%n)
        im = np.random.uniform(-1,1,[103+13*_n for _n in range(n)])
        res = np.empty_like(im)

        for padded_tile, s_src, s_dest in tile_iterator(im,
                                  blocksize=(50,)*im.ndim,
                                  padsize = (64,)*im.ndim,
                                  mode = "wrap"):

            # reassemble the result at the correct position
            res[s_src] = padded_tile[s_dest]

        print("OK" if np.allclose(res, im) else "ERRRRRRRRRRROOOOOOOORRRRRR")


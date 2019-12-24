"""


mweigert@mpi-cbg.de

"""
from __future__ import print_function, unicode_literals, absolute_import, division
import sys
import numpy as np
from gputools import fft_plan, OCLArray, OCLImage, \
    fft, get_device, OCLProgram, pad_to_shape, tile_iterator

from gputools.utils.utils import _is_power2, next_power_of_2
from gputools.utils.tile_iterator import tile_iterator

from ._abspath import abspath

import logging
logger = logging.getLogger(__name__)

def convolve_spatial2(im, psfs,
                      mode = "constant",
                      grid_dim = None,
                      sub_blocks = None,
                      pad_factor = 2,
                      plan = None,
                      return_plan = False):
    """
    GPU accelerated spatial varying convolution of an 2d image with a
    (Gy,Gx) grid of psfs assumed to be equally spaced within the image
    the input image im is subdivided into (Gy,Gx) blocks, each block is
    convolved with the corresponding psf and linearly interpolated to give the
    final result

    The psfs can be given either in

    A) Stackmode
    psfs.shape =  (Gy,Gx, Hy, Hx)
    then psfs[j,i] is the psf at the center of each block (i,j) in the image

    B) Flatmode
    psfs.shape = im.shape
    then the psfs are assumed to be definied on the gridpoints of the images itself
    in this case grid_dim = (Gy,Gx) has to be given

    as of now each image dimension has to be divisible by the grid dim, i.e.
    ::
        Nx % Gx == 0
        Ny % Gy == 0

    GPU Memory consumption is of order 8*Nx*Ny
    If not enough GPU memory is available, consider using
    sub_blocks = (n,m)
    then the operation is carried out in a tiled fashion reducing
    memory consumption to 8*Nx*Ny*(1/n+2/Gx)*(1/m+2/Gy)

    Example
    -------


    im = np.zeros((128,128))
    im[::10,::10] = 1.

    # Stackmode
    psfs = np.ones((16,16,4,4))
    res = convolve_spatial2(im, psfs, mode = "wrap")

    # Flatmode
    _X,_Y = np.meshgrid(*(np.arange(128),)*2)
    psfs = np.clip(np.sin(2*np.pi*_X/8),0,1)*np.clip(np.cos(2*np.pi*_Y/8),0,1)
    res = convolve_spatial2(im, psfs, grid_dim = (16,16))


    Parameters
    ----------
    im: ndarray
        the image to convolve
    psfs: ndarray
        the (Gx,Gy) psf grid, either of shape (Gx,Gy, Hy, Hx) or im.shape

    mode: string, optional
        Padding mode. Can be "constant", "wrap", "edge", or "reflect".
    grid_dim: tuple, optional
        the (Gy,Gx) grid dimension, has to be provided if psfs.shape = im.shape

    sub_blocks: tuple, optional
        tiling mode, give e.g. (2,2) to sequentially operate on quadratnts
    pad_factor: int
        the factor of its size each block get tiled, use pad_factor=2 if the psfs
        are well localized, use pad_factor = 3 if not (e.g. if you experience blocking)_

    plan: fft_plan, optional
        when given use this as the fft plan
    return_plan: bool, optional
        return (res, plan) with plan being the fft plan for further use

    Returns
    -------
    res: ndarray
        the convolved image



    """

    ndim = im.ndim

    if ndim != 2:
        raise ValueError("wrong dimensions of input!")


    if grid_dim:
        if psfs.shape != im.shape:
            raise ValueError("if grid_dim is set, then im.shape = hs.shape !")
    else:
        if not psfs.ndim==2*ndim:
            raise ValueError("wrong dimensions of psf grid! (Gy,Gx,Ny,Nx)")

    if grid_dim:
        Gs = grid_dim
    else:
        Gs = psfs.shape[:ndim]

    if not np.all([n%g==0 for n,g in zip(im.shape,Gs)]):
        raise NotImplementedError("shape of image has to be divisible by Gx Gy  = %s shape mismatch"%(str(psfs.shape[:2])))

    if sub_blocks is None:
        return _convolve_spatial2(im,
                                  psfs,
                                  mode = mode,
                                  pad_factor = pad_factor,
                                  plan = plan,
                                  return_plan = return_plan,
                                  grid_dim = grid_dim)
    else:
        # cut the image into tile and operate on every of them
        N_sub = [n//s for n,s  in zip(im.shape,sub_blocks)]
        Nblocks = [n//g for n,g  in zip(im.shape,Gs)]
        Npads = [n*(s>1) for n,s  in zip(Nblocks, sub_blocks)]
        grid_dim_sub = [g//s+2*(s>1) for g,s   in zip(Gs, sub_blocks)]

        logger.debug(
            "N_sub: {}, Nblocks: {}, grid_dim_sub, {}, Npads, {}".format(
                N_sub, Nblocks, grid_dim_sub, Npads
            ))

        if grid_dim:
            res = np.empty(im.shape, np.float32)
            plan = None
            for (im_tile, im_s_src, im_s_dest), (hs_tile, hs_s_src, hs_s_dest)\
                in zip(tile_iterator(im,blocksize=N_sub,
                                     padsize=Npads,
                                     mode = mode),\
                    tile_iterator(psfs, blocksize=N_sub,
                                  padsize=Npads,
                                  mode = mode)):

                res_tile, plan = _convolve_spatial2(im_tile.copy(),
                                              hs_tile.copy(),
                                              mode = mode,
                                              pad_factor = pad_factor,
                                              return_plan=True,
                                              plan = plan,
                                              grid_dim = grid_dim_sub)


                res[im_s_src] = res_tile[im_s_dest]
            return res

        else:
            raise NotImplementedError()



def _convolve_spatial2(im, hs,
                      mode = "constant",
                      grid_dim = None,
                      pad_factor = 2,
                      plan = None,
                      return_plan = False):
    """
    spatial varying convolution of an 2d image with a 2d grid of psfs

    shape(im_ = (Ny,Nx)
    shape(hs) = (Gy,Gx, Hy,Hx)

    the input image im is subdivided into (Gy,Gx) blocks
    hs[j,i] is the psf at the center of each block (i,j)

    as of now each image dimension has to be divisible by the grid dim, i.e.
    Nx % Gx == 0
    Ny % Gy == 0

    mode can be:
    "constant" - assumed values to be zero
    "wrap" - periodic boundary condition
    "edge" - values repeat the value at the edge of the image
    "reflect" - values are reflected around the edge of the image
    """

    if grid_dim:
        Gs = tuple(grid_dim)
    else:
        Gs = hs.shape[:2]

    mode_str = {"constant":"CLK_ADDRESS_CLAMP",
                "wrap":"CLK_ADDRESS_REPEAT",
                "edge":"CLK_ADDRESS_CLAMP_TO_EDGE",
                "reflect":"CLK_ADDRESS_MIRRORED_REPEAT"}

    print(mode_str[mode])
    Ny, Nx = im.shape
    Gy, Gx = Gs

    # the size of each block within the grid
    Nblock_y, Nblock_x = Ny // Gy, Nx // Gx

    # the size of the overlapping patches with safety padding
    Npatch_x, Npatch_y = next_power_of_2(pad_factor * Nblock_x), next_power_of_2(pad_factor * Nblock_y)

    prog = OCLProgram(abspath("kernels/conv_spatial2.cl"),
                      build_options=["-D","ADDRESSMODE=%s"%mode_str[mode]])

    if plan is None:
        plan = fft_plan((Gy, Gx, Npatch_y,Npatch_x), axes = (-2,-1))

    x0s = Nblock_x*np.arange(Gx)
    y0s = Nblock_y*np.arange(Gy)

    patches_g = OCLArray.empty((Gy,Gx,Npatch_y,Npatch_x),np.complex64)

    #prepare psfs
    if grid_dim:
        h_g = OCLArray.zeros((Gy,Gx,Npatch_y,Npatch_x),np.complex64)
        tmp_g = OCLArray.from_array(hs.astype(np.float32, copy = False))
        for i,_x0 in enumerate(x0s):
            for j,_y0 in enumerate(y0s):
                prog.run_kernel("fill_psf_grid2",
                                (Nblock_x,Nblock_y),None,
                        tmp_g.data,
                        np.int32(Nx),
                        np.int32(i*Nblock_x),
                        np.int32(j*Nblock_y),
                        h_g.data,
                        np.int32(Npatch_x),
                        np.int32(Npatch_y),
                        np.int32(-Nblock_x//2+Npatch_x//2),
                        np.int32(-Nblock_y//2+Npatch_y//2),
                        np.int32(i*Npatch_x*Npatch_y+j*Gx*Npatch_x*Npatch_y)
                            )
    else:
        hs = np.fft.fftshift(pad_to_shape(hs,(Gy,Gx,Npatch_y,Npatch_x)),axes=(2,3))
        h_g = OCLArray.from_array(hs.astype(np.complex64))

    #prepare image
    im_g = OCLImage.from_array(im.astype(np.float32,copy=False))

    for i,_x0 in enumerate(x0s):
        for j,_y0 in enumerate(y0s):
            prog.run_kernel("fill_patch2",(Npatch_x,Npatch_y),None,
                    im_g,
                    np.int32(_x0+Nblock_x//2-Npatch_x//2),
                    np.int32(_y0+Nblock_y//2-Npatch_y//2),
                    patches_g.data,
                    np.int32(i*Npatch_x*Npatch_y+j*Gx*Npatch_x*Npatch_y))
    #return np.abs(patches_g.get())
    # convolution
    fft(patches_g,inplace=True,  plan = plan)
    fft(h_g,inplace=True,  plan = plan)
    prog.run_kernel("mult_inplace",(Npatch_x*Npatch_y*Gx*Gy,),None,
                    patches_g.data, h_g.data)
    fft(patches_g,inplace=True, inverse = True, plan = plan)

    logger.debug("Nblock_x: {}, Npatch_x: {}".format(Nblock_x, Npatch_x))
    #return np.abs(patches_g.get())
    #accumulate
    res_g = OCLArray.empty(im.shape,np.float32)

    for j in range(Gy+1):
        for i in range(Gx+1):
            prog.run_kernel("interpolate2",(Nblock_x,Nblock_y),None,
                            patches_g.data,res_g.data,
                            np.int32(i),np.int32(j),
                            np.int32(Gx),np.int32(Gy),
                            np.int32(Npatch_x),np.int32(Npatch_y))

    res = res_g.get()

    if return_plan:
        return res, plan
    else:
        return res



if __name__ == '__main__':


    im = np.zeros((128,128))
    im[::10,::10] = 1.
    psfs = np.ones((2, 2, 3, 3))
    #hs *= 1./np.sum(hs[0,0])

    out = convolve_spatial2(im, psfs, mode ="constant", pad_factor = 2)


"""


mweigert@mpi-cbg.de

"""
from __future__ import print_function, unicode_literals, absolute_import, division
import sys
import numpy as np
from gputools import fft_plan, OCLArray, OCLImage, fft, \
    get_device, OCLProgram, pad_to_shape, tile_iterator
from itertools import product
from gputools.utils.utils import _is_power2, next_power_of_2
from ._abspath import abspath


def convolve_spatial3(im, psfs,
                      mode="constant",
                      grid_dim=None,
                      sub_blocks=None,
                      pad_factor=2,
                      plan=None,
                      return_plan=False,
                      verbose=False):
    """
    GPU accelerated spatial varying convolution of an 3d image with a
    (Gz, Gy, Gx) grid of psfs assumed to be equally spaced within the image

    the input image im is subdivided into (Gz, Gy,Gx) blocks, each block is
    convolved with the corresponding psf and linearly interpolated to give the
    final result

    The psfs can be given either in

    A) Stackmode

    psfs.shape =  (Gz, Gy, Gx, Hz, Hy, Hx)
    then psfs[k,j,i] is the psf at the center of each block (i,j,k) in the image

    B) Flatmode

    psfs.shape = im.shape
    then the psfs are assumed to be definied on the gridpoints of the images itself
    in this case grid_dim = (Gz,Gy,Gx) has to be given

    as of now each image dimension has to be divisible by the grid dim, i.e.
    ::
        Nx % Gx == 0
        Ny % Gy == 0
        Nz % Gz == 0


    GPU Memory consumption is of order 8*Nx*Ny*Nz
    If not enough GPU memory is available, consider using
    sub_blocks = (n,m,l)
    then the operation is carried out in a tiled fashion reducing
    memory consumption to 8*Nx*Ny*(1/n+2/Gx)*(1/m+2/Gy)*(1/l+2/Gz)
    (so there is no much use if n>Gx/2...)
    Example
    -------


    im = np.zeros((64,64,64))
    im[::10,::10,::10] = 1.

    # Stackmode
    psfs = np.ones((8,8,8,4,4,4))
    res = convolve_spatial3(im, psfs, mode = "wrap")

    # Flatmode
    _Xs = np.meshgrid(*(np.arange(64),)*2)
    psfs = np.prod([np.clip(np.sin(2*np.pi*_X/8),0,1) for _X in _Xs],axis=0)
    res = convolve_spatial2(im, psfs, grid_dim = (16,16,16))


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

    if ndim != 3:
        raise ValueError("wrong dimensions of input!")

    if grid_dim:
        if psfs.shape != im.shape:
            raise ValueError("if grid_dim is set, then im.shape = hs.shape !")
    else:
        if not psfs.ndim == 2 * ndim:
            raise ValueError("wrong dimensions of psf grid! should be (Gz,Gy,Gx,Nz,Ny,Nx)")

    if grid_dim:
        Gs = grid_dim
    else:
        Gs = psfs.shape[:ndim]

    if not np.all([n % g == 0 for n, g in zip(im.shape, Gs)]):
        raise NotImplementedError(
            "shape of image has to be divisible by Gx Gy  = %s shape mismatch" % (str(psfs.shape[:2])))

    if sub_blocks == None:
        return _convolve_spatial3(im,
                                  psfs,
                                  mode=mode,
                                  pad_factor=pad_factor,
                                  plan=plan,
                                  return_plan=return_plan,
                                  grid_dim=grid_dim)
    else:
        if not np.all([g % n == 0 for n, g in zip(sub_blocks, Gs)]):
            raise ValueError("psf grid dimension has to be divisible corresponding n_blocks")

        N_sub = [n // s for n, s in zip(im.shape, sub_blocks)]
        Nblocks = [n // g for n, g in zip(im.shape, Gs)]
        Npads = [n * (s > 1) for n, s in zip(Nblocks, sub_blocks)]
        grid_dim_sub = [g // s + 2 * (s > 1) for g, s in zip(Gs, sub_blocks)]

        if grid_dim:
            res = np.empty(im.shape, np.float32)
            plan = None
            for i, ((im_tile, im_s_src, im_s_dest), (hs_tile, hs_s_src, hs_s_dest)) \
                    in enumerate(zip(tile_iterator(im, blocksize=N_sub,
                                                   padsize=Npads,
                                                   mode=mode,
                                                   verbose=verbose), \
                                     tile_iterator(psfs, blocksize=N_sub,
                                                   padsize=Npads,
                                                   mode=mode,
                                                   verbose=verbose
                                                   ))):

                if verbose:
                    print("convolve_spatial3 ... %s\t/ %s" % (i + 1, np.prod(sub_blocks)))

                res_tile, plan = _convolve_spatial3(im_tile.copy(),
                                                    hs_tile.copy(),
                                                    mode=mode,
                                                    pad_factor=pad_factor,
                                                    return_plan=True,
                                                    plan=plan,
                                                    grid_dim=grid_dim_sub)

                res[im_s_src] = res_tile[im_s_dest]
            return res

        else:
            raise NotImplementedError("sub_blocks only implemented for Flatmode")


def _convolve_spatial3(im, hs,
                       mode="constant",
                       grid_dim=None,
                       plan=None,
                       return_plan=False,
                       pad_factor=2):
    if im.ndim != 3:
        raise ValueError("wrong dimensions of input!")

    if not (hs.ndim == 6 or (hs.ndim == 3 and grid_dim)):
        raise ValueError("wrong dimensions of psf grid!")

    if grid_dim:
        if hs.shape != im.shape:
            raise ValueError("if grid_dim is set, then im.shape = hs.shape !")
        Gs = tuple(grid_dim)
    else:
        if not hs.ndim == 6:
            raise ValueError("wrong dimensions of psf grid! (Gy,Gx,Ny,Nx)")
        Gs = hs.shape[:3]

    if not np.all([n % g == 0 for n, g in zip(im.shape, Gs)]):
        raise NotImplementedError(
            "shape of image has to be divisible by Gx Gy  = %s shape mismatch" % (str(hs.shape[:2])))

    mode_str = {"constant":"CLK_ADDRESS_CLAMP",
                "wrap":"CLK_ADDRESS_REPEAT",
                "edge":"CLK_ADDRESS_CLAMP_TO_EDGE",
                "reflect":"CLK_ADDRESS_MIRRORED_REPEAT"}

    Ns = im.shape

    # the size of each block within the grid
    Nblocks = [n // g for n, g in zip(Ns, Gs)]

    # the size of the overlapping patches with safety padding
    Npatchs = tuple([next_power_of_2(pad_factor * nb) for nb in Nblocks])

    prog = OCLProgram(abspath("kernels/conv_spatial3.cl"),
                      build_options=["-D", "ADDRESSMODE=%s" % mode_str[mode]])

    if plan is None:
        plan = fft_plan(Gs + Npatchs, axes=(-3, -2, -1))

    Xs = [nb * np.arange(g) for nb, g in zip(Nblocks, Gs)]

    patches_g = OCLArray.empty(Gs + Npatchs, np.complex64)

    # prepare psfs
    if grid_dim:
        h_g = OCLArray.zeros(Gs + Npatchs, np.complex64)

        tmp_g = OCLArray.from_array(hs.astype(np.float32, copy=False))
        for (k, _z0), (j, _y0), (i, _x0) in product(*[enumerate(X) for X in Xs]):

            prog.run_kernel("fill_psf_grid3",
                            Nblocks[::-1], None,
                            tmp_g.data,
                            np.int32(im.shape[2]),
                            np.int32(im.shape[1]),
                            np.int32(i * Nblocks[2]),
                            np.int32(j * Nblocks[1]),
                            np.int32(k * Nblocks[0]),
                            h_g.data,
                            np.int32(Npatchs[2]),
                            np.int32(Npatchs[1]),
                            np.int32(Npatchs[0]),
                            np.int32(-Nblocks[2] // 2 + Npatchs[2] // 2),
                            np.int32(-Nblocks[1] // 2 + Npatchs[1] // 2),
                            np.int32(-Nblocks[0] // 2 + Npatchs[0] // 2),
                            np.int32(i * np.prod(Npatchs) +
                                     j * Gs[2] * np.prod(Npatchs) +
                                     k * Gs[2] * Gs[1] * np.prod(Npatchs)))

    else:
        hs = np.fft.fftshift(pad_to_shape(hs, Gs + Npatchs), axes=(3, 4, 5))
        h_g = OCLArray.from_array(hs.astype(np.complex64))

    im_g = OCLImage.from_array(im.astype(np.float32, copy=False))

    # this loops over all i,j,k
    for (k, _z0), (j, _y0), (i, _x0) in product(*[enumerate(X) for X in Xs]):
        prog.run_kernel("fill_patch3", Npatchs[::-1], None,
                        im_g,
                        np.int32(_x0 + Nblocks[2] // 2 - Npatchs[2] // 2),
                        np.int32(_y0 + Nblocks[1] // 2 - Npatchs[1] // 2),
                        np.int32(_z0 + Nblocks[0] // 2 - Npatchs[0] // 2),
                        patches_g.data,
                        np.int32(i * np.prod(Npatchs) +
                                 j * Gs[2] * np.prod(Npatchs) +
                                 k * Gs[2] * Gs[1] * np.prod(Npatchs)))

    # convolution
    fft(patches_g, inplace=True, plan=plan)
    fft(h_g, inplace=True, plan=plan)
    prog.run_kernel("mult_inplace", (np.prod(Npatchs) * np.prod(Gs),), None,
                    patches_g.data, h_g.data)

    fft(patches_g,
        inplace=True,
        inverse=True,
        plan=plan)

    # return patches_g.get()
    # accumulate
    res_g = OCLArray.zeros(im.shape, np.float32)

    for k, j, i in product(*[list(range(g + 1)) for g in Gs]):
        prog.run_kernel("interpolate3", Nblocks[::-1], None,
                        patches_g.data,
                        res_g.data,
                        np.int32(i), np.int32(j), np.int32(k),
                        np.int32(Gs[2]), np.int32(Gs[1]), np.int32(Gs[0]),
                        np.int32(Npatchs[2]), np.int32(Npatchs[1]), np.int32(Npatchs[0]))

    res = res_g.get()

    if return_plan:
        return res, plan
    else:
        return res


if __name__ == '__main__':
    im = np.zeros((64, 64, 64))
    im[::10, ::10, ::10] = 1.
    hs = np.ones((4, 4, 4, 10, 10, 10))
    hs = np.ones_like(im)

    out = convolve_spatial3(im, hs, mode="constant", grid_dim=(4, 4, 4), sub_blocks=(2, 2, 2))

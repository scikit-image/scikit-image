"""
spatially varying convolutions


mweigert@mpi-cbg.de

"""

from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
from gputools import fft_plan, OCLArray, OCLImage, fft, get_device, OCLProgram, pad_to_shape
from gputools.utils.utils import _is_power2, next_power_of_2
from ._abspath import abspath
from itertools import product

#
# def convolve_spatial2(im, hs, plane = None, return_plan = False):
#     """
#     spatial varying convolution of an 2d image with a 2d grid of psfs
#
#     shape(im_ = (Ny,Nx)
#     shape(hs) = (Gy,Gx,Hy,Hx)
#
#     the psfs are assumed to be defined equally spaced
#     i.e. hs[0,0] is at (0,0) and hs[-1,-1] at (Ny-1,Nx-1)
#     """
#
#     if not np.all([n%(g-1)==0 for n,g in zip(im.shape,hs.shape[:2])]):
#         raise NotImplementedError("Gx Gy  = %s shape mismatch (all dimensions have to be divisible by G+1"%(str(hs.shape)))
#
#     Ny, Nx = im.shape
#     Gy, Gx = hs.shape[:2]
#
#     # the size of each block within the grid
#     Nblock_y, Nblock_x = Ny/(Gy-1), Nx/(Gx-1)
#
#
#     # the size of the overlapping patches with safety padding
#     Npatch_x, Npatch_y = _next_power_of_2(3*Nblock_x), _next_power_of_2(3*Nblock_y)
#     Npatch_x, Npatch_y = _next_power_of_2(2*Nblock_x), _next_power_of_2(2*Nblock_y)
#
#
#     hs = np.fft.fftshift(pad_to_shape(hs,(Gy,Gx,Npatch_y,Npatch_x)),axes=(2,3))
#
#     prog = OCLProgram(abspath("kernels/conv_spatial.cl"))
#
#     if plane is None:
#         plan = fft_plan((Npatch_y,Npatch_x))
#
#     print Nblock_x, Npatch_x
#     patches_g = OCLArray.empty((Gy,Gx,Npatch_y,Npatch_x),np.complex64)
#
#     h_g = OCLArray.from_array(hs.astype(np.complex64))
#
#     im_g = OCLImage.from_array(im.astype(np.float32,copy=False))
#
#     x0s = np.linspace(0,Nx-1,Gx).astype(int)
#     y0s = np.linspace(0,Ny-1,Gy).astype(int)
#
#     print x0s
#
#     for i,_x0 in enumerate(x0s):
#         for j,_y0 in enumerate(y0s):
#             prog.run_kernel("fill_patch2",(Npatch_x,Npatch_y),None,
#                     im_g,np.int32(_x0-Npatch_x/2),np.int32(_y0-Npatch_y/2),
#                     patches_g.data,
#                     np.int32(i*Npatch_x*Npatch_y+j*Gx*Npatch_x*Npatch_y))
#
#     # im_g = OCLArray.from_array(im.astype(np.float32,copy=False))
#     # for i,_x0 in enumerate(x0s):
#     #     for j,_y0 in enumerate(y0s):
#     #         prog.run_kernel("fill_patch_2d2",(Npatch_x,Npatch_y),None,
#     #                     im_g.data,
#     #                     np.int32(Nx),np.int32(Ny),
#     #                     np.int32(_x0-Npatch_x/2),np.int32(_y0-Npatch_y/2),
#     #                     patches_g.data,np.int32(i*Npatch_x*Npatch_y+j*Gx*Npatch_x*Npatch_y))
#
#     # convolution
#     fft(patches_g,inplace=True, batch = Gx*Gy, plan = plan)
#     fft(h_g,inplace=True, batch = Gx*Gy, plan = plan)
#     patches_g = patches_g *h_g
#     fft(patches_g,inplace=True, inverse = True, batch = Gx*Gy, plan = plan)
#
#
#
#     #accumulate
#     res_g = OCLArray.empty(im.shape,np.float32)
#
#     for i in xrange(Gx-1):
#         for j in xrange(Gy-1):
#             prog.run_kernel("interpolate2",(Nblock_x,Nblock_y),None,
#                             patches_g.data,res_g.data,
#                             np.int32(i),np.int32(j),
#                             np.int32(Gx),np.int32(Gy),
#                             np.int32(Npatch_x),np.int32(Npatch_y))
#
#
#     res = res_g.get()
#
#     if return_plan:
#         return res, plan
#     else:
#         return res

def convolve_spatial2(im, hs,
                      mode = "constant",
                      plan = None,
                      return_plan = False):
    """
    spatial varying convolution of an 2d image with a 2d grid of psfs

    shape(im_ = (Ny,Nx)
    shape(hs) = (Gy,Gx, Hy,Hx)

    the input image im is subdivided into (Gy,Gz) blocks
    hs[j,i] is the psf at the center of each block (i,j)

    as of now each image dimension has to be divisble by the grid dim, i.e.
    Nx % Gx == 0
    Ny % Gy == 0

    mode can be:
    "constant" - assumed values to be zero
    "wrap" - periodic boundary condition
    """

    if im.ndim !=2 or hs.ndim !=4:
        raise ValueError("wrong dimensions of input!")

    if not np.all([n%g==0 for n,g in zip(im.shape,hs.shape[:2])]):
        raise NotImplementedError("shape of image has to be divisible by Gx Gy  = %s shape mismatch"%(str(hs.shape[:2])))


    mode_str = {"constant":"CLK_ADDRESS_CLAMP",
                "wrap":"CLK_ADDRESS_REPEAT"}

    Ny, Nx = im.shape
    Gy, Gx = hs.shape[:2]


    # the size of each block within the grid
    Nblock_y, Nblock_x = Ny/Gy, Nx/Gx


    # the size of the overlapping patches with safety padding
    Npatch_x, Npatch_y = next_power_of_2(3 * Nblock_x), next_power_of_2(3 * Nblock_y)
    #Npatch_x, Npatch_y = _next_power_of_2(2*Nblock_x), _next_power_of_2(2*Nblock_y)

    print(Nblock_x, Npatch_x)

    hs = np.fft.fftshift(pad_to_shape(hs,(Gy,Gx,Npatch_y,Npatch_x)),axes=(2,3))


    prog = OCLProgram(abspath("kernels/conv_spatial.cl"),
                      build_options=["-D","ADDRESSMODE=%s"%mode_str[mode]])

    if plan is None:
        plan = fft_plan((Npatch_y,Npatch_x))


    patches_g = OCLArray.empty((Gy,Gx,Npatch_y,Npatch_x),np.complex64)

    h_g = OCLArray.from_array(hs.astype(np.complex64))

    im_g = OCLImage.from_array(im.astype(np.float32,copy=False))

    x0s = Nblock_x*np.arange(Gx)
    y0s = Nblock_y*np.arange(Gy)

    print(x0s)

    for i,_x0 in enumerate(x0s):
        for j,_y0 in enumerate(y0s):
            prog.run_kernel("fill_patch2",(Npatch_x,Npatch_y),None,
                    im_g,
                    np.int32(_x0+Nblock_x/2-Npatch_x/2),
                    np.int32(_y0+Nblock_y/2-Npatch_y/2),
                    patches_g.data,
                    np.int32(i*Npatch_x*Npatch_y+j*Gx*Npatch_x*Npatch_y))

    # convolution
    fft(patches_g,inplace=True, batch = Gx*Gy, plan = plan)
    fft(h_g,inplace=True, batch = Gx*Gy, plan = plan)
    prog.run_kernel("mult_inplace",(Npatch_x*Npatch_y*Gx*Gy,),None,
                    patches_g.data, h_g.data)

    fft(patches_g,inplace=True, inverse = True, batch = Gx*Gy, plan = plan)

    #return patches_g.get()

    #accumulate
    res_g = OCLArray.empty(im.shape,np.float32)

    for i in range(Gx+1):
        for j in range(Gy+1):
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





def convolve_spatial3(im, hs,
                      mode = "constant",
                      plan = None,
                      return_plan = False,
                      pad_factor = 2):
    """
    spatial varying convolution of an 3d image with a 3d grid of psfs

    shape(im_ = (Nz,Ny,Nx)
    shape(hs) = (Gz,Gy,Gx, Hz,Hy,Hx)

    the input image im is subdivided into (Gx,Gy,Gz) blocks
    hs[k,j,i] is the psf at the center of each block (i,j,k)

    as of now each image dimension has to be divisble by the grid dim, i.e.
    Nx % Gx == 0
    Ny % Gy == 0
    Nz % Gz == 0

    mode can be:
    "constant" - assumed values to be zero
    "wrap" - periodic boundary condition


    """
    if im.ndim !=3 or hs.ndim !=6:
        raise ValueError("wrong dimensions of input!")

    if not np.all([n%g==0 for n,g in zip(im.shape,hs.shape[:3])]):
        raise NotImplementedError("shape of image has to be divisible by Gx Gy  = %s !"%(str(hs.shape[:3])))


    mode_str = {"constant":"CLK_ADDRESS_CLAMP",
                "wrap":"CLK_ADDRESS_REPEAT"}

    Ns = tuple(im.shape)
    Gs = tuple(hs.shape[:3])


    # the size of each block within the grid
    Nblocks = [n/g for n,g  in zip(Ns,Gs)]


    # the size of the overlapping patches with safety padding
    Npatchs = tuple([next_power_of_2(pad_factor * nb) for nb in Nblocks])

    print(hs.shape)
    hs = np.fft.fftshift(pad_to_shape(hs,Gs+Npatchs),axes=(3,4,5))



    prog = OCLProgram(abspath("kernels/conv_spatial.cl"),
                      build_options=["-D","ADDRESSMODE=%s"%mode_str[mode]])

    if plan is None:
        plan = fft_plan(Npatchs)

    patches_g = OCLArray.empty(Gs+Npatchs,np.complex64)

    h_g = OCLArray.from_array(hs.astype(np.complex64))

    im_g = OCLImage.from_array(im.astype(np.float32,copy=False))

    Xs = [nb*np.arange(g) for nb, g in zip(Nblocks,Gs)]




    print(Nblocks)
    # this loops over all i,j,k
    for (k,_z0), (j,_y0),(i,_x0) in product(*[enumerate(X) for X in Xs]):
        prog.run_kernel("fill_patch3",Npatchs[::-1],None,
                im_g,
                    np.int32(_x0+Nblocks[2]/2-Npatchs[2]/2),
                    np.int32(_y0+Nblocks[1]/2-Npatchs[1]/2),
                    np.int32(_z0+Nblocks[0]/2-Npatchs[0]/2),
                    patches_g.data,
                    np.int32(i*np.prod(Npatchs)+
                             j*Gs[2]*np.prod(Npatchs)+
                             k*Gs[2]*Gs[1]*np.prod(Npatchs)))



    print(patches_g.shape, h_g.shape)




    # convolution
    fft(patches_g,inplace=True, batch = np.prod(Gs), plan = plan)
    fft(h_g,inplace=True, batch = np.prod(Gs), plan = plan)
    prog.run_kernel("mult_inplace",(np.prod(Npatchs)*np.prod(Gs),),None,
                    patches_g.data, h_g.data)

    fft(patches_g,
        inplace=True,
        inverse = True,
        batch = np.prod(Gs),
        plan = plan)

    #return patches_g.get()
    #accumulate
    res_g = OCLArray.zeros(im.shape,np.float32)

    for k, j, i in product(*[list(range(g+1)) for g in Gs]):
        prog.run_kernel("interpolate3",Nblocks[::-1],None,
                        patches_g.data,
                        res_g.data,
                        np.int32(i),np.int32(j),np.int32(k),
                        np.int32(Gs[2]),np.int32(Gs[1]),np.int32(Gs[0]),
                        np.int32(Npatchs[2]),np.int32(Npatchs[1]),np.int32(Npatchs[0]))


    res = res_g.get()

    if return_plan:
        return res, plan
    else:
        return res


#
# def convolve_spatial3(im, hs, plan = None, return_plan = False, n_split_volumes = 1):
#     """
#     spatial varying convolution of an 3d image with a 2d grid of psfs
#
#     shape(im) = (Nz,Ny,Nx)
#     shape(hs) = (Gy,Gx, Hz, Hy, Hx)
#
#     the psfs are assumed to be defined equally spaced
#     i.e. hs[0,0] is at (0,0) and hs[-1,-1] at (Ny-1,Nx-1)
#     """
#
#     # if not np.all([_is_power2(n) for n in im.shape]):
#     #     raise NotImplementedError("im.shape == %s has to be power of 2 as of now"%(str(im.shape)))
#
#     if not _is_power2(im.shape[0]):
#         raise NotImplementedError("im.shape[0] == %s has to be power of 2 as of now"%(str(im.shape[0])))
#
#
#     if not np.all([n%(g-1)==0 for n,g in zip(im.shape[:2],hs.shape[:2])]):
#         raise NotImplementedError("Gx Gy  = %s shape mismatch (e.g. Nx has to be divisible by Gx+1)"%(str(hs.shape)))
#
#
#
#     if n_split_volumes==1:
#         return _convolve_spatial3(im, hs, plan = plan, return_plan = return_plan)
#     else:
#         #split into subvolumes with the right overlap
#
#         res = np.empty_like(im)
#         Nz, Ny, Nx = im.shape
#         Gy, Gx = hs.shape[:2]
#         Nblock_y, Nblock_x = Ny/(Gy-1), Nx/(Gx-1)
#
#
#         if Gx>=Gy:
#             Gx_part = int(np.ceil(1.*(Gx-1)/n_split_volumes))
#             print Gx_part
#
#
#
#             s_hs = [slice(max(0,i*Gx_part -1),min((i+1)*Gx_part +1,Gx))
#                       for i in xrange(n_split_volumes)]
#
#             pad_left = [0]+[1]*max(0,n_split_volumes-1)
#             pad_right = [1]*max(0,n_split_volumes-1)+[0]
#
#
#             Gx_start = [max(0,i*Gx_part - p) for i,p in zip(range(n_split_volumes),pad_left)]
#             Gx_end = [min((i+1)*Gx_part + p,Gx) for i,p in zip(range(n_split_volumes),pad_right)]
#
#
#             s_hs = [slice(g1,g2)
#                       for g1,g2 in zip(Gx_start,Gx_end)]
#
#             s_im_in = [slice(g1*Nblock_x,g2*Nblock_x)
#                       for g1,g2 in zip(Gx_start,Gx_end)]
#
#
#             # s_im_out = [slice(max(0,(i*Gx_part -1)*Nblock_x),min(((i+1)*Gx_part +1)*Nblock_x,Nx))
#             #             for i in xrange(n_split_volumes)]
#
#             # s_res = [slice(max(0,(i*Gx_part -1)*Nblock_x),min(((i+1)*Gx_part +1)*Nblock_x,Nx))
#             #             for i in xrange(n_split_volumes)]
#             #
#
#             im_part= im[...,s_im_in[0]].copy()
#             hs_part = hs[:,s_hs[0],...].copy()
#
#             return _convolve_spatial3(im_part,hs_part,plan = plan)
#
#         pass




if __name__ == '__main__':
    pass

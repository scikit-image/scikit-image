"""

fftshift on OCLArrays

as of now, only supports even  dimensions (as ifftshift == fftshift then ;)

kernels adapted from
Abdellah, Marwan.
cufftShift: high performance CUDA-accelerated FFT-shift library.
Proc High Performance Computing Symposium.
2014.


mweigert@mpi-cbg.de

"""
from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
from gputools import OCLArray, OCLProgram
from ._abspath import abspath


DTYPE_KERNEL_NAMES = {np.float32:"fftshift_1_f",
                   np.complex64:"fftshift_1_c"}


def fftshift(arr_obj, axes = None, res_g = None, return_buffer = False):
    """
    gpu version of fftshift for numpy arrays or OCLArrays

    Parameters
    ----------
    arr_obj: numpy array or OCLArray (float32/complex64)
        the array to be fftshifted
    axes: list or None
        the axes over which to shift (like np.fft.fftshift)
        if None, all axes are taken
    res_g:
        if given, fills it with the result (has to be same shape and dtype as arr_obj)
        else internally creates a new one
    Returns
    -------
        if return_buffer, returns the result as (well :) OCLArray
        else returns the result as numpy array

    """

    if axes is None:
        axes = list(range(arr_obj.ndim))


    if isinstance(arr_obj, OCLArray):
        if not arr_obj.dtype.type in DTYPE_KERNEL_NAMES:
            raise NotImplementedError("only works for float32 or complex64")
    elif isinstance(arr_obj, np.ndarray):
        if np.iscomplexobj(arr_obj):
            arr_obj = OCLArray.from_array(arr_obj.astype(np.complex64,copy = False))
        else:
            arr_obj = OCLArray.from_array(arr_obj.astype(np.float32,copy = False))
    else:
        raise ValueError("unknown type (%s)"%(type(arr_obj)))

    if not np.all([arr_obj.shape[a]%2==0 for a in axes]):
        raise NotImplementedError("only works on axes of even dimensions")

    if res_g is None:
        res_g = OCLArray.empty_like(arr_obj)


    # iterate over all axes
    # FIXME: this is still rather inefficient
    in_g = arr_obj
    for ax in axes:
        _fftshift_single(in_g, res_g, ax)
        in_g = res_g

    if return_buffer:
        return res_g
    else:
        return res_g.get()


def _fftshift_single(d_g, res_g, ax = 0):
    """
    basic fftshift of an OCLArray


    shape(d_g) =  [N_0,N_1...., N, .... N_{k-1, N_k]
    = [N1, N, N2]

    the we can address each element in the flat buffer by

     index = i + N2*j + N2*N*k

    where   i = 1 .. N2
            j = 1 .. N
            k = 1 .. N1

    and the swap of elements is performed on the index j
    """

    dtype_kernel_name = {np.float32:"fftshift_1_f",
                   np.complex64:"fftshift_1_c"
                   }

    N = d_g.shape[ax]
    N1 = 1 if ax==0 else np.prod(d_g.shape[:ax])
    N2 = 1 if ax == len(d_g.shape)-1 else np.prod(d_g.shape[ax+1:])

    dtype = d_g.dtype.type

    prog = OCLProgram(abspath("kernels/fftshift.cl"))
    prog.run_kernel(dtype_kernel_name[dtype],(N2,N//2,N1),None,
                    d_g.data, res_g.data,
                    np.int32(N),
                    np.int32(N2))


    return res_g


#
# def _fftshift_core(d_g, res_g, axes = 1):
#     """
#     basic fftshift of a OCLArray
#     """
#
#     dtype_kernel_name = {np.float32:"fftshift_1_f",
#                    np.complex64:"fftshift_1_c"
#                    }
#
#     N = d_g.shape[axes]
#     N_pref = 1 if axes==0 else np.prod(d_g.shape[:axes])
#     N_post = 1 if axes == len(d_g.shape)-1 else np.prod(d_g.shape[axes+1:])
#
#     if axes == 0:
#         stride1 = d_g.shape[1]
#         stride2 = 1
#     if axes == 1:
#         stride1 = 1
#         stride2 = d_g.shape[1]
#
#     # stride1 = N_post
#     # stride2 = N_pref
#     # #offset = N_pref
#     offset = 0
#
#
#     print "strides: ", stride1, stride2
#
#     print N_pref, N,  N_post
#
#     dtype = d_g.dtype.type
#
#     prog = OCLProgram(abspath("kernels/fftshift.cl"))
#     prog.run_kernel(dtype_kernel_name[dtype],(N/2,N_pref*N_post),None,
#                     d_g.data, res_g.data,
#                     np.int32(N),
#                     np.int32(stride1),
#                     np.int32(stride2),
#                     np.int32(offset))
#
#
#     return res_g

#
# def fftshift1(d_g):
#     """
#     1d fftshift inplace
#
#     see
#
#     """
#
#     N,  = d_g.shape
#
#     dtype_kernel_name = {np.float32:"fftshift_1_f",
#                    np.complex64:"fftshift_1_c"
#                    }
#     dtype = d_g.dtype.type
#
#     if not isinstance(d_g, OCLArray):
#         raise ValueError("only works on  OCLArrays")
#
#     if not dtype in dtype_kernel_name.keys():
#         raise NotImplementedError("only works for float32 or complex64")
#
#     if not N%2==0:
#         raise NotImplementedError("only works on even length arryas")
#
#     prog = OCLProgram(abspath("kernels/fftshift.cl"))
#     prog.run_kernel(dtype_kernel_name[dtype],(N/2,),None,
#                     d_g.data, d_g.data,  np.int32(N))
#
#     return d_g
#
# def fftshift2(d_g):
#     """
#     2d fftshift inplace
#     """
#
#
#     Ny, Nx   = d_g.shape
#
#     dtype_kernel_name = {np.float32:"fftshift_2_f",
#                    np.complex64:"fftshift_2_c"
#                    }
#     dtype = d_g.dtype.type
#
#     if not isinstance(d_g, OCLArray):
#         raise ValueError("only works on  OCLArrays")
#
#     if not dtype in dtype_kernel_name.keys():
#         raise NotImplementedError("only works for float32 or complex64")
#
#     if not np.all([n%2==0 for n in d_g.shape]):
#         raise NotImplementedError("only works on even length arryas")
#
#     prog = OCLProgram(abspath("kernels/fftshift.cl"))
#     prog.run_kernel(dtype_kernel_name[dtype],(Nx,Ny,),None,
#                     d_g.data, d_g.data,
#                     np.int32(Nx), np.int32(Ny))

    # return d_g

if __name__ == '__main__':

    Nx, Ny, Nz = (256,)*3
    d = np.linspace(0,1,Nx*Ny*Nz).reshape(Nz, Ny,Nx).astype(np.float32)

    d[Nz//2-30:Nz//2+30,Ny//2-20:Ny//2+20,Nx//2-20:Nx//2+20] = 2.

    d_g = OCLArray.from_array(d)
    out_g = OCLArray.empty_like(d)


    out = fftshift(d, axes= (0,1,2))

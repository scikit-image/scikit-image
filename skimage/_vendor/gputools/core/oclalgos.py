
from __future__ import absolute_import, print_function
from gputools import get_device

from pyopencl import elementwise, reduction, scan, algorithm


class OCLElementwiseKernel(elementwise.ElementwiseKernel):
    """ e.g.
    k = OCLElementwiseKernel(
    "cfloat_t *a, cfloat_t b",
    "a[i] = cfloat_add(b,a[i])",
    "name")

    """
    def __init__(self,*args,**kwargs):
        elementwise.ElementwiseKernel.__init__(self,get_device().context,*args,**kwargs)

class OCLReductionKernel(reduction.ReductionKernel):
    """ e.g.
    r = OCLReductionKernel(
    np.complex64, neutral="0",
    reduce_expr="a+b", map_expr="cfloat_mul(x[i],y[i])",
    arguments="__global cfloat_t *x, __global cfloat_t *y")

    """
    def __init__(self,*args,**kwargs):
        reduction.ReductionKernel.__init__(self,get_device().context,*args,**kwargs)
        
class OCLGenericScanKernel(scan.GenericScanKernel):
    def __init__(self,*args,**kwargs):
        scan.GenericScanKernel.__init__(self,get_device().context,*args,**kwargs)


        
def with_default_context(func):
    def func_with_context(*args,**kwargs):
        func(get_device().context,*args,**kwargs)
    return func_with_context
    



if __name__ == '__main__':
    from volust.volgpu import *
    import numpy as np

    d = np.linspace(0,1,100).astype(np.complex64)

    d += 1.j
    b = OCLArray.from_array(d)

    k = OCLElementWiseKernel(
    "float *a_g, float b",
    "a_g[i] = b + a_g[i]",
    "add")
    

    k2 = OCLElementWiseKernel(
    "cfloat_t *a_g, cfloat_t b",
        "a_g[i] = cfloat_add(b,a_g[i])",
    "add")
    
    r = OCLReductionKernel(
        np.complex64, neutral="0",
        reduce_expr="a+b", map_expr="cfloat_mul(x[i],y[i])",
        arguments="__global cfloat_t *x, __global cfloat_t *y")
    

    

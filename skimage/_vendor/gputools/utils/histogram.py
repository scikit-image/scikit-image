import numpy as np
from gputools.core.ocltypes import OCLArray, OCLProgram, get_device
from gputools.core.ocltypes import cl_buffer_datatype_dict
from ._abspath import abspath


def histogram(x, n_bins = 256):
    if not x.dtype.type in cl_buffer_datatype_dict:
        raise ValueError("dtype %s not supported" % x.dtype.type)
    DTYPE = cl_buffer_datatype_dict[x.dtype.type]

    x_g = OCLArray.from_array(x)

    x0 = x_g.min().get()
    x1 = x_g.max().get()

    local_size = min(get_device().get_info("MAX_WORK_GROUP_SIZE"),
                     2**int(np.log2(np.sqrt(len(x)))))

    red_size = len(x)//local_size
    part_hist_g = OCLArray.zeros((n_bins,red_size), np.uint32)
    hist_g = OCLArray.zeros((n_bins,), np.uint32)

    prog = OCLProgram(abspath("kernels/histogram.cl"), build_options =
                      ["-D","N_BINS=%s"%n_bins,"-D","RED_SIZE=%s"%red_size,
                      "-D","DTYPE=%s"%DTYPE])
    
    prog.run_kernel("histogram_partial",(len(x),),(local_size,),
                    x_g.data, part_hist_g.data, np.float32(x0),np.float32(x1))

    prog.run_kernel("histogram_sum",(n_bins,),None,
                    part_hist_g.data, hist_g.data)
    
    return hist_g.get()

import numpy as np
def fft(x,n=None,axis=1,overwrite_x=False):
    from .._shared.fft import fftmodule #shared fft /module/ for np and scipy
                                        #likely a better way to just hijack the existing failover
    try:
        import gputools.fft
        if (overwrite_x == True and n == None): # the only conditions that gputools supports
            print("using gpu")
            return gputools.fft(x)
        elif n == None:
            out = np.copy(x)
            return gputools.fft(out)
        else:
            print("falling back to fftmodule") 
            return fftmodule.fftn(x,n,axis,overwrite_x)
    except ImportError: # instant fallback
        return fftmodule.fftn(x,n,axis,overwrite_x)

def fftconvolve():
    pass
def fft(x,n=None,axis=1,overwrite_x=False):
    from .._shared.fft import fftmodule
    try:
        import gputools.fft
        if (overwrite_x == True and n == None):
            print("using gpu")
            return gputools.fft(x)
        else:
            print("falling back to fftmodule")
            return fftmodule.fftn(x,n,axis,overwrite_x)
    except ImportError:
        return fftmodule.fftn(x,n,axis,overwrite_x)

def fftconvolve():
    pass
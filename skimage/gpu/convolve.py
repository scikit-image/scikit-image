from scipy import ndimage as ndi
import numpy as np
try:
    import gputools
except ImportError:
    def convolve(input,weights,output=None,mode='reflect',cval=0.0,origin=0):
        ndi.convolve(input,weights,output=output,mode=mode,cval=cval,origin=origin)

def convolve(input,weights,output=None,mode='reflect',cval=0.0,origin=0):
    if mode == 'constant' and cval == 0.0:
        print('using gpu!')
        out = gputools.convolve(np.array(input),np.array(weights))
        if output != None:
            output = out
    else:
        print('falling back to scikit-ndi')
        out = ndi.convolve(input,weights,output=output,mode=mode,cval=cval,origin=origin)

    return out

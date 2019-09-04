from scipy import ndimage as ndi
import numpy as np

#setup the imports, attempt gputools, if not, fallback instantly
try:
    import gputools
except ImportError:
    def convolve(input,weights,output=None,mode='reflect',cval=0.0,origin=0):
        ndi.convolve(input,weights,output=output,mode=mode,cval=cval,origin=origin)

def convolve(input,weights,output=None,mode='reflect',cval=0.0,origin=0):
    if mode == 'constant' and cval == 0.0: #These are the only two conditions that gputools supports
        print('using gpu!') #print for debugging
        out = gputools.convolve(np.array(input),np.array(weights)) 
        # we need to do some type conversion here in order to maintain full compatibility
        # as ndi autocasts to np arrays, but gputools does not
        if output != None:
            output = out
    else:
        print('falling back to scikit-ndi')
        out = ndi.convolve(input,weights,output=output,mode=mode,cval=cval,origin=origin)

    return out

from scipy import ndimage as ndi
try:
    import gputools
except ImportError:
    def convolve(input,weights,output=None,mode='reflect',cval=0.0,origin=0):
        ndi.convolve(input,weights,output=output,mode=mode,cval=cval,origin=origin)

def convolve(input,weights,output=None,mode='reflect',cval=0.0,origin=0):
    if output == 'constant' && cval == 0.0:
        out = gputools.convolve(input,weights)
        if output != None:
            output = out
        print('using gpu!')
    else:
        print('falling back to scikit-ndi')
        out = ndi.convolve(input,weights,output=output,mode=mode,cval=cval,origin=origin)

    return out


from skimage.filters import median as cpuMedian

import gputools 
def median(image,selem=None,out=None,mask=None,shift_x=False,shift_y=False,mode='nearest',cval=0.0,behavior='ndimage'):
    #mask is depricated so we don't check it, non-ndimage defaults to rank, which is unsupported by gputools
    #what does size do?? TODO find out
    #selem is size, which is default 3 in gputools
    #selem also supports shapes, which gputools doesn't support
    #NOTE:: GPUTOOLS MEDIAN FILTER DOESN'T SEEM TO BE IN THE __INIT__ FILE, HAD TO EDIT LOCAL COPY
    if (selem == None and shift_x==False and shift_y ==False and mode == 'constant' and behavior =='ndimage'):
        try:
            return gputools.median_filter(image,cval=cval)
        except ValueError: #not 2 or 3d array
            return cpuMedian(image,selem=None,out=None,mask=None,shift_x=False,shift_y=False,mode='nearest',cval=0.0,behavior='ndimage')
    else:
        return cpuMedian(image,selem=None,out=None,mask=None,shift_x=False,shift_y=False,mode='nearest',cval=0.0,behavior='ndimage')



"""
==============================
Compare execution time for
    - skimage.rank.median,
    - skimage.filter import median_filter
    - scipy.ndimage.filters import percentile_filter,

    and

    - skimage.cmorph.dilate
    - skimage.rank.maximum

==============================

to complete

"""
import numpy as np
import matplotlib.pyplot as plt
import time

from skimage import data
from skimage.morphology import dilation,disk
from skimage.filter import median_filter
from scipy.ndimage.filters import percentile_filter
import skimage.rank as rank

def log_timing(func):
    """ Decorator that returns both function results and execution time
    (result, ms)
    """
    def wrapper(*arg):
        t1 = time.time()
        res = func(*arg)
        t2 = time.time()
        ms = (t2-t1)*1000.0
        return (res,ms)
    return wrapper


@log_timing
def cr_med(image,selem):
    return rank.median(image=image,selem = selem)

@log_timing
def cr_max(image,selem):
    return rank.maximum(image=image,selem = selem)

@log_timing
def cm_dil(image,selem):
    return dilation(image=image,selem = selem)

@log_timing
def ctmf_med(image,radius):
    return median_filter(image=image,radius=radius)

@log_timing
def ndi_med(image,n):
    return percentile_filter(image,50,size=n*2-1)

def compare_dilate():
    """ Comparison between
    - crank.maximum rankfilter implementation
    - cmorph.dilate cython implementation

    on increasing structuring element size and increasing image size
    """
    a = data.camera()

    rec = []
    e_range = range(1,20,1)
    for r in e_range:
        elem = disk(r+1)
        #        elem = (np.random.random((r,r))>.5).astype('uint8')
        rc,ms_rc = cr_max(a,elem)
        rcm,ms_rcm = cm_dil(a,elem)
        rec.append((ms_rc,ms_rcm))
        # same structuring element, the results must match
        assert  (rc==rcm).all()

    rec = np.asarray(rec)

    plt.figure()
    plt.title('increasing element size')
    plt.plot(e_range,rec)
    plt.legend(['crank.maximum','cmorph.dilate'])
    plt.figure()
    plt.imshow(np.hstack((rc,rcm)))

    r = 9
    elem = disk(r+1)

    rec = []
    s_range = range(100,1000,100)
    for s in s_range:
        a = (np.random.random((s,s))*256).astype('uint8')
        (rc,ms_rc) = cr_max(a,elem)
        (rcm,ms_rcm) = cm_dil(a,elem)
        rec.append((ms_rc,ms_rcm))
        # same structuring element, the results must match
        assert  (rc==rcm).all()

    rec = np.asarray(rec)

    plt.figure()
    plt.title('increasing image size')
    plt.plot(s_range,rec)
    plt.legend(['crank.maximum','cmorph.dilate'])
    plt.figure()
    plt.imshow(np.hstack((rc,rcm)))


def compare_median():
    """ Comparison between
    - crank.median rankfilter implementation
    - ctmf.median_filter filter

    on increasing structuring element size and increasing image size
    """
    a = data.camera()

    rec = []
    e_range = range(2,30,4)
    for r in e_range:
        elem = disk(r+1)
        rc,ms_rc = cr_med(a,elem)
        rctmf,ms_rctmf = ctmf_med(a,r)
        rndi,ms_ndi = ndi_med(a,r)
        rec.append((ms_rc,ms_rctmf,ms_ndi))
        # check if results are identical
        # obviously they cannot be identical since structuring element are different (octagon<>disk)
        # assert  (rc==rctmf).all()

    rec = np.asarray(rec)

    plt.figure()
    plt.title('increasing element size')
    plt.plot(e_range,rec)
    plt.legend(['rank.median','ctmf.median_filter','ndimage.percentile'])
    plt.ylabel('time (ms)')
    plt.xlabel('element radius')
    plt.figure()
    plt.imshow(np.hstack((rc,rctmf,rndi)))
    plt.xlabel('rank.median vs ctmf.median_filter vs ndimage.percentile')

    r = 9
    elem = disk(r+1)

    rec = []
    s_range = [100,200,500,1000,2000]
    for s in s_range:
        a = (np.random.random((s,s))*256).astype('uint8')
        (rc,ms_rc) = cr_med(a,elem)
        rctmf,ms_rctmf = ctmf_med(a,r)
        rndi,ms_ndi = ndi_med(a,r)
        rec.append((ms_rc,ms_rctmf,ms_ndi))
        # check if results are identical
        # obviously they cannot be identical since structuring element are different (octagon<>disk)
        # assert  (rc==rctmf).all()

    rec = np.asarray(rec)

    plt.figure()
    plt.title('increasing image size')
    plt.plot(s_range,rec)
    plt.legend(['rank.median','ctmf.median_filter','ndimage.percentile'])
    plt.ylabel('time (ms)')
    plt.xlabel('image size')



compare_dilate()
compare_median()
plt.show()

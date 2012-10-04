import numpy as np
import matplotlib.pyplot as plt

from skimage import data
from skimage.morphology import cmorph
from skimage.rank import crank8

from tools import log_timing

@log_timing
def cr_max(image,selem):
    return crank8.maximum(image=image,selem = selem)

@log_timing
def cm_dil(image,selem):
    return cmorph.dilate(image=image,selem = selem)


def compare():
    """comparison between
    - crank.maximum rankfilter implementation
    - cmorph.dilate cython implementation
    on increasing structuring element size and increasing image size
    """
#    a = (np.random.random((500,500))*256).astype('uint8')
    a = data.camera()

    rec = []
    e_range = range(1,20,1)
    for r in e_range:
        elem = np.ones((r,r),dtype='uint8')
        #        elem = (np.random.random((r,r))>.5).astype('uint8')
        rc,ms_rc = cr_max(a,elem)
        rcm,ms_rcm = cm_dil(a,elem)
        rec.append((ms_rc,ms_rcm))
        # check if results are identical
        assert  (rc==rcm).all()

    rec = np.asarray(rec)

    plt.figure()
    plt.title('increasing element size')
    plt.plot(e_range,rec)
    plt.legend(['crank.maximum','cmorph.dilate'])
    plt.figure()
    plt.imshow(np.hstack((rc,rcm)))

    r = 9
    elem = np.ones((r,r),dtype='uint8')

    rec = []
    s_range = range(100,1000,100)
    for s in s_range:
        a = (np.random.random((s,s))*256).astype('uint8')
        (rc,ms_rc) = cr_max(a,elem)
        (rcm,ms_rcm) = cm_dil(a,elem)
        rec.append((ms_rc,ms_rcm))
        assert  (rc==rcm).all()

    rec = np.asarray(rec)

    plt.figure()
    plt.title('increasing image size')
    plt.plot(s_range,rec)
    plt.legend(['crank.maximum','cmorph.dilate'])
    plt.figure()
    plt.imshow(np.hstack((rc,rcm)))

    plt.show()

if __name__ == '__main__':
    compare()
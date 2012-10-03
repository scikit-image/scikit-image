import numpy as np
from time import time
import matplotlib.pyplot as plt
from skimage import data

from tools import log_timing,init_logger

import crank
import crank16
import crank_percentiles
import crank16_percentiles
from pyrankfilter import filter
from cmorph import dilate


@log_timing
def c_max(image,selem):
    return crank.maximum(image=image,selem = selem)

@log_timing
def w_max(image,selem):
    return filter.maximum(image,struct_elem = selem)

@log_timing
def cm_max(image,selem):
    return dilate(image=image,selem = selem)

def compare():
    """comparison between
    - Cython maximum rankfilter implementation
    - weaves maximum rankfilter implementation
    - cmorph.dilate cython implementation
    on increasing structuring element size and increasing image size
    """
    a = (np.random.random((500,500))*256).astype('uint8')

    rec = []
    for r in range(1,20,1):
        elem = np.ones((r,r),dtype='uint8')
        #        elem = (np.random.random((r,r))>.5).astype('uint8')
        (rc,ms_rc) = c_max(a,elem)
        (rw, ms_rw) = w_max(a,elem)
        (rcm,ms_rcm) = cm_max(a,elem)
        rec.append((ms_rc,ms_rw,ms_rcm))
        assert  (rc==rw).all()
        assert  (rc==rcm).all()

    rec = np.asarray(rec)

    plt.plot(rec)
    plt.legend(['sliding cython','sliding weaves','cmorph'])
    plt.figure()
    plt.imshow(np.hstack((rc,rw,rcm)))

    r = 9
    elem = np.ones((r,r),dtype='uint8')

    rec = []
    for s in range(100,1000,100):
        a = (np.random.random((s,s))*256).astype('uint8')
        (rc,ms_rc) = c_max(a,elem)
        (rw, ms_rw) = w_max(a,elem)
        (rcm,ms_rcm) = cm_max(a,elem)
        rec.append((ms_rc,ms_rw,ms_rcm))
        assert  (rc==rw).all()
        assert  (rc==rcm).all()

    rec = np.asarray(rec)

    plt.figure()
    plt.plot(rec)
    plt.legend(['sliding cython','sliding weaves','cmorph'])
    plt.figure()
    plt.imshow(np.hstack((rc,rw,rcm)))

    plt.show()

def test_image_size():
    """try several image sizes to check bounds conditions
    """
    niter = 10
    elem = np.asarray([[1,1,1],[1,1,1],[1,1,1]],dtype='uint8')
    for m,n in np.random.random_integers(1,100,size=(10,2)):
        a = np.ones((m,n),dtype='uint8')
        r = crank.mean(image=a,selem = elem,shift_x=0,shift_y=0)
        assert a.shape == r.shape
        r = crank.mean(image=a,selem = elem,shift_x=0,shift_y=-1)
        assert a.shape == r.shape
        r = crank.mean(image=a,selem = elem,shift_x=0,shift_y=+1)
        assert a.shape == r.shape
        r = crank.mean(image=a,selem = elem,shift_x=-1,shift_y=0)
        assert a.shape == r.shape
        r = crank.mean(image=a,selem = elem,shift_x=+1,shift_y=0)
        assert a.shape == r.shape
        r = crank.mean(image=a,selem = elem,shift_x=-1,shift_y=-1)
        assert a.shape == r.shape
        r = crank.mean(image=a,selem = elem,shift_x=+1,shift_y=+1)
        assert a.shape == r.shape

    return True


if __name__ == '__main__':

    logger = init_logger('app.log')
    a = np.zeros((10,10),dtype='uint8')
    a[2,2] = 255
#    a[2,3] = 255
#    a[2,4] = 255

    print a

    mask = np.ones_like(a)
#    mask[:3,:3] = 0

#    elem = np.asarray([[0,1,0],[1,1,1],[0,1,0]],dtype='uint8')
    elem = np.asarray([[1,1,0],[1,1,1],[0,0,1]],dtype='uint8')

    niter = 1
    t0 = time()

    for iter in range(niter):
        r = crank.mean(image=a,selem = elem,shift_x=0,shift_y=0,mask = mask)
        p = crank.pop(image=a,selem = elem,shift_x=0,shift_y=0,mask = mask)
    t1 = time()
    print '%f msec'%(t1-t0)

    print 'cython mean'
    print r
    print p

    t0 = time()
    for iter in range(niter):
        r = filter.mean(a,struct_elem = elem,struct_elem_center=(1,1),mask = mask)
    t1 = time()
    print '%f msec'%(t1-t0)

    print 'filter.mean:'
    print r

    print a
    r = crank.maximum(image=a,selem = elem,shift_x=0,shift_y=0,mask = mask)
    print r

    r = crank.gradient(image=r,selem = elem,shift_x=0,shift_y=0,mask = mask)
    print r
    im = np.zeros((10,10),dtype='uint8')
    im[2:6,2:6] = 255
    elem = np.asarray([[1,1,1],[1,1,1],[1,1,1]],dtype='uint8')
    f = crank.gradient(image=im,selem = elem)
    print f
    f = crank.egalise(image=im,selem = elem)
    print f

#    compare()
#    test_image_size()

#    a = (data.coins()).astype('uint8')
    a8 = (data.coins()).astype('uint8')
    a = (data.coins()).astype('uint16')*16
    selem = np.ones((20,20),dtype='uint8')
#    f1 = filter.soft_gradient(a,struct_elem = selem,bitDepth=8,infSup=[.1,.9])
#    f2 = crank16.bottomhat(a,selem = selem,bitdepth=12)
    f1 = crank_percentiles.mean(a8,selem = selem,p0=.1,p1=.9)
    f2 = crank16_percentiles.mean(a,selem = selem,bitdepth=12,p0=.1,p1=.9)
#    plt.imshow(f2)
    plt.imshow(np.hstack((f1,f2)))
    plt.colorbar()
    plt.show()



# cython: boundscheck=False
# cython: wraparound=False
# cython: profile=False

################################################################################
# filename: ridge_directed_ring_detector.pyx
# first online: https://github.com/eldad-a/ridge-directed-ring-detector
# 
# For academic citation please use:
#     Afik, E. 
#     Robust and highly performant ring detection algorithm for 3d particle tracking using 2d microscope imaging.
#     Sci. Rep. 5, 13584; doi: 10.1038/srep13584 (2015).
# 
#
#  Copyright (c) 2012, Eldad Afik
#  All rights reserved.
#  
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#  
#  * Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#  
#  * Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#  
#  * Neither the name of ridge_directed_ring_detector nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#  
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
#  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#  
################################################################################

# TODO: cast indices to unsigned int
# TODO: add a test to verify rij <-> coords conversion works as expected
# (assersion)
# TODO: add edge detection version

from __future__ import division 
from libc.math cimport sqrt, copysign, cos, abs, fabs, ceil, exp, log2
from libc.math cimport M_PI as pi

from cpython cimport bool

#from collections import defaultdict

import numpy as np
cimport numpy as np

import cv2
from cv2 import fitEllipse

from scipy import optimize

import cython
cimport cython

DTYPE = np.float32
ctypedef np.float32_t DTYPE_t

RIJ = np.uint32
ctypedef np.uint32_t RIJ_t

INDX = np.int
ctypedef Py_ssize_t INDX_t

ctypedef unsigned short KINT_t

ctypedef struct coord_t:
    RIJ_t r
    RIJ_t i
    RIJ_t j

cdef DTYPE_t cos_q_pi = cos(pi/8)

###   (r,i,j)  <->  rij   ###
## set maximal possible image size and circle radii
#cdef RIJ_t max_rads=256, max_rows=1456, max_cols=1936
DEF MaxRads = 256
DEF MaxRows = 1456
DEF MaxCols = 1936
cdef RIJ_t shiftRows = <RIJ_t>(fround(ceil(log2(MaxCols))))
cdef RIJ_t shiftRads = shiftRows + <RIJ_t>(fround(ceil(log2(MaxRows))))
cdef RIJ_t modCols = (1<<shiftRows)-1
cdef RIJ_t modRows = (1<<shiftRads)-1


# for the votes2rings function
DEF One = 1
DEF Two = 2
DEF Three = 3
DEF MaxRingsNo = 1000


@cython.profile(False)
cpdef inline RIJ_t coord2rij(RIJ_t r, RIJ_t i, RIJ_t j):
    return j + (i<<shiftRows) + (r<<shiftRads)

@cython.profile(False)
cpdef inline coord_t rij2coord(RIJ_t rij):
    cdef coord_t coords
    coords.i = (rij&modRows)>>shiftRows
    coords.j = rij&modCols
    coords.r = rij>>shiftRads
    return coords

@cython.profile(False)
cdef inline INDX_t fround(DTYPE_t x):
    return <INDX_t>(x+.5) if x>=0. else <INDX_t>(x-.5)

@cython.profile(False)
cpdef inline DTYPE_t f_least_principal_curvature(DTYPE_t Lxx, 
                                                DTYPE_t Lyy, 
                                                DTYPE_t Lxy):
    '''
    compute the smallest eigen-value of the Hessian matrix
    
    Inputs:
    Lxx,Lyy,Lxy are the Hessian matrix components 
    '''
    #cdef:
    #    DTYPE_t trH
    #    DTYPE_t discriminant
    #    DTYPE_t curvature
    #trH = Lxx+Lyy
    # numerically stabilising for the cases of extrimly small Lxy:
    #discriminant = (Lxx-Lyy)**2 + 4*Lxy*Lxy
    #curvature = 0.5*(trH - sqrt( discriminant ))
    #return curvature
    return 0.5*(Lxx + Lyy - sqrt((Lxx-Lyy)**2 + 4*Lxy*Lxy ))


cpdef least_principal_curvature(DTYPE_t [:,:] Lxx, DTYPE_t [:,:] Lyy,
                                DTYPE_t [:,:] Lxy):
    '''
    compute the smallest eigen-value of the Hessian matrix
    
    Inputs:
        Lxx,Lyy,Lxy are the Hessian matrix components 
    '''
    assert Lxx is not None and Lyy is not None and Lxy is not None

    cdef:
        INDX_t Nrows 
        INDX_t Ncols 
        DTYPE_t [:,:] curv 
        INDX_t i,j
    
    Nrows = Lxx.shape[0]
    Ncols = Lxx.shape[1]
    curv = np.empty((Nrows,Ncols),dtype=DTYPE)
    
    for i in xrange(Nrows):
        for j in xrange(Ncols):
            curv[i,j] = f_least_principal_curvature(Lxx[i,j], 
                                                    Lyy[i,j],
                                                    Lxy[i,j])
    return curv


#@cython.profile(False)
@cython.cdivision(True)
cpdef inline least_principal_direction(DTYPE_t Lxx, DTYPE_t Lyy, DTYPE_t Lxy):
    '''
    compute the [cos , sin, tan] of the angle formed 
    by the eigen-vector of the Hessian Matrix, 
    corresponding to the smaller eigen-value
        
    Inputs:
        Lxx,Lyy,Lxy are the Hessian matrix components 
    
    Note: input are expected to be single numbers (not arrays)
    '''
    cdef DTYPE_t D, tangent, denominator
    if Lxy==0: 
        return (1.,0.) if Lxx<Lyy else (0.,1.)
    elif Lxy>0:
        D = .5*(Lxx-Lyy)/Lxy
        tangent = -D - sqrt(D*D + 1)
    elif Lxy<0:
        D = .5*(Lxx-Lyy)/Lxy
        tangent = -D + sqrt(D*D + 1)
    denominator = sqrt(1 + tangent*tangent)
    return  1./denominator, tangent / denominator


#@cython.profile(False)
cpdef inline INDX_t [:,:] vote4(int Rmin, int x0, int y0, int Rmax, int x1, int
        y1):
    '''
    convert the bresenham line 2d to one with no if's, provided that the
    radius axis is always the longest
    '''
    cdef:
        int dx 
        int dr
        int sx, sr, d, i, dx2, dr2
        INDX_t [:,:] coords 

    dr = Rmax - Rmin
    sr=1
    coords = np.empty((dr+1,2), dtype=INDX)
    dr2 = 2 * dr

    # TODO: do x and y at the same run 

    ## fill x
    dx = abs(x1 - x0)
    sx = 1 if x0<x1 else -1
    
    dx2 = 2 * dx
    d = dx2 - dr

    for i in xrange(dr+1):
        coords[i,0] = x0
        while d >= 0:
            x0 = x0 + sx
            d = d - dr2
        d = d + dx2

    ## fill y
    dx = abs(y1 - y0)
    sx = 1 if y0<y1 else -1
    
    dx2 = 2 * dx
    d = dx2 - dr

    for i in xrange(dr+1):
        coords[i,1] = y0
        while d >= 0:
            y0 = y0 + sx
            d = d - dr2
        d = d + dx2

    return coords


#@cython.profile(False)
@cython.cdivision(True)
cpdef DTYPE_t [:] get_1d_gaussian_kernel_r(RIJ_t r):
    cdef:
        DTYPE_t sigma 
        RIJ_t ksize
        RIJ_t cntr
        DTYPE_t [:] kernel 
        DTYPE_t scale2x 
        INDX_t i
    
    sigma = .05*r + .25
    ## based on opencv: createGaussianFilter (smooth.cpp)
    ## NOTE: normalised such that the cntr is unity
    #cdef RIJ_t ksize = <RIJ_t>(8*sigma + 1) | 1 # bitwise OR with 1 adds one to even int
    ## note that I take half the kernel size (for less computations):
    ksize = <RIJ_t>(4*sigma + 1) | 1 # bitwise OR with 1 adds one to even int
    cntr = ksize//2
    #k = cv2.getGaussianKernel(ksize, sigma, cv2.CV_32F)
    kernel = np.empty((ksize),DTYPE)
    scale2x = -0.5/sigma**2

    for i in xrange(ksize):
        kernel[i] = exp(scale2x*(i-cntr)**2)
    return kernel


###   Pre-processing the image   ###
## get Lxx,Lyy & Lxy (the Hessian matrix entries), and the least principal
## curvature (smallest eigen-value of the Hessian)


###   Sparse ridge directed circle Hough transfrom:   ###
##  using loops, analyse only the pixels that passed the threshold, visiting
##  each once: 
##          (a) Non-minimum supression based on quantised principal direction
##          (b) Construct sparse 3D Hough space with two extra Radius slabs for
##              complete local max search (Rmin-1 & Rmax+1)
##          (c) Gaussian smoothed and radius-scaled
##          (c) For each point in the 3D Hough space passing a threshold
##              (based on 40% of 2*pi*r) perform non-maximum suppresion 
##              (in a 3x3x3 cube)

cpdef ridge_circle_hough_transform(DTYPE_t [:,:] Lxx, 
                             DTYPE_t [:,:] Lyy, 
                             DTYPE_t [:,:] Lxy, 
                             DTYPE_t [:,:] curv, 
                             DTYPE_t curv_thresh=-20,
                             RIJ_t Rmin=5,
                             RIJ_t Rmax=55):

    assert Lxx is not None and Lyy is not None and Lxy is not None and\
            curv is not None

    cdef:
        INDX_t rij, i, j, r, r_, x, y, 
        RIJ_t counter, Nrads, Nrows, Ncols
        int x0, y0, x1, y1
        char sign
        DTYPE_t cosQ, sinQ
        dict directed_ridges
        #DTYPE_t [:,:] directed_ridges = np.empty((Nrows*Ncols,4), DTYPE)
        np.ndarray[RIJ_t ,ndim=1] votes
        INDX_t [:,:] vote4xy

    Rmin-=1
    Rmax+=1
    Nrows = curv.shape[0]
    Ncols = curv.shape[1]
    Nrads = Rmax-Rmin+1
    directed_ridges = {}
    counter = 0
    votes = np.empty(Nrows*Ncols*Nrads*2, dtype=RIJ)

    # iterate over all image entries (principal curv in this case).
    # avoid dealing with the boundaries by iterating over all but the entries
    # on the exterior.
    for i in xrange(1,Nrows-1):         
        for j in xrange(1,Ncols-1):
            #
            ###   Find Ridges:    ###
            #
            # treshold the curvature (note that it should be smaller than...)
            if curv[i,j] > curv_thresh: continue
            # perform non-minimum suppression in the least principal direction
            cosQ,sinQ = least_principal_direction(Lxx[i,j], Lyy[i,j], Lxy[i,j])
            if fabs(cosQ) > cos_q_pi:
                if (curv[i,j] >= curv[i,j+1]) | (curv[i,j] >= curv[i,j-1]):
                    continue
            elif fabs(sinQ) > cos_q_pi:
                if (curv[i,j] >= curv[i+1,j]) | (curv[i,j] >= curv[i-1,j]):
                    continue
            elif copysign(1,sinQ) == copysign(1,cosQ):
                if (curv[i,j] >= curv[i+1,j+1]) | (curv[i,j] >= curv[i-1,j-1]):
                    continue
            elif copysign(1,sinQ) != copysign(1,cosQ):
                if (curv[i,j] >= curv[i-1,j+1]) | (curv[i,j] >= curv[i+1,j-1]):
                    continue
           
            # add the least principal direction to the ridge sparse array
            directed_ridges[i,j] = cosQ, sinQ

            ###   Circle Hough Transform   ###

            # TODO: consider Bresenham circle (arcs) instead of Gaussian
            # smoothing

            x0 = fround(sinQ*Rmin)
            y0 = fround(cosQ*Rmin)
            x1 = fround(sinQ*Rmax)
            y1 = fround(cosQ*Rmax)
            vote4xy = vote4(Rmin, x0, y0, Rmax, x1, y1)

            for sign in xrange(-1,2,2):
            # 20130405: corrected erraneous xrange(-1,2)
            # for the inwards and outwards without the zero (inplace)
                for r_ in xrange(Nrads):
                    x = <RIJ_t>(i + sign*vote4xy[r_,0])
                    y = <RIJ_t>(j + sign*vote4xy[r_,1])
                    r = r_+Rmin
                    ## do not need to check for above origin due to the unsign:
                    #if (x<0) | (x>=Nrows) | (y<0) | (y>=Ncols): break
                    if (x>=Nrows) | (y>=Ncols): break

                    votes[counter] = coord2rij(r,x,y)
                    counter+=1
                    
    votes = votes[:counter]
    votes.sort()
    Rmin+=1
    Rmax-=1

    return {'directed_ridges':directed_ridges, 'votes':votes}


#@cython.boundscheck(True)
#@cython.wraparound(True)
@cython.cdivision(True)
cpdef votes2rings(RIJ_t [:] votes,
                  RIJ_t Rmin, RIJ_t Rmax, 
                  RIJ_t Nrows, RIJ_t Ncols, 
                  RIJ_t vote_thresh=1, DTYPE_t circle_thresh=pi):
    """
    # A function which merges the Hough Space construction, smoothing, and 
    # local maxima finding; 
    # doing this in triples, that is - 3 equi-radius planes at a time
    # Note there's some change of notation / variables naming between this
    # function and those which it merges (votes2array, smooth_voted4 & 
    # get_circles).
    #
    # SCHEME:
    # provided a sorted votes array (sorted for r, at least)
    # iterate over all votes (repeated rij's included)
    for rij in votes:
        if r==R:
            # populate the r%3 plane
            hough_slice[Rmod,i,j] += 1
            if rij_nxt != rij:
                hough_modified[Rmod, hough_counter[Rmod]] = i,j
                hough_counter[Rmod] += 1
                if hough_slice[Rmod,i,j] >= vote_thresh:
                    hough_hotspots[Rmod, hotspots_counter] = i,j
                    hotspots_counter[Rmod] += 1
        else:
            # smooth the R plane of the Hough space.
            # In the (R+2)%3 find local maxima (rings) using the radius 
            # dependent gaussian kernel.
            # then clean the (R+1)%3:
            smoothed_slice[(R+1)%3, hough_hotspots[:hotspots_counter]] = 0
            hough_slice[(R+1)%3, hough_modified[:hough_counter[Rmod]]] = 0
            R += 1
    """
    assert votes is not None
    assert vote_thresh>0, 'vote_thresh must be a positive integer, got %s' % \
                                                                   vote_thresh
    cdef:
        RIJ_t votes_size, R, r, i, j, x, y, Ro
        unsigned char Rmod, Romod, R_
        RIJ_t hot, n, rij, rij_nxt
        RIJ_t [:,:,:] hough_slice # a.k.a. sparse_3d_Hough
        DTYPE_t [:,:,:] smoothed_slice # a.k.a. smoothed_hough_array
        RIJ_t [:,:,:] hough_hotspots, hough_modified
        RIJ_t [:] hough_counter, hotspots_counter
        coord_t coords
        DTYPE_t ksigma, kscale2x, value
        RIJ_t ksize, kcentre
        DTYPE_t [:] kernel
        RIJ_t [:,:] rings 
        RIJ_t ring_counter
        INDX_t di, dj, k, ki, kj
        DTYPE_t vote
        bool local_max

    Rmin-=1
    Rmax+=1
    ## prepare a general kernel array to work on
    ksigma = .05*Rmax + .25
    ## note that I take half the kernel size (for less computations):
    ksize = <RIJ_t>(4*ksigma + 1) | 1 # bitwise OR with 1 adds one to 
    # even int
    kernel = np.empty((ksize),DTYPE)

    votes_size = votes.size
    #One,Two,Three = 1,2,3
    R = Rmin
    Rmod = R%Three
    Nrads = Rmax-Rmin+1
    hough_slice = np.zeros((Three,Nrows,Ncols), RIJ)
    smoothed_slice = np.zeros_like(hough_slice, DTYPE)
    hough_modified = np.empty((Three,Nrows*Ncols,2),RIJ)
    hough_counter = np.zeros((3),RIJ)
    hough_hotspots = np.empty((Three,Nrows*Ncols,2),RIJ)
    hotspots_counter =  np.zeros((3),RIJ)
    rings = np.empty((MaxRingsNo,3),RIJ)
    ring_counter = 0

    rij = votes[0]

    ## FIRST: populate the first two radius slabs, (and smooth each one in its 
    ## turn). Then exit this loop, and c ontinue for the rest at "SECOND" (no
    ## need for local max search as the Rmin_ is there just for correct local
    ## max search)
    for n in xrange(1, votes_size):
        coords = rij2coord(rij)
        r,i,j = coords.r, coords.i, coords.j
        if r==R:
            # populate the r%3 plane
            hough_slice[Rmod,i,j] += 1
            rij_nxt = votes[n]
            if rij_nxt != rij:
                hough_modified[Rmod, hough_counter[Rmod],0] = i
                hough_modified[Rmod, hough_counter[Rmod],1] = j
                hough_counter[Rmod] += 1
                if hough_slice[Rmod,i,j] >= vote_thresh:
                    hough_hotspots[Rmod, hotspots_counter[Rmod], 0] = i
                    hough_hotspots[Rmod, hotspots_counter[Rmod], 1] = j
                    hotspots_counter[Rmod] += 1
                rij = rij_nxt
        else:
            # smooth the R plane of the Hough space.
            # TODO consider generating the gaussian kernel as a table a priori
            # (as a class attribute for many images analysis)
            ksigma = .05*R + .25
            ## based on opencv: createGaussianFilter (smooth.cpp)
            ## NOTE: normalised such that the cntr is unity
            #cdef RIJ_t ksize = <RIJ_t>(8*sigma + 1) | 1 
            ## note that I take half the kernel size (for less computations):
            ksize = <RIJ_t>(4*ksigma + 1) | 1 # bitwise OR with 1 adds one to 
            # even int
            kcentre = ksize//2
            kscale2x = -0.5/ksigma**2
            
            for ki in xrange(ksize):
                kernel[ki] = exp(kscale2x*(ki-kcentre)**2)

            for hot in xrange(hotspots_counter[Rmod]):
                i = hough_hotspots[Rmod,hot,0]
                j = hough_hotspots[Rmod,hot,1]
                from_row = i-kcentre if i-kcentre>0 else 0
                to_row = i+kcentre+1 if i+kcentre+1<Nrows else Nrows
                from_col = j-kcentre if j-kcentre>0 else 0
                to_col = j+kcentre+1 if j+kcentre+1<Ncols else Ncols
                value = 0.
                for x in xrange(from_row,to_row):
                    ki = x + kcentre - i
                    for y in xrange(from_col,to_col):
                        kj = y + kcentre - j
                        value += hough_slice[Rmod,x,y]*kernel[ki]*kernel[kj]
                smoothed_slice[Rmod,i,j] = value/R

            ## skip local max in the 3D sparse Hough space at this stage as
            ## should not be done for the Rmin-1 slab, and the Rmin will be
            ## searched for local max at the Rmin+1 population stage
            #
            ## skip clearance at this stage, as this should be done when all
            ## three slabs are populated
            R = r
            Rmod = R%Three
            rij = votes[n]
            if R==Rmin+2: break


    ## SECOND: account for Rmin+2 till Rmax-2 including local max finding, and
    ## smoothing the Rmax-1 (local max not searched for in the Rmax-1 & Rmax
    ## slabs at this stage)
    for n in xrange(n, votes_size):
        coords = rij2coord(rij)
        r,i,j = coords.r, coords.i, coords.j
        if r==R:
            # populate the r%3 plane
            hough_slice[Rmod,i,j] += 1
            rij_nxt = votes[n]
            if rij_nxt != rij:
                hough_modified[Rmod, hough_counter[Rmod],0] = i
                hough_modified[Rmod, hough_counter[Rmod],1] = j
                hough_counter[Rmod] += 1
                if hough_slice[Rmod,i,j] >= vote_thresh:
                    hough_hotspots[Rmod, hotspots_counter[Rmod], 0] = i
                    hough_hotspots[Rmod, hotspots_counter[Rmod], 1] = j
                    hotspots_counter[Rmod] += 1
                rij = rij_nxt
        else:
            # smooth the R plane of the Hough space.
            # TODO consider generating the gaussian kernel as a table a priori
            # (as a class attribute for many images analysis)
            ksigma = .05*R + .25
            ## based on opencv: createGaussianFilter (smooth.cpp)
            ## NOTE: normalised such that the cntr is unity
            #cdef RIJ_t ksize = <RIJ_t>(8*sigma + 1) | 1 
            ## note that I take half the kernel size (for less computations):
            ksize = <RIJ_t>(4*ksigma + 1) | 1 # bitwise OR with 1 adds one to 
            # even int
            kcentre = ksize//2
            #kernel = np.empty((ksize),DTYPE)
            kscale2x = -0.5/ksigma**2
            
            for ki in xrange(ksize):
                kernel[ki] = exp(kscale2x*(ki-kcentre)**2)

            for hot in xrange(hotspots_counter[Rmod]):
                i = hough_hotspots[Rmod,hot,0]
                j = hough_hotspots[Rmod,hot,1]
                from_row = i-kcentre if i-kcentre>0 else 0
                to_row = i+kcentre+1 if i+kcentre+1<Nrows else Nrows
                from_col = j-kcentre if j-kcentre>0 else 0
                to_col = j+kcentre+1 if j+kcentre+1<Ncols else Ncols
                value = 0.
                for x in xrange(from_row,to_row):
                    ki = x + kcentre - i
                    for y in xrange(from_col,to_col):
                        kj = y + kcentre - j
                        value += hough_slice[Rmod,x,y]*kernel[ki]*kernel[kj]
                smoothed_slice[Rmod,i,j] = value/R

            ###   find local max in the 3D sparse Hough space   ###
            #
            # (ii) For every entry in the sparse array which exceeds the
            # threshold verify maximum compared to (rescaled) nearest neighbours
            # (iii) if local max => append to list

            R_ = (R+One)%Three
            Ro = R-One
            Romod = Ro%Three

            for hot in xrange(hotspots_counter[Romod]):
                i = hough_hotspots[Romod,hot,0]
                j = hough_hotspots[Romod,hot,1]
                vote = smoothed_slice[Romod,i,j]
                if vote < circle_thresh: continue
                local_max=True
                for k in xrange(3):#(R_,Romod,Rmod):
                    for di in xrange(-1,2):#(-1,0,1):
                        for dj in xrange(-1,2):#(-1,0,1):
                            local_max &= vote >= smoothed_slice[k,i+di,j+dj]
                            # there should not be any IndexErrors here
                if local_max: 
                    rings[ring_counter,0] = i
                    rings[ring_counter,1] = j
                    rings[ring_counter,2] = Ro
                    ring_counter += 1
    
            # then clean the (R+One)%Three:
            for hot in xrange(hough_counter[R_]):
                i = hough_modified[R_,hot,0]
                j = hough_modified[R_,hot,1]
                hough_slice[R_,i,j] = 0
            #hough_slice[R_,...] = 0
            ## TODO: check whether it is better to simply put this loop in the
            ## previous one, such that extra entries in the smoothed would be
            ## set to zero (redundant), but only one loop.
            for hot in xrange(hotspots_counter[R_]):
                i = hough_hotspots[R_,hot,0]
                j = hough_hotspots[R_,hot,1]
                smoothed_slice[R_,i,j] = 0
            #smoothed_slice[R_,...] = 0
            hough_counter[R_] = 0
            hotspots_counter[R_] = 0
            R = r
            Rmod = R%Three
            rij = votes[n]

    ## smooth the Rmax+1 and find local maximum in Rmax, skip clean up:
    ksigma = .05*R + .25
    ## based on opencv: createGaussianFilter (smooth.cpp)
    ## NOTE: normalised such that the cntr is unity
    #cdef RIJ_t ksize = <RIJ_t>(8*sigma + 1) | 1 
    ## note that I take half the kernel size (for less computations):
    ksize = <RIJ_t>(4*ksigma + 1) | 1 # bitwise OR with 1 adds one to 
    # even int
    kcentre = ksize//2
    #kernel = np.empty((ksize),DTYPE)
    kscale2x = -0.5/ksigma**2
    
    for ki in xrange(ksize):
        kernel[ki] = exp(kscale2x*(ki-kcentre)**2)

    for hot in xrange(hotspots_counter[Rmod]):
        i = hough_hotspots[Rmod,hot,0]
        j = hough_hotspots[Rmod,hot,1]
        from_row = i-kcentre if i-kcentre>0 else 0
        to_row = i+kcentre+1 if i+kcentre+1<Nrows else Nrows
        from_col = j-kcentre if j-kcentre>0 else 0
        to_col = j+kcentre+1 if j+kcentre+1<Ncols else Ncols
        value = 0.
        for x in xrange(from_row,to_row):
            ki = x + kcentre - i
            for y in xrange(from_col,to_col):
                kj = y + kcentre - j
                value += hough_slice[Rmod,x,y]*kernel[ki]*kernel[kj]
        smoothed_slice[Rmod,i,j] = value/R

    ###   find local max in the 3D sparse Hough space   ###
    #
    # (ii) For every entry in the sparse array which exceeds the
    # threshold verify maximum compared to (rescaled) nearest neighbours
    # (iii) if local max => append to list

    R_ = (R+One)%Three
    Ro = R-One
    Romod = Ro%Three

    for hot in xrange(hotspots_counter[Romod]):
        i = hough_hotspots[Romod,hot,0]
        j = hough_hotspots[Romod,hot,1]
        vote = smoothed_slice[Romod,i,j]
        if vote < circle_thresh: continue
        local_max=True
        for k in xrange(3):#(R_,Romod,Rmod):
            for di in xrange(-1,2):#(-1,0,1):
                for dj in xrange(-1,2):#(-1,0,1):
                    local_max &= vote >= smoothed_slice[k,i+di,j+dj]
                    # there should not be any IndexErrors here
        if local_max: 
            rings[ring_counter,0] = i
            rings[ring_counter,1] = j
            rings[ring_counter,2] = Ro
            ring_counter += 1

    Rmin+=1
    Rmax-=1

    return rings[:ring_counter,...]


#@cython.boundscheck(True)
#@cython.wraparound(True)
@cython.cdivision(True)
cpdef votes2array(RIJ_t [:] votes, #INDX_t [:] votes_sorter,
                  RIJ_t Rmin, RIJ_t Rmax, 
                  RIJ_t Nrows, RIJ_t Ncols, 
                  RIJ_t vote_thresh=1):

    assert vote_thresh>0, 'vote_thresh must be a positive integer, got %s' % \
                                                                   vote_thresh
    cdef:
        INDX_t r, r_, i, j, rij, rij_, Nrads, n, voted4counter, votes_size
        RIJ_t [:,:,:] sparse_3d_Hough
        RIJ_t [:,:] voted4
        coord_t coords

    Rmin-=1
    Rmax+=1
    votes_size = votes.size
    Nrads = Rmax-Rmin+1
    sparse_3d_Hough = np.zeros((Nrads,Nrows,Ncols), RIJ)
    voted4 = np.empty((Nrads,3), RIJ)
    
    n=0
    voted4counter=0
#    rij = votes[votes_sorter[0]]
    rij = votes[0]
    coords = rij2coord(rij)
    r,i,j = coords.r, coords.i, coords.j
    r_ = r-Rmin

    for n in xrange(1,votes_size):
        sparse_3d_Hough[r_,i,j] += 1
        rij_nxt = votes[n]

        if rij_nxt!=rij:
            # need to keep track of those that were updated, such that
            # these could be also set to zero (not all)
            if sparse_3d_Hough[r_,i,j] >= vote_thresh: 
            #if sparse_3d_Hough[r%Nrads,i,j] >= vote_thresh: 
                #voted4[voted4counter] = rij # change rij storage struct?
                votes[voted4counter] = rij # change rij storage struct?
                voted4counter += 1
            #coords = rij2coord(rij_nxt)
            #r_ = coords.r
            #if r_!=r: 
            #    sparse_3d_Hough[r_%Nrads,...] = 0
            rij = rij_nxt
            coords = rij2coord(rij)
            r,i,j = coords.r, coords.i, coords.j
            r_ = r-Rmin


    sparse_3d_Hough[r_,i,j] += 1
    #sparse_3d_Hough[r_%Nrads,i,j] += 1

    Rmin+=1
    Rmax-=1

    return sparse_3d_Hough, votes[:voted4counter]


#@cython.boundscheck(True)
#@cython.wraparound(True)
@cython.cdivision(True)
cpdef smooth_voted4(RIJ_t [:,:,:] sparse_3d_Hough, RIJ_t [:] voted4, 
                    RIJ_t Rmin):

    assert voted4 is not None and sparse_3d_Hough is not None

    cdef:
        RIJ_t x, y, r, r_, i, j, ki, kj, rij, voted4_size
        DTYPE_t value
        RIJ_t Nrows, Ncols, Nrads 
        DTYPE_t [:,:,:] smoothed_hough_array
        DTYPE_t [:] kernel 
        RIJ_t width, ksize
        coord_t coords

    Rmin-=1
    Nrows = sparse_3d_Hough.shape[1]
    Ncols = sparse_3d_Hough.shape[2]
    Nrads = sparse_3d_Hough.shape[0]

    voted4_size = voted4.size
    smoothed_hough_array = np.empty_like(sparse_3d_Hough, DTYPE)
    kernels_list = [get_1d_gaussian_kernel_r(r+Rmin) for r in xrange(Nrads)]
    r_ = 0
    kernel = kernels_list[r_]
    ksize = kernel.size
    width = ksize//2

    for n in xrange(voted4_size):
        rij = voted4[n]
        coords = rij2coord(rij)
        r,i,j = coords.r, coords.i, coords.j
        if r_ != r - Rmin:
            r_ = r - Rmin
            kernel = kernels_list[r_]
            ksize = kernel.size
            width = ksize//2
        from_row = i-width if i-width>0 else 0
        to_row = i+width+1 if i+width+1<Nrows else Nrows
        from_col = j-width if j-width>0 else 0
        to_col = j+width+1 if j+width+1<Ncols else Ncols
        value=0.
        for x in xrange(from_row,to_row):
            ki = x + width - i
            for y in xrange(from_col,to_col):
                kj = y + width - j
                value += sparse_3d_Hough[r_,x,y]*kernel[ki]*kernel[kj]
        smoothed_hough_array[r_,i,j] = value/r

    Rmin+=1

    return smoothed_hough_array


@cython.cdivision(True)
cpdef get_circles(DTYPE_t [:,:,:] sparse_3d_Hough, RIJ_t [:] voted4, 
                  RIJ_t Rmin, DTYPE_t circle_thresh=pi):

    assert sparse_3d_Hough is not None
    cdef:
        RIJ_t rij, i, j, r, r_, n, voted4_size, ring_counter
        int di, dj, dk, dx
        DTYPE_t vote
        RIJ_t [:,:] rings 
        bool local_max
        coord_t coords

    dx=1
    voted4_size = voted4.size
    rings = np.empty((voted4_size,3),RIJ)
    ring_counter = 0
    #
    ###   find local max in the 3D sparse Hough space   ###
    #
    # (ii) For every entry in the sparse array which exceeds the Threshold 
    #   verify maximum compared to (rescaled) nearest neighbours
    # (iii) if local max => append to list

    for n in xrange(voted4_size):
        rij = voted4[n]
        coords = rij2coord(rij)
        r,i,j = coords.r, coords.i, coords.j
        r_ = r - Rmin + 1 # first slab is Rmin-1
        vote = sparse_3d_Hough[r_,i,j]
        if vote < circle_thresh: continue
        local_max=True
        for dk in (-dx,0,dx):
            for di in (-dx,0,dx):
                for dj in (-dx,0,dx):
                    #try: # expect some errors as this is not fully debugged n
                    local_max &= vote >= sparse_3d_Hough[r_+dk, i+di, j+dj]
                    #except IndexError:
                    #    continue
        if local_max: 
            rings[ring_counter,0] = i
            rings[ring_counter,1] = j
            rings[ring_counter,2] = r
            ring_counter += 1

    return rings[:ring_counter,...]


cpdef get_ring_mask(r, dr):
    ''' 
    returns the pixel coordinates of a circle of radius are, centres at the
    origin, assuming:
                     (r - dr)**2 <= i**2 + j**2 <= (r + dr)**2 
    '''
    # generate_mask not considering the borders yet
    y,x = np.ogrid[-r-dr: r+dr+1, -r-dr: r+dr+1]
    ring_mask = abs(x**2+y**2-r**2-dr**2)<=2*r*dr 
    return ring_mask


#@cython.profile(True)
def fitCircle(coords, i, j):
    '''
    find the least squares circle fitting a set of 2D
    points (x,y) based on:
    http://www.scipy.org/Cookbook/Least_Squares_Circle
    '''
        
    def calc_R(centre):
        '''
        calculate the distance of each 2D points from the center c=(xc, yc) 
        '''
        return np.sqrt((coords[:,0]-centre[0])**2 + (coords[:,1]-centre[1])**2)
    
    def cost_fn(centre):
        '''
        calculate the algebraic distance between the 2D points and the mean
        circle centered at c=(xc, yc)
        '''
        Ri = calc_R(centre)
        return Ri - Ri.mean()
    
    # Basic usage of optimize.leastsq
    centre, ier = optimize.leastsq(cost_fn, (i,j))
    
    Ri         = calc_R(centre)
    R          = Ri.mean()
    #residu   = sum((Ri - R)**2)
    #residu2  = sum((Ri**2-R**2)**2)
        
    return centre, R#, residu, residu2


#@cython.profile(True)
cpdef subpxl_circles(RIJ_t [:,:] rings, directed_ridges, 
                   RIJ_t Nrows, RIJ_t Ncols, RIJ_t Rmin, RIJ_t Rmax,
                   RIJ_t thickness=3, DTYPE_t eccentricity=0.):   
    ###   sub-pxl correction (if requested)   ###
    #
    # (i)  for each local max of the 3D Hough space get the ridges pxls within
    #   the annulus of thickness dr
    # (ii) fitEllipse (possibly conditioned on the number of ridge pxls scaled
    #   by the radius)
    # (iii) append to sub-pxl circles (possibly conditioned on eccentricity
    #   and/or mean squared error.
    
    cdef:
        np.ndarray[np.uint8_t, ndim=2, cast=True] img_mask, ring_mask
        np.ndarray[INDX_t, ndim=2] coords # keep numpy array for the newaxis
        INDX_t i,j,r,n, rings_size
        INDX_t row_min,row_max, col_min, col_max
        DTYPE_t Tilt, R
        #tuple cntr, MajAx
        DTYPE_t [:,:] rings_subpxl
        bool subpxled

    rings_size = rings.shape[0]
    img_mask = np.zeros((Nrows,Ncols), np.uint8)
    rings_subpxl = np.empty_like(rings,DTYPE)

    for (i,j) in directed_ridges.iterkeys():
        img_mask[i,j] += 1

    for n in xrange(rings_size):
        i = rings[n,0]
        j = rings[n,1]
        r = rings[n,2]
        subpxled = True
        ring_mask = get_ring_mask(r, thickness)
        #need to acount for the image borders:
        row_min = i-r-thickness if i-r-thickness>0 else 0
        row_max = i+r+thickness+1 if i+r+thickness+1<Nrows else Nrows
        col_min = j-r-thickness if j-r-thickness>0 else 0
        col_max = j+r+thickness+1 if j+r+thickness+1<Ncols else Ncols
        coords = np.transpose(np.nonzero(
            img_mask[row_min:row_max,col_min:col_max] &
            ring_mask[row_min-i+r+thickness:r+thickness+row_max-i,\
                      col_min-j+r+thickness:r+thickness+col_max-j]
            ))
        if eccentricity:
            ## eccentricity larger than zero, that is, requested an ellipse fit
            ## make sure that cv2.fitEllipse does not crash.
            ## but better remove circles which do not have enough points on the
            ## circumference
            if len(coords)>5: 
                cntr, MajAx, Tilt = \
                        fitEllipse(coords[:,np.newaxis,:].astype(np.int32))
                ## skip if the fit results in an eccentric ellipse:
                if np.sqrt(1-MajAx[0]**2/MajAx[1]**2) < eccentricity: 
                    R = sqrt(MajAx[0]*MajAx[1])/2 
                else:
                    subpxled = False
            else:
                subpxled = False
        else:
            # eccentricity==0 that is, need to fit to a circle:
            if len(coords)>3: 
                cntr, R = fitCircle(coords, i-row_min,j-col_min)
            else:
                subpxled = False
        if (R+.5 < Rmin) | (R-.5 > Rmax): 
            subpxled = False

        if not subpxled:
            cntr = (0,0)
            R = 0
        ## skip in case the circle fit is based on less than half its
        ## circumference:
        #if (r<Rmin) or (r>Rmax) or (coords.shape[0]<pi*r) : continue 
        rings_subpxl[n,0] = cntr[0]+row_min
        rings_subpxl[n,1] = cntr[1]+col_min
        rings_subpxl[n,2] = R

    return rings_subpxl


cpdef directed_ridge_detector(DTYPE_t [:,:] Lxx,
                              DTYPE_t [:,:] Lyy, 
                              DTYPE_t [:,:] Lxy, 
                              DTYPE_t [:,:] curv, 
                              DTYPE_t curv_thresh=-20):

    assert Lxx is not None and Lyy is not None and Lxy is not None and\
            curv is not None

    cdef:
        INDX_t i, j,  
        RIJ_t Nrows, Ncols
        char sign
        DTYPE_t cosQ, sinQ
        dict directed_ridges

    Nrows = curv.shape[0]
    Ncols = curv.shape[1]
    directed_ridges = {}

    # iterate over all image entries (principal curv in this case).
    # avoid dealing with the boundaries by iterating over all but the entries
    # on the exterior.
    for i in xrange(1,Nrows-1):         
        for j in xrange(1,Ncols-1):
            #
            ###   Find Ridges:    ###
            #
            # treshold the curvature (note that it should be smaller than...)
            if curv[i,j] > curv_thresh: continue
            # perform non-minimum suppression in the least principal direction
            cosQ,sinQ = least_principal_direction(Lxx[i,j], Lyy[i,j], Lxy[i,j])
            if fabs(cosQ) > cos_q_pi:
                if (curv[i,j] >= curv[i,j+1]) | (curv[i,j] >= curv[i,j-1]):
                    continue
            elif fabs(sinQ) > cos_q_pi:
                if (curv[i,j] >= curv[i+1,j]) | (curv[i,j] >= curv[i-1,j]):
                    continue
            elif copysign(1,sinQ) == copysign(1,cosQ):
                if (curv[i,j] >= curv[i+1,j+1]) | (curv[i,j] >= curv[i-1,j-1]):
                    continue
            elif copysign(1,sinQ) != copysign(1,cosQ):
                if (curv[i,j] >= curv[i-1,j+1]) | (curv[i,j] >= curv[i+1,j-1]):
                    continue
           
            # add the least principal direction to the ridge sparse array
            directed_ridges[i,j] = cosQ, sinQ
                    
    return directed_ridges


class RidgeHoughTransform():
    '''
    Perform circle Hough transform based on ridge binary image (direction
    aided)
    Expects an (float-type) image input at instantiation
    '''

    params = {
            ## Binarising image parameters:
            'sigma':1.8, # GX1920 full resolution; 10ms exp, gain13
            'ksize':5,
            'curv_thresh':-25,
            ## Hough transform parameters:
            'Rmin':7,
            'Rmax':85,
            'vote_thresh':3,
            'circle_thresh': 0.33 * 2 * pi,  #2.4 #3.2
            ## sub-pxling parameters:
            'dr':3,  # half-thickness of the ring to fit to an ellipse
	    'eccentricity':0
            }
    deriv = {}


    def __init__(self, img=None):
        self.img = img
        #self.Nrows, self.Ncols = img.shape


    def img_preprocess(self):
        '''
        Blurs the image, calculates its second derivatives (Hessian matrix
        components), and its least principal curvature
        '''
        assert self.img.dtype==DTYPE
        ksize = self.params['ksize']

        if self.params['sigma']:
            Blurred = cv2.GaussianBlur(self.img,(0,0),self.params['sigma'])
        else:
            Blurred = self.img
        self.deriv['Lxx'] = cv2.Sobel(Blurred, cv2.CV_32F, 2,0, ksize=ksize)
        self.deriv['Lyy'] = cv2.Sobel(Blurred, cv2.CV_32F, 0,2, ksize=ksize)
        self.deriv['Lxy'] = cv2.Sobel(Blurred, cv2.CV_32F, 1,1, ksize=ksize)
        ## Could possibly mask the image based on Laplacian (rather than
        ## minimal principal curvature).
        self.deriv['principal_curv'] = least_principal_curvature(\
                self.deriv['Lxx'], self.deriv['Lyy'], self.deriv['Lxy'])

        
    def rings_detection(self):
        assert self.params['Rmin']>=3
        assert self.params['Rmin'] <= self.params['Rmax']
        ht_out = ridge_circle_hough_transform(self.deriv['Lxx'], 
                self.deriv['Lyy'], self.deriv['Lxy'],
                self.deriv['principal_curv'], self.params['curv_thresh'],
                self.params['Rmin'], self.params['Rmax'])
        rings = votes2rings(ht_out['votes'], 
                                 self.params['Rmin'], self.params['Rmax'],
                                 self.img.shape[0], self.img.shape[1], 
                                 self.params['vote_thresh'], 
                                 self.params['circle_thresh'])
        rings_subpxl = subpxl_circles(rings, 
                                 ht_out['directed_ridges'], 
                                 self.img.shape[0], self.img.shape[1], 
                                 self.params['Rmin'], self.params['Rmax'],
                                 self.params['dr'],
                                 self.params['eccentricity'])
        self.output = {'rings' : np.asarray(rings),
                       'rings_subpxl' : np.asarray(rings_subpxl),
                       }


    def directed_ridge_detector(self):
        directed_ridges = directed_ridge_detector(self.deriv['Lxx'], 
                self.deriv['Lyy'], self.deriv['Lxy'],
                self.deriv['principal_curv'], self.params['curv_thresh'])
        self.output = {'directed_ridges' : directed_ridges,
                       }


    def debugging_rings_detection(self):
        assert self.params['Rmin']>=3
        assert self.params['Rmin'] <= self.params['Rmax']
        ht_out = ridge_circle_hough_transform(self.deriv['Lxx'], 
                self.deriv['Lyy'], self.deriv['Lxy'],
                self.deriv['principal_curv'], self.params['curv_thresh'],
                self.params['Rmin'], self.params['Rmax'])
        array_out = votes2array(ht_out['votes'], 
                                self.params['Rmin'], self.params['Rmax'], 
                                self.img.shape[0], self.img.shape[1],
                                self.params['vote_thresh'])
        smooth_out = smooth_voted4(array_out[0], array_out[1], 
                                   self.params['Rmin'])
        rings = get_circles(smooth_out, array_out[1], self.params['Rmin'],
                            self.params['circle_thresh'])
        rings_subpxl = subpxl_circles(rings, ht_out['directed_ridges'],
                                      self.img.shape[0], self.img.shape[1],
                                      self.params['Rmin'], self.params['Rmax'],
                                      self.params['dr'],
                                      self.params['eccentricity'])

        self.output = {'directed_ridges': ht_out['directed_ridges'],
                       'votes': ht_out['votes'],
                       'sparse_hough_array' : array_out[0],
                       'voted4' : array_out[1],
                       'sparse_hough_smooth' : smooth_out,
                       'rings' : rings,
                       'rings_subpxl' : rings_subpxl}



import numpy
import scipy
import matplotlib
import random
import skimage
from skimage import io

def fcc(X, cen):
     [K, nu] = cen.shape
     [m,n] = X.shape
     id1 = numpy.zeros((m,1))
     for i in range(m):
         min = 0
         for j in range(n):
             min = min + (X[i][j]-cen[0][j])**2
         id1[i][0] = 0
         for k in range(K):
             s = 0
             for h in range(n):
                 s = s + (X[i][h]-cen[k][h])**2 
             if(s<min):
                 min = s
                 id1[i,0] = k
        
     return id1
     
def cc(X, id1, K):
    [m, n] = X.shape
    centroids = numpy.zeros((K,n))
    for e in range(K):
        cnt = 0
        for f in range(m):
            if((id1[f])==e):
                centroids[e,:] = centroids[e,:] + X[f,:]
                cnt = cnt + 1
        centroids[e,:] = centroids[e,:] / cnt
    return centroids

def runkmeans(X, cen, max_iters):
    [m, n] = X.shape
    [K, nu] = cen.shape
    centroids = cen
    previous_centroids = centroids
    id1 = numpy.zeros((m,1))
    
    for w in range(max_iters):
        id1 = fcc(X, centroids)
        centroids = cc(X, id1, K)
        
        
    return (centroids, id1)


def kmeans(im, colors):
    im1 = skimage.img_as_ubyte(im, force_copy=False)
    im2 = im1/255
    
    [x, y, z] = im2.shape
    reshaped_im = im2.reshape(x*y,z)
    
    K = colors
    max_iters = 10
    
    cen = numpy.zeros((K,z))
    id = numpy.linspace(1,y*x,x*y)
    random.shuffle(id)
    j = id[0:K]
    
    l = 0
    for t in range(K):
        a = j[l]
        cen[t,:] = reshaped_im[a,:]
        l = l+1
    
    [centroids, idx] = runkmeans(reshaped_im, cen, max_iters)
    idx = fcc(reshaped_im, centroids)
    
    imrecovered = numpy.zeros((x*y,z))
    [p, q] = idx.shape
    
    for u in range(p):
        a = idx[u][0]
        imrecovered[u,:] = centroids[a,:]
        
    imfinal = imrecovered.reshape(x,y,z)
    return imfinal
    
    
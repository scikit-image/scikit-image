# Include necessary files and libraries
import numpy as np
from skimage import morphology as skm
from matplotlib import pyplot as plt
from scipy.ndimage import distance_transform_cdt

# Add readme
# Reference insipration and sources
# Move _cv_main to a new file
# 

def _cv_curvature(phi):
    m,n = phi.shape
    P = np.hstack((np.zeros((m,1)),phi,np.zeros((m,1))))
    m,n = P.shape
    P = np.vstack((np.zeros((1,n)),P,np.zeros((1,n))))
    fy = P[2:,1:-1] - P[0:-2,1:-1]
    fx = P[1:-1,2:] - P[1:-1,0:-2]
    fyy = P[2:,1:-1] + P[0:-2,1:-1] - 2 * phi
    fxx = P[1:-1,2:] + P[1:-1,0:-2] - 2 * phi
    fxy = .25*(P[2:,2:]+P[:-2,:-2]-P[:-2,2:]-P[2:,:-2])
    G = np.sqrt(fx**2+fy**2)
    K = (fxx*fy**2-2*fxy*fx*fy+fyy*fx**2)/(np.power(fx**2+fy**2,1.5)+1e-10)
    KG = K*G
    return KG/np.max(np.abs(KG))
    

def _cv_heavyside(x,eps=1.):
    return 0.5*(1.+2./np.pi*np.arctan(x/eps))

def _cv_delta(x,eps=1.):
    return eps/(eps**2+x**2)
    
def _cv_difference_from_average_term(U0,Hphi,lambda_pos,lambda_neg):
    H = Hphi
    Hinv = 1.-H
    Hsum = np.sum(H)
    Hinvsum = np.sum(Hinv)
    c1 = np.sum(U0*H)
    c2 = np.sum(U0*Hinv)
    if(Hsum!=0):
        c1 = c1/Hsum
    if(Hinvsum!=0):
        c2 = c2/Hinvsum
    return lambda_pos*(U0-c1)**2*H+lambda_neg*(U0-c2)**2*Hinv

def _cv_edge_length_term(phi,mu):
    toret = _cv_curvature(phi)
    return mu*toret
    
def _cv_energy(U0,phi,mu,lambda1,lambda2,heavyside=_cv_heavyside):
    H = heavyside(phi)
    return np.sum(_cv_difference_from_average_term(U0,H,lambda1,lambda2)+_cv_edge_length_term(phi,mu))
    
def _cv_signed_distance_from_zero_level(phi):
    return distance_transform_cdt(phi>0)-distance_transform_cdt(phi<=0)

def _cv_initial_shape(U0):
    print U0.shape
    yv = np.arange(U0.shape[0]).reshape(U0.shape[0],1)
    xv = np.arange(U0.shape[1])
    return (np.sin(np.pi/100.*yv)*(np.sin(np.pi/100.*xv)))

def _cv_calculate_variation(U0,phi,mu,lambda1,lambda2,heavyside=_cv_heavyside):
    H = heavyside(phi)
    Hinv = 1.-H
    Hsum = abs(np.sum(H))
    Hinvsum = abs(np.sum(Hinv))
    c1 = np.sum(U0*H)
    c2 = np.sum(U0*Hinv)
    if(Hsum!=0):
        c1 = c1/Hsum
    if(Hinvsum!=0):
        c2 = c2/Hinvsum
    return _cv_edge_length_term(phi,mu)-lambda1*(U0-c1)**2+lambda2*(U0-c2)**2

def chan_vese(U0,mu,lambda1,lambda2,tol=1e-8,maxiter=10000,dt=1):
    phi = _cv_initial_shape(U0)
    i=0
    delta=1e18
    old_energy = _cv_energy(U0,phi,mu,lambda1,lambda2)
    energies=[]
    while(delta>tol and i<maxiter or i<1):
        dphi_dt = _cv_calculate_variation(U0,phi,mu,lambda1,lambda2)
        phi+=dphi_dt*dt
        new_energy = _cv_energy(U0,phi,mu,lambda1,lambda2)
        energies.append(old_energy)
        delta=np.abs(new_energy-old_energy)
        old_energy=new_energy
        i+=1
    return phi>0,energies

def _cv_main():
    from skimage.data import camera
    from skimage.data import coins
    #U0 = plt.imread("rgb_tile_014_i01_j05 - crop.png")[:,:,0].astype('float')-0.5
    U0 = coins().astype('float')/255.
    print np.max(U0)
    cv = chan_vese(U0,mu=0.8,lambda1=1,lambda2=1,tol=1,maxiter=15,dt=100)
    print ("Chan-Vese algorithm finished after "+str(len(cv[1]))+" iterations.")
    print cv[1]
    plt.imshow(cv[0])
    plt.colorbar()
    plt.show()
    plt.plot(cv[1])
    plt.show()
    return
    
if __name__ == '__main__':
    _cv_main()
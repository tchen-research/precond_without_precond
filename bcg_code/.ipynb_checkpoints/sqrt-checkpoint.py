import numpy as np
import scipy as sp

from .lanczos import *
from .nystrom_pcg import *

def get_sqrt_sols(A,B,iters,reorth=0):
    """
    Output solutions of block-CG applied to (A,B) for all values of iters,mus
    """
    
    d,l = np.shape(B)
    t_max = max(iters)
    
    Qt,Qtp1,A_,B_,B_0 = block_lanczos(A,B,t_max,reorth=reorth)

    #T_band = get_block_tridiag(A_,B_)
    T_band = get_banded_block_tridiag(A_,B_)[:l+1]

    
    x = np.full((len(iters),d,l),np.nan)
    for i,t in enumerate(iters):
        if t==0:
            x[i,:] = 0
            continue

        #Θ,S = np.linalg.eigh(T_band[:l*t,:l*t]) 
        Θ,S = sp.linalg.eig_banded(T_band[:,:l*t])
        
        e1 = np.zeros((t,1))
        e1[0,0] = 1
        E1 = np.kron(e1,B_0)
        
        x[i] = Qt[:,:l*t]@(S@(np.sqrt(Θ)[:,None]*((S.T[:,:l]@B_0))))

    return x
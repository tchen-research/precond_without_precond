import numpy as np
import scipy as sp

from .lanczos import *
from .nystrom_pcg import *

def get_BCG_sols(A,B,iters,μs,reorth=True,reorth_max=np.inf):
    """
    Output solutions of block-CG applied to (A,B) for all values of iters,mus
    """
    
    d,l = np.shape(B)
    t_max = max(iters)
    
    Qt,Qtp1,A_,B_,B_0 = block_lanczos(A,B,t_max,reorth=reorth,reorth_max=reorth_max)

    T_band = get_banded_block_tridiag(A_,B_)
    I_band = np.zeros((2*l+1,l*t_max))
    I_band[l] = 1

    x = np.full((len(iters),len(μs),d),np.nan)
    for i,t in enumerate(iters):
        if t==0:
            x[i,:] = 0
            continue

        e1 = np.zeros((t,1))
        e1[0,0] = 1
        E1 = np.kron(e1,B_0)
            
        for j,μ in enumerate(μs):
            Tμ_band = T_band[:,:l*t]+μ*I_band[:,:l*t]
            x[i,j] = Qt[:,:l*t]@sp.linalg.solve_banded((l,l),Tμ_band,E1[:,0])

    return x

def get_PCG_sols(A,B,P,iters,reorth=True):
    """
    Output solutions of preconditioned-CG applied to (A,B,P) for all values of iters,mus
    """

    d,l = np.shape(B)
    t_max = max(iters)
    
    x = np.full((len(iters),d),np.nan)

    PAP = sequence_prod([P,A,P])
    Qt,Qtp1,A_,B_,B_0 = block_lanczos(PAP,P@B,t_max,reorth=reorth)

    T_band = get_banded_block_tridiag(A_,B_)
    for i,t in enumerate(iters):
        if t==0:
            x[i] = 0
            continue

        e1 = np.zeros((t,1))
        e1[0,0] = 1
        E1 = np.kron(e1,B_0)
        
        x[i] = P@(Qt[:,:l*t]@sp.linalg.solve_banded((l,l),T_band[:,:l*t],E1[:,0]))

    return x
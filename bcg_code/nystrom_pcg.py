import numpy as np
import scipy as sp
from .lanczos import *

def nystrom(A,Ω,s):
    """
    Output the Nystrom approximation A<K_s> where K_s = [Ω,AΩ,...,A^{s-1}Ω]
    """

    d,b = np.shape(Ω)
    Qt,Qtp1,A_,B_,B_0 = block_lanczos(A,Ω,s,reorth=s)

    T = get_block_tridiag(A_,B_) # this could be more efficient beacuse banded
    AQ = Qt@T
    AQ[:,-b:] += Qtp1@B_[-1]
    
    C = np.linalg.cholesky(T) # this is also banded
    Z = sp.linalg.solve_triangular(C,AQ.T,lower=True).T # triangular solve
    U,σ,_ = np.linalg.svd(Z,full_matrices=False) # SVD

    return U,σ**2

class deflation_precond12():
    """
    Build square root of deflation preconditioner P^{-1} = (θ+μ) U(D+μI)^{-1}U^T + (I - UU^T)
    """

    def __init__(self,U,D,θ,μ):

        self.U = U
        self.D = D
        self.θ = θ
        self.μ = μ
        self.dtype = D.dtype
        self.shape = (U.shape[0],U.shape[0])

    def __matmul__(self,x):

        Utx = self.U.T@x
        return np.sqrt(self.θ+self.μ)*self.U@(np.diag(1/np.sqrt(self.D+self.μ))@Utx) + x - self.U@Utx

class deflation_precond12_inv():
    """
    Build square root of deflation preconditioner P = (θ+μ)^{-1} U(D+μI)U^T + (I - UU^T)
    """

    def __init__(self,U,D,θ,μ):

        self.U = U
        self.D = D
        self.θ = θ
        self.μ = μ
        self.dtype = D.dtype
        self.shape = (U.shape[0],U.shape[0])

    def __matmul__(self,x):

        Utx = self.U.T@x
        return (1/np.sqrt(self.θ+self.μ))*self.U@(np.diag(np.sqrt(self.D+self.μ))@Utx) + x - self.U@Utx
   
class sequence_prod():

    def __init__(self,matrices):
        self.matrices = matrices
        self.dtype = matrices[0].dtype
        self.shape = (matrices[0].shape[0],matrices[-1].shape[-1])

    def __matmul__(self,x):
        y = np.copy(x)
        for M in self.matrices:
            y = M@y
        return y

    def matvec(self,x):
        return self.__matmul__(x)

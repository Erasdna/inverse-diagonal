from src.cg import cg
import numpy as np
from src.typealias import RealArray
from multiprocessing import cpu_count
from pqdm.processes import pqdm

def MC_step(z : RealArray,A : RealArray,precon_inv : RealArray,tol : float) -> RealArray:
    """MC step: calculates (A^-1 @ z) * z 

    Args:
        z (RealArray): Sample Rademacher vector
        A (RealArray): Hermitian matrix
        precon_inv (RealArray): Inverse of precondtitioner for the cg method
        tol (float): tolerance of the cg method

    Returns:
        (RealArray): Result of the Monte Carlo step 
    """
    return cg(A,z,precon_inv,tol) * z

def MC(A: RealArray,precon_inv : RealArray,tol : float,N : int, mat: RealArray = None) -> RealArray:
    """Monte Carlo estimate of the diagonal of (A^-1 - mat)

    Args:
        A (RealArray): Hermitian matrix A to be invertes
        precon_inv (RealArray): Inverse of preconditioner for cg 
        tol (float): Tolerance of the cg method
        N (int): Number of Monte Carlo steps
        mat (RealArray): Matrix of same dimension as A

    Returns:
        (RealArray): Matrix of size (N,N_a) containing Monte Carlo estimate for each 1<n<N
    """
    z_mat=np.random.choice([-1,1],size=(A.shape[0],N))
    input=[[el,A,precon_inv,tol] for el in z_mat.T]
    ret=np.array(pqdm(input,MC_step,n_jobs=cpu_count(),argument_type='args'))
    div=1/np.arange(1,N+1)
    if mat is not None:
        ret=-((mat @ z_mat) * z_mat).T
    return np.cumsum(ret,axis=1)*div[:,None]
    
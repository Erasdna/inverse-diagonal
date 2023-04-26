from src.cg import cg, chol_solve
import numpy as np
from src.typealias import RealArray
from multiprocessing import cpu_count
from pqdm.processes import pqdm
from scipy.sparse import csr_matrix

def MC_step(z : RealArray, A : RealArray, L : RealArray, LT, tol : float, mat=None) -> RealArray:
    """MC step: calculates (A^-1 @ z) * z 

    Args:
        z (RealArray): Sample Rademacher vector
        A (RealArray): Hermitian matrix
        precon_inv (RealArray): Inverse of precondtitioner for the cg method
        tol (float): tolerance of the cg method

    Returns:
        (RealArray): Result of the Monte Carlo step 
    """
    y = cg(A,z,L,LT,tol)
    if mat is not None:
        y -= mat @ z
    return y * z

def MC(A: RealArray, L: RealArray ,tol : float,N : int, mat: RealArray = None) -> RealArray:
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
    LT = csr_matrix(L.T)
    #z_mat=np.random.choice([-1,1],size=(A.shape[0],N))
    #input=[[el, A , L, LT, tol, mat] for el in z_mat.T]
    #print(cg(A, np.random.random(A.shape[0]),L, np.random.random(A.shape[0]), tol).shape,)
    ins = ((np.random.choice([-1,1],size=A.shape[0]), A, L, LT, tol, mat) for _ in range(N))
    ret=np.array(pqdm(ins, MC_step, n_jobs=cpu_count(),argument_type='args'))
    #ret = np.array([MC_step(el, A, L,LT, tol) for el in z_mat.T])
    div=1/np.arange(1,N+1)
    return np.cumsum(ret,axis=1)*div[:,None]
    
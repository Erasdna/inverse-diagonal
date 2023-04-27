from typing import Callable
from src.cg import cg, chol_solve
import numpy as np
from src.typealias import RealArray
from multiprocessing import cpu_count
from pqdm.processes import pqdm
from scipy.sparse import csr_matrix
from scipy.stats import norm

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
    





def MC_lanzos_control_variates(Z: Callable[[RealArray], RealArray], Y: Callable[[RealArray], RealArray], mu_y: RealArray, N:int, n: int, clip=False):
    def mc_cv_step(Z: Callable[[RealArray], RealArray], Y: Callable[[RealArray], RealArray]):
        z = np.random.choice([-1,1],size=n, replace=True)
        return Z(z), Y(z)
    #ins = ((Z, Y) for _ in range(N))
    #ret=np.array(pqdm(ins, mc_cv_step, n_jobs=cpu_count(),argument_type='args'))
    ret = np.array([mc_cv_step(Z, Y) for _ in range(N)])
    #ret = np.array([MC_step(el, A, L,LT, tol) for el in z_mat.T])
    est_mu_z = np.mean(ret[:, 0], axis=0)
    est_mu_y = np.mean(ret[:, 1], axis=0)
    est_sigma_y = np.mean((ret[:, 1] - mu_y) ** 2, axis=0) # TODO: We tecnically know this one already, but probably hard to compute
    est_sigma_zy = np.mean((ret[:, 0] - est_mu_z) * (ret[:, 1] - mu_y), axis=0)
    alpha = est_sigma_zy / est_sigma_y
    if clip:
        sigma_alpha = np.mean((((ret[:, 1] - mu_y) / (ret[:, 0] - est_mu_z)) - alpha) ** 2, axis=0)
        c_alpha = norm.ppf(1 - (0.01 / 2))
        alpha = np.where(np.abs(alpha - 1) < c_alpha * np.sqrt(sigma_alpha / N), 1, alpha)
        print(f"alpha: {alpha}")
    return est_mu_z - alpha * (est_mu_y - mu_y), est_mu_z - (est_mu_y - mu_y)
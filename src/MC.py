from typing import Callable
from src.cg import cg, chol_solve
import numpy as np
from src.typealias import RealArray
from multiprocessing import cpu_count
from pqdm.processes import pqdm
from scipy.sparse import csr_matrix
from scipy.stats import norm

def MC_step(z : RealArray, 
            A : RealArray, 
            L : RealArray, 
            LT: RealArray, 
            tol : float, 
            W=None, 
            WT=None) -> RealArray:
    """MC step: calculates (A^-1 @ z) * z 

    Args:
        z (RealArray): Sample Rademacher vector
        A (RealArray): Hermitian matrix
        L (RealArray): Left Cholesky decomposition
        LT (RealArray): Transpose of L
        tol (float): tolerance of the cg method
        W (RealArray): Lanczos metho matrix W
        WT (RealArray): Transpose of W


    Returns:
        (RealArray): Result of the Monte Carlo step 
    """
    y = cg(A,z,L,LT,tol)
    if W is not None and WT is not None:
        return y * z , (W @ (WT @ z)) * z
    else:
        return y * z

def MC(A: RealArray, 
       L: RealArray, 
       LT:RealArray, 
       tol : float, 
       N : int) -> RealArray:
    """Monte Carlo estimate of the diagonal of (A^-1 - mat)

    Args:
        A (RealArray): Hermitian matrix A to be invertes
        L (RealArray): Left Cholesky decomposition
        LT (RealArray): Transpose of L        tol (float): Tolerance of the cg method
        N (int): Number of Monte Carlo steps

    Returns:
        (RealArray): Matrix of size (N,N_a) containing Monte Carlo estimate for each 1<n<N
    """
    ins = ((np.random.choice([-1,1],size=A.shape[0]), A, L, LT, tol) for _ in range(N))
    ret=np.array(pqdm(ins, MC_step, n_jobs=cpu_count(),argument_type='args'))
    div=1/np.arange(1,N+1)
    return np.cumsum(ret,axis=0)*div[:,None]
    

def MC_lanzos_control_variates(A: RealArray, 
                               L:RealArray, 
                               LT: RealArray, 
                               W_diag: RealArray, 
                               W: RealArray, 
                               WT: RealArray,  
                               N:int, 
                               tol: float, 
                               clip=False):
    """Parallel implementation of the combined Mc Lanczos method. Calculates an estimate of the diagonal of the inverse of the 
    matrix A for k Lanczos iterations and N MC iterations. Returns estimate with optimal variance control parameter
    and with fixed variance control parameter.

    Args:
        A (RealArray): Matrix of which we wish to estimate the diagonal of the inverse
        L (RealArray): Lower triangular part of preconditioner
        LT (RealArray): Upper triangular part of preconditioner
        W_diag (RealArray): Estimated diagonal from the Lanczos only method
        W (RealArray): Variance reducing matrix used for MC
        WT (RealArray): Transpose of W
        N (int): Number of MC steps
        tol (float): Tolerance of CG method
        clip (bool, optional): Clip. Defaults to False.

    Returns:
        optimal (list(RealArray)): List of size N containing N_A elements, estimate using optimal variance reducing parameter 
        fixed (list(RealArray)): List of size N containing N_A elements, estimate fixing the variance reducing parameter to 1
        mean_alphas (list(float)): List of size N containing the mean value of the matrix alpha at each MC step
    """
    #Sample Rademacher vectors to create input to parallel MC
    ins = ((np.random.choice([-1,1],size=A.shape[0],replace=True), A, L, LT, tol, W, WT) for _ in range(N))
    #Run MC with variate reduction
    ret=np.array(pqdm(ins, MC_step, n_jobs=cpu_count(), argument_type='args'))
    
    #We store MC estimates for 1<n<N 
    fixed=np.zeros((N,A.shape[0]))
    optimal=np.zeros((N,A.shape[0]))
    mean_alpha=np.zeros(N)
    
    for i in range(1,N+1):
        est_mu_z = np.mean(ret[:i, 0], axis=0)
        est_mu_y = np.mean(ret[:i, 1], axis=0)
        est_sigma_y = np.mean((ret[:i, 1] - W_diag) ** 2, axis=0)
        est_sigma_zy = np.mean((ret[:i, 0] - est_mu_z) * (ret[:i, 1] - W_diag), axis=0)
        alpha = est_sigma_zy / est_sigma_y
        if clip:
            sigma_alpha = np.mean((((ret[:i, 1] - W_diag) / (ret[:i, 0] - est_mu_z)) - alpha) ** 2, axis=0)
            c_alpha = norm.ppf(1 - (0.01 / 2))
            alpha = np.where(np.abs(alpha - 1) < c_alpha * np.sqrt(sigma_alpha / N), 1, alpha)
        
        #Calculate estimates
        fixed[i-1,:]=est_mu_z - (est_mu_y - W_diag)
        optimal[i-1,:]=est_mu_z - alpha * (est_mu_y - W_diag)
        mean_alpha[i-1]=np.mean(np.abs(alpha))
    return optimal, fixed, mean_alpha

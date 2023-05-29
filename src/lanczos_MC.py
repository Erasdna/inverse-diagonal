from src.cg import cg
from src.MC import MC_lanzos_control_variates
import scipy.sparse as sparse
from src.lanczos import lanczos_estimate
from src.typealias import RealArray

def lanczos_MC(A : RealArray, 
               G : RealArray, 
               U : RealArray, 
               alpha : float, 
               beta : float, 
               k : int, 
               N : int) -> tuple[RealArray,RealArray,RealArray]:
    """Calculates the combined MC - Lanczos estimate of the diagonal of A

    Args:
        A (RealArray): Sparse matrix which diagonal will be estimated
        G (RealArray): Preconditioner of A
        U (RealArray): Orthonormal Krylov subspace basis matrix
        alpha (float): Diagonal of Lanczos Hessenberg matrix
        beta (float): First diagonal of Lanczos Hessenberg matrix
        k (int): Lanczos iterations
        N (int): MC iterations

    Returns:
        tuple[RealArray,RealArray,RealArray]: 
            - Estimate of the diagonal of the inverse using fixed control variate
            - Estimate of the diagonal of the inverse using estimate of optimal control variate
            - Average estimated control variate alpha 
    """
    est_diag_l, W=lanczos_estimate(G,U,alpha,beta,k)
        
    GT = sparse.csr_matrix(G.T)
    
    est_diag_comb, est_diag_comb_fix, optimal_alpha= MC_lanzos_control_variates(A, G,GT , est_diag_l, W @ W.T, N,1e-9, clip=False)
 
    return est_diag_comb,est_diag_comb_fix, optimal_alpha
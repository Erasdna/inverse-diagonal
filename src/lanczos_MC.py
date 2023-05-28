from src.cg import cg
from src.MC import MC_lanzos_control_variates
import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
from src.lanczos import lanczos_estimate

def lanczos_MC(A,G,U,alpha,beta,k,N):
    est_diag_l, W=lanczos_estimate(G,U,alpha,beta,k)
        
    GT = sparse.csr_matrix(G.T)
    
    est_diag_comb, est_diag_comb_fix, optimal_alpha= MC_lanzos_control_variates(A, G,GT , est_diag_l, W @ W.T, N,1e-9, clip=False)
    #Z = lambda z: cg(A, z, G, GT, 1e-9) * z
    #Y = lambda z: (W @ (W.T @ z)) * z
    #est_diag_comb, est_diag_comb_fix= MC_lanzos_control_variates_2(Z, Y, est_diag_l, N, A.shape[0], clip=True)
    
    return est_diag_comb,est_diag_comb_fix, optimal_alpha
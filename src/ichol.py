import numpy as np
import scipy.sparse as sparse
from scipy.io import mmread
from src.typealias import RealArray

def incomplete_cholesky(A : RealArray) -> RealArray:
    """Method for computing the incomplete Cholesky decomposition of the matrix A

    Args:
        A (RealArray): Sparse matrix A for which Cholesky is computed 

    Returns:
        RealArray: Matrix L such that LL^T = A, lower diagonal cholesky decompistion with same sparsity structure as A
    """
    n = A.shape[0]
    L = sparse.lil_matrix(A.shape)
    for k in range(n):
        L[k, k] = np.sqrt(A[k, k] - L[k, :k].power(2).sum())
        i = A[(k+1):, k].nonzero()[0] + (k+1)
        L[i[:, None], k] = (A[i, k].todense() - (L[i, :k] @ L[k, :k].T if k > 0 else 0)).reshape(len(i), 1) / L[k, k]
    return L

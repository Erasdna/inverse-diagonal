import numpy as np
import scipy.sparse as sparse
from scipy.io import mmread


        

def incomplete_cholesky(A):
    n = A.shape[0]
    L = sparse.lil_matrix(A.shape)
    for k in range(n):
        L[k, k] = np.sqrt(A[k, k] - L[k, :k].power(2).sum())
        i = A[(k+1):, k].nonzero()[0] + (k+1)
        L[i[:, None], k] = (A[i, k].todense() - (L[i, :k] @ L[k, :k].T if k > 0 else 0)).reshape(len(i), 1) / L[k, k]
    return L


if __name__ == "__main__":
    m = mmread("nos3/nos3.mtx")
    L=ichol(m)
    ind=L.nonzero()
    for i in range(len(ind[0])):
        if ind[0][i]<ind[1][i]:
            print("Oooops")
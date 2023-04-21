import numpy as np
import scipy.sparse as sparse
from scipy.io import mmread

def ichol1(A):
    #Error if not square? Assume square ufn
    A=A.tocsr()
    shape=A.shape
    n=shape[0]
    L=sparse.dok_matrix(shape)
    #Perform Cholesky
    for k in range(n):
        L[k,k]=np.sqrt(A[k,k])
        L[k+1:n,k]=(1/L[k,k])*A[k+1:n,k]
        
        #Identify non zero indicies in submatrix of A
        ind=A[k+1:n,k+1:n].nonzero()
        subL=L[k+1:n,k].copy()
        #Only calculate products in positions where A is non-zero
        L_prod=subL[ind[0]].multiply(subL[ind[1]])
        #Useful for recovering matrix structure
        tmp=sparse.dok_matrix((n-k-1,n-k-1))
        tmp[ind]=L_prod.T
        A[k+1:n,k+1:n]-=tmp

    return L

def ichol(A):
    #Error if not square? Assume square ufn
    A=A.tocsr()
    shape=A.shape
    n=shape[0]
    L=sparse.dok_matrix(A)
    print(len(A.nonzero()[0]))
    #Perform Cholesky
    for k in range(n):
        L[k,k]=np.sqrt(L[k,k])
        nonzero_1 = L[k+1:, k].nonzero()[0] + (k + 1)
        if len(nonzero_1) == 1:
            (nonzero_1, ) = nonzero_1
            L[nonzero_1, k] /= L[k, k]
            nonzero_1 = [nonzero_1]
        else:
            L[nonzero_1[:, np.newaxis], k] /= L[k, k]
            #L[k+1:n,k]=(1/L[k,k])*A[k+1:n,k]
        for j in nonzero_1:
            #print(L[j:n, j].nonzero())
            nonzero_2 = L[j:, j].nonzero()[0] + j
            if len(nonzero_2) == 1:
                (nonzero_2, ) = nonzero_2
                L[nonzero_2, j] -= L[nonzero_2, k] * L[j, k]
            else:
                L[nonzero_2[:, np.newaxis], j] -= (L[nonzero_2, k] * L[j, k]).reshape(len(nonzero_2), 1)

    L = sparse.tril(L)
        
        ##Identify non zero indicies in submatrix of A
        #ind=A[k+1:n,k+1:n].nonzero()
        #subL=L[k+1:n,k].copy()
        ##Only calculate products in positions where A is non-zero
        #L_prod=subL[ind[0]].multiply(subL[ind[1]])
        ##Useful for recovering matrix structure
        #tmp=sparse.dok_matrix((n-k-1,n-k-1))
        #tmp[ind]=L_prod.T
        #A[k+1:n,k+1:n]-=tmp
    return L
        

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
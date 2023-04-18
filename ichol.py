import numpy as np
import scipy.sparse as sparse

def ichol(A):
    #Error if not square? Assume square ufn
    A=A.tocsr()
    shape=A.shape
    n=shape[0]
    L=sparse.dok_matrix(shape)
    #Perform Cholesky
    for k in range(n-1):
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
        
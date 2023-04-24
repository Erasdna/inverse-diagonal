from src.MC import MC
from scipy.io import mmread, loadmat
from src.ichol import incomplete_cholesky
from scipy.sparse.linalg import inv, splu
from scipy.sparse import csr_matrix
import numpy as np
import matplotlib.pyplot as plt


if __name__=="__main__":
    A = loadmat("nos3/nos3.mat")["Problem"][0][0][1]
    L=incomplete_cholesky(A.tocsc())
    print("Finished calculating L")
    #precon_inv=inv(L.T) @ inv(L)
    M = csr_matrix(L @ L.T)
    print(L.nnz / (960 ** 2))
    print(inv(M).nnz / (960 ** 2))
    tol=1e-9
    N=900
    res=MC(A=A,L=L.todense(), tol=tol,N=N,mat=None)
    
    diagA=inv(A).diagonal()
    diff=np.linalg.norm(res-diagA[:],axis=1)
    
    fig,ax=plt.subplots()
    ax.semilogy(np.arange(1,N+1),diff/np.linalg.norm(diagA),lw=2, marker="^")
    ax.grid()
    ax.set_xlabel("N")
    ax.set_ylabel("Relative error")
    plt.savefig("MC.png")

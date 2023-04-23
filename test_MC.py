from src.MC import MC
from scipy.io import mmread
from src.ichol import incomplete_cholesky
from scipy.sparse.linalg import inv
import numpy as np
import matplotlib.pyplot as plt


if __name__=="__main__":
    A = mmread("nos3/nos3.mtx")
    L=incomplete_cholesky(A.tocsc())
    print("Finished calculating L")
    precon_inv=inv(L.T) @ inv(L)
    tol=1e-9
    N=900
    res=MC(A=A,precon_inv=precon_inv,tol=tol,N=N,mat=np.eye(N=A.shape[0]))
    diagA=inv(A).diagonal()
    diff=np.linalg.norm(res-diagA[:],axis=1)
    
    fig,ax=plt.subplots()
    ax.semilogy(np.arange(1,N+1),diff/np.linalg.norm(diagA),lw=2, marker="^")
    ax.grid()
    ax.set_xlabel("N")
    ax.set_ylabel("Relative error")
    plt.savefig("MC.png")

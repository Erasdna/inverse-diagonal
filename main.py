import scipy
import scipy.sparse as sparse
from scipy.io import loadmat
import numpy as np
from src.MC import MC
from src.ichol import incomplete_cholesky
from src.lanczos import lanczos_decomposition, lanczos_decomposition_2
from scipy.sparse.linalg import spsolve, inv
from scipy.io import mmread
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("agg")

A = loadmat("nos3/nos3.mat")["Problem"][0][0][1] # lol æsj

#Bruk matrix market formatet: 
#Får en error da...
#A = mmread("nos3/nos3.mtx") #:solbrillemoji:
A=A.tocsc()

G = sparse.csr_matrix(incomplete_cholesky(A))
preconditioned = spsolve(G, spsolve(G, A).transpose()).transpose()
x = np.random.randn(A.shape[0])
error_l = []
error_comb = []
error_mc = []
A_inv = np.linalg.inv(A.todense())
K = 500
U, alpha, beta = lanczos_decomposition_2(preconditioned, x, K)

for k in range(100, K):

    V = np.array(U[:,:k])
    T = sparse.diags((alpha[:k], beta[:k-1], beta[:k-1]), (0, 1, -1))
    #if not np.all(np.linalg.eigvals(T.todense()) > 0):
    #    print(k)
    L = np.linalg.cholesky(T.todense())
    #print(G.shape, V.shape, L.shape)

    W = spsolve(L, spsolve(G.T, V).transpose()).transpose()


    est_diag_l = np.sum(W**2, axis=1)
    est_diag_comb = est_diag_l + MC(A, G, 1e-9, 100, W @ W.T)[-1]
    est_diag_mc = MC(A, G, 1e-9, 100)[-1]
    
    error_l.append(np.sqrt(np.sum((np.diag(A_inv) - est_diag_l)**2) / np.sum(np.diag(A_inv) ** 2)) )
    error_comb.append(np.sqrt(np.sum((np.diag(A_inv) - est_diag_comb)**2) / np.sum(np.diag(A_inv) ** 2)) )
    error_mc.append(np.sqrt(np.sum((np.diag(A_inv) - est_diag_mc)**2) / np.sum(np.diag(A_inv) ** 2)) )

    print(f"MSE at k={k} is {error_l[-1]}")
    print(error_comb[-1])
    print(error_mc[-1])

plt.semilogy(range(70, K), error_l, range(70, k), error_comb)
plt.savefig("error.pdf")







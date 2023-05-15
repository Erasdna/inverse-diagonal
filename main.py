from src.cg import cg
import scipy
import scipy.sparse as sparse
from scipy.io import loadmat
import numpy as np
from src.MC import MC, MC_lanzos_control_variates, MC_lanzos_control_variates_2
from src.ichol import incomplete_cholesky
from src.lanczos import lanczos_decomposition
from scipy.sparse.linalg import spsolve, inv
from scipy.io import mmread
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("agg")

#A = loadmat("nos3/nos3.mat")["Problem"][0][0][1] # lol æsj

#Bruk matrix market formatet: 
#Får en error da...
A = mmread("nos3/nos3.mtx") #:solbrillemoji:
A=A.tocsc()

np.random.seed(55)
G = sparse.csr_matrix(incomplete_cholesky(A))
preconditioned = spsolve(G, spsolve(G, A).transpose()).transpose()
x = np.random.randn(A.shape[0])
error_l = []
error_comb = []
error_comb_fix = []
A_inv = np.linalg.inv(A.todense())
K = range(16, 17)
U, alpha, beta = lanczos_decomposition(preconditioned, x, 100)

for k in K:

    V = np.array(U[:,:k])
    T = sparse.diags((alpha[:k], beta[:k-1], beta[:k-1]), (0, 1, -1))
    #if not np.all(np.linalg.eigvals(T.todense()) > 0):
    #    print(k)
    L = np.linalg.cholesky(T.todense())
    #print(G.shape, V.shape, L.shape)

    W = spsolve(L, spsolve(G.T, V).transpose()).transpose()


    est_diag_l = np.sum(W**2, axis=1)
    #est_diag_mc = MC(A, G, 1e-9, 100)[-1]
    
    GT = sparse.csr_matrix(G.T)
    Z = lambda z: cg(A, z, G, GT, 1e-9) * z
    #WWT = W @ W.T
    #sigma_y = np.sum(WWT ** 2, axis=1) - np.diag(WWT ** 2)
    Y = lambda z: (W @ (W.T @ z)) * z
    est_diag_comb, est_diag_comb_fix= MC_lanzos_control_variates_2(Z, Y, est_diag_l, 400, A.shape[0], clip=True)
    #est_diag_comb, est_diag_comb_fix,aa= MC_lanzos_control_variates(A, G, GT, est_diag_l,W @ W.T, 400, 1e-9, clip=True)
    #est_diag_comb_fix = MC_lanzos_control_variates(Z, Y, est_diag_l, sigma_y, 100,  A.shape[0], True)


    error_l.append(np.sqrt(np.sum((np.diag(A_inv) - est_diag_l)**2) / np.sum(np.diag(A_inv) ** 2)) )
    error_comb.append(np.sqrt(np.sum((np.diag(A_inv) - est_diag_comb)**2) / np.sum(np.diag(A_inv) ** 2)) )
    error_comb_fix.append(np.sqrt(np.sum((np.diag(A_inv) - est_diag_comb_fix)**2) / np.sum(np.diag(A_inv) ** 2)))
    #error_mc.append(np.sqrt(np.sum((np.diag(A_inv) - est_diag_mc)**2) / np.sum(np.diag(A_inv) ** 2)) )

    print(f"MSE at k={k} is {error_l[-1]}")  
    print(f"error ours {error_comb[-1]}")
    print(f"error theirs: {error_comb_fix[-1]}")
    #print(f"Alpha : {aa}")
    #print(error_mc[-1])

plt.semilogy(K, error_l, label="lanczos")
plt.semilogy(K, error_comb, label="lanczos+mc optimal alpha")
plt.semilogy(K, error_comb_fix, label="lanczos+mc")
plt.grid(True)
plt.legend()
plt.savefig("error.pdf")







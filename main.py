import scipy
import scipy.sparse as sparse
from scipy.io import loadmat
import numpy as np
from src.MC import MC
from src.ichol import incomplete_cholesky
from src.lanczos import lanczos_decomposition
from scipy.sparse.linalg import spsolve, inv
from scipy.io import mmread
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("agg")

A = loadmat("nos3/nos3.mat")["Problem"][0][0][1] # lol æsj

#Bruk matrix market formatet: 
#Får en error da...
#A = mmread("nos3/nos3.mtx") #:solbrillemoji:
#A=A.tocsc()

G = incomplete_cholesky(A)
preconditioned = spsolve(G, spsolve(G, A).transpose()).transpose()
x = np.random.randn(A.shape[0])
error = []
tol=1e-9
A_inv = np.linalg.inv(A.todense())
K = 100
U, alpha, beta = lanczos_decomposition(preconditioned, x, K)

for k in range(5, 100):

    
    V = np.array(U[1:k+2])
    T = sparse.diags((alpha[:k+1], beta[:k], beta[:k]), (0, 1, -1))
    L = np.linalg.cholesky(T.todense())
    print(G.shape, V.shape, L.shape)

    W = spsolve(L, spsolve(G.T, V.T).transpose()).transpose()


    est_diag = np.sum(W**2, axis=1)
    
    error.append(np.sqrt(np.sum(np.diag(A_inv) - est_diag) ** 2 / np.sum(np.diag(A_inv) ** 2)) )
    print(f"MSE at k={k} is {error[-1]}")

plt.plot(range(5, 100), error)
plt.savefig("error.pdf")







import scipy
import scipy.sparse as sparse
from scipy.io import loadmat
import numpy as np
from src.ichol import ichol1, ichol, incomplete_cholesky
from src.lanczos import lanczos_decomposition
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("agg")

A = loadmat("nos3/nos3.mat")["Problem"][0][0][1] # lol æsj
G = incomplete_cholesky(A)
preconditioned = spsolve(G, spsolve(G, A).transpose()).transpose()
x = np.random.randn(A.shape[0])
error = []
for k in range(5, 100):

    U, alpha, beta = lanczos_decomposition(preconditioned, x, k)
    V = np.array(U[:-1])
    T = sparse.diags((alpha, beta, beta), (0, 1, -1))
    L = np.linalg.cholesky(T.todense())
    print(G.shape, V.shape, L.shape)

    W = spsolve(L, spsolve(G.T, V.T).transpose()).transpose()

    A_inv = np.linalg.inv(A.todense())

    est_diag = np.sum(W**2, axis=1)
    print(f"MSE at k={k} is {np.sum(np.diag(A_inv) - est_diag) ** 2}")
    error.append(np.sum(np.diag(A_inv) - est_diag) ** 2)

plt.plot(range(5, 100), error)
plt.savefig("error.pdf")







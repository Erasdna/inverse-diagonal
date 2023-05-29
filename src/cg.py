import numpy as np
import scipy
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve, splu, use_solver, spsolve_triangular
from scipy.linalg import solve_triangular, solve_banded, cho_solve_banded, lu_solve, solve, cho_solve


#@numba.njit
#def tri_solve(Lvals, iL, jL,b):
#    n = len(b)
#    x = np.zeros(n)
#    x[0] = b[0]/L[0,0]
#
#    for i in range(1,n):
#        comp = 0
#        for k in range(0,i):
#            index = L[i,k]
#            preSolution = x[k]
#            comp = comp + index * preSolution
#        x[i] = 1/L[i,i] * (b[i] - comp)
#    return x

def chol_solve(L, LT, b):
    use_solver(useUmfpack=False)
    y = spsolve(L, b, permc_spec="NATURAL")
    return spsolve(LT, y, permc_spec="NATURAL")


def cg(A, target, L, LT, tol):
    initial_guess=np.random.rand(target.shape[0])
    r_old=target - A @ initial_guess
    p=chol_solve(L, LT, r_old)
    x=initial_guess
    while np.linalg.norm(r_old)/np.linalg.norm(x)>tol:
        Ap= A @ p
        Minvr=chol_solve(L, LT, r_old)
        alpha= r_old.T @ Minvr / (p.T @ Ap)
        x+=alpha*p
        r_new=r_old-alpha*Ap
        Minvrnew = chol_solve(L,LT, r_new)
        beta=(r_new.T @ Minvrnew)/(r_old.T @ Minvr)
        p=beta*p + Minvrnew
        r_old=r_new.copy()
    return x
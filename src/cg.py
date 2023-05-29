import numpy as np
import scipy
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve, splu, use_solver, spsolve_triangular
from scipy.linalg import solve_triangular, solve_banded, cho_solve_banded, lu_solve, solve, cho_solve
from src.typealias import RealArray

def chol_solve(L : RealArray, 
               LT : RealArray, 
               b : RealArray) -> RealArray:
    """Helper matrix function solver for the CG method

    Args:
        L (RealArray): Left preconditioner
        LT (RealArray): Right preconditioner
        b (RealArray): Target vector

    Returns:
        RealArray: Solution x to LL^T x =b
    """
    use_solver(useUmfpack=False)
    y = spsolve(L, b, permc_spec="NATURAL")
    return spsolve(LT, y, permc_spec="NATURAL")


def cg(A : RealArray, 
       target : RealArray, 
       L : RealArray, 
       LT : RealArray, 
       tol: float) -> RealArray:
    """Preconditioned CG method Ax=target with Cholesky preconditioner

    Args:
        A (RealArray): Matrix A, the linear system
        target (RealArray): Target vector
        L (RealArray): Cholesky decomposition of A
        LT (RealArray): Transpose of L
        tol (float): Convergence threshold

    Returns:
        RealArray: Solution of Ax=target to precision tol
    """
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
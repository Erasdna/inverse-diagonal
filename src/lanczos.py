import numpy as np
from src.typealias import RealArray

def lanczos_decomposition(A: RealArray, x: RealArray, k: int):
    """lanczos method for finding orthogonal basis of krylov space generated by A and x

    Args:
        A (RealArray): Hermitian matrix
        x (RealArray): Vector to generate Krylov subspace
        k (int): Number of generated basis vectors is k + 1 
    """

    #TODO: Reorthogoninze

    alpha = []
    beta = []

    u_1 = x / np.linalg.norm(x)
    U = [u_1]
    for j in range(k + 1):
        u_j = U[-1]
        w = A @ u_j
        alpha.append(np.inner(u_j, w))
        u_tilde = w - alpha[-1] * u_j
        if np.linalg.norm(u_tilde) < 0.7 * np.linalg.norm(w):
            alpha[-1] = np.inner(u_tilde, w)
        if beta:
            u_tilde -= U[-2] * beta[-1]
        beta.append(np.linalg.norm(u_tilde))
        U.append(u_tilde / beta[-1])
    return U, alpha, beta



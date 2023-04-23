import numpy as np

def cg(A, target, precon_inv, tol):
    initial_guess=np.random.rand(target.shape[0]) #Idk?
    r_old=target - A @ initial_guess
    p=precon_inv @ r_old
    x=initial_guess
    while np.linalg.norm(r_old)/np.linalg.norm(x)>tol:
        Ap= A @ p
        Minvr=precon_inv @ r_old
        alpha= r_old.T @ Minvr / (p.T @ Ap)
        x+=alpha*p
        r_new=r_old-alpha*Ap
        beta=(r_new.T @ precon_inv @ r_new)/(r_old.T @ Minvr)
        p=beta*p + precon_inv @ r_new
        r_old=r_new.copy()
    return x
        
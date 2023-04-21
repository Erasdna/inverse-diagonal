import numpy as np

def cg(A, initial_guess, target, precon_inv, maxit):
    r_old=target - A @ initial_guess
    p=precon_inv @ r_old
    x=initial_guess
    r_new=r_old.copy()
    for i in range(maxit-1):
        #print(i)
        Ap= A @ p
        Minvr=precon_inv @ r_old
        alpha= r_old.T @ Minvr / (p.T @ Ap)
        x+=alpha*p
        r_new=r_old-alpha*Ap
        beta=(r_new.T @ precon_inv @ r_new)/(r_old.T @ Minvr)
        p=beta*p + precon_inv @ r_new
        r_old=r_new.copy()
    return x
        
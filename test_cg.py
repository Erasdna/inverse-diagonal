import numpy as np
from src.ichol import incomplete_cholesky
from src.cg import cg
from scipy.io import mmread
from scipy.sparse.linalg import inv,spsolve

m = mmread("nos3/nos3.mtx")
L=incomplete_cholesky(m.tocsc())
print("Finished calculating L")
precon_inv=inv(L.T) @ inv(L)
target=np.random.choice([-1,1],(L.shape[0],))
init=np.random.randn(L.shape[0])
x=cg(m,target,precon_inv,1e-10)
print(np.linalg.norm(target-m @ x))
print(spsolve(m.tocsc(),target)-x)
print(np.linalg.norm(spsolve(m.tocsc(),target)-x))
import scipy.sparse as sparse
from scipy.io import mmread
from ichol import ichol
import matplotlib.pyplot as plt

m = mmread("nos3/nos3.mtx")
L=ichol(m)
ind=L.nonzero()
for i in range(len(ind[0])):
    if ind[0][i]<ind[1][i]:
        print("Oooops")
import numpy as np
from src.MC import MC
from src.lanczos_MC import lanczos_MC
import scipy.sparse as sparse
from scipy.io import mmread
from src.ichol import incomplete_cholesky
from scipy.sparse.linalg import spsolve
from src.lanczos import lanczos_decomposition, lanczos_estimate
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
from cycler import cycler
import os
import json
from tqdm import tqdm
import matplotlib
from scipy.io import loadmat
matplotlib.use("agg")


if __name__=="__main__":
    plt.rc('axes', prop_cycle=(cycler('color', ['tab:blue', 'tab:red', 'tab:green', 'k', 'tab:purple']) +
                            cycler('linestyle', ['-', '-.', '--', ':','-'])))
    
    #Reference matrix
    A = mmread("nos3/nos3.mtx") # https://sparse.tamu.edu/HB/nos3 download as matrix market format
    #A = loadmat("nos3/nos3.mat")["Problem"][0][0][1] # lol æsj
    A=A.tocsc()
    A_inv = np.linalg.inv(A.todense())
    diagA=np.diag(A_inv)
    G = sparse.csr_matrix(incomplete_cholesky(A))
    
    #Set seed
    seed=55
    np.random.seed(seed)
    
    FIGS=["MC_big","MC_many","Lanczos_k","Lanczos_MC", "Optimal_alpha"]
    MODE=["Run","Load"]
    fig=sys.argv[1]
    mode=sys.argv[2]
    path=os.getcwd()
    figpath = path + "/Figs/" + sys.argv[1]
    datapath= path + "/Data/" + sys.argv[1] + ".json"
    #If not in FIGS, be very mad!!
    
    if fig=="MC_big":            
        if mode=="Load":
            with open(datapath, "rb") as file:
                dat=json.load(file)
            relative_error=dat["Relative error"]
            N=dat["MC steps"]
        elif mode=="Run":
            N=10000
            MC_vals=MC(A,G,G.T,1e-9,N)
            diff=MC_vals - diagA[:,None].T
            relative_error=np.linalg.norm(diff,axis=1)/np.linalg.norm(diagA)
        
        fig,ax=plt.subplots()
        ax.loglog(np.arange(40,N),relative_error[40:],lw=2,label="MC")
        ax.set_xlabel("Monte Carlo steps: $N$")
        ax.set_ylabel("$||d_{MC} - diag(A^{-1})||/||diag(A^{-1})||$")
        ax.grid()
        ax.legend()
        plt.savefig(figpath + ".eps")
        plt.savefig(figpath + ".png")
        if mode=="Run":
            stats={
                "Relative error" : relative_error.tolist(),
                "MC steps" : N 
            }
            with open(datapath,'w') as file:
                json.dump(stats,file)
    
    elif fig=="MC_many":
        iterations=100
        if mode=="Load":
            with open(datapath, "rb") as file:
                dat=json.load(file)
            mean=np.array(dat["Mean relative error"])
            quant_25=np.array(dat["2.5 percentile"])
            quant_975=np.array(dat["97.5 percentile"])
            N=dat["MC steps"]
        elif mode=="Run":
            N=960
            dat=np.zeros((iterations,N))
            for it in tqdm(range(iterations)):
                MC_vals=MC(A,G,G.T,1e-9,N)
                diff=MC_vals - diagA[:,None].T
                relative_error=np.linalg.norm(diff,axis=1)/np.linalg.norm(diagA)
                dat[it,:]=relative_error
            mean=np.mean(dat,axis=0)
            quant_25=np.quantile(dat,0.025, axis=0)
            quant_975=np.quantile(dat,0.975, axis=0)
        fig,ax=plt.subplots()
        ax.semilogy(np.arange(40,N),mean[40:],lw=2,label="MC")
        ax.fill_between(np.arange(40,N),quant_25[40:],quant_975[40:],color='b',alpha=0.2)
        ax.set_xlabel("Monte Carlo steps: $N$")
        ax.set_xlim([-2,N])
        ax.set_ylabel("$||d_{MC}^N - diag(A^{-1})||/||diag(A^{-1})||$")
        ax.grid()
        ax.legend()
        plt.savefig(figpath + ".eps")
        plt.savefig(figpath + ".png")
        if mode=="Run":
            stats={
                "Mean relative error" : mean.tolist(),
                "2.5 percentile" : quant_25.tolist(),
                "97.5 percentile" : quant_975.tolist(),
                "iterations" : iterations,
                "MC steps" : N
            }
            with open(datapath,'w') as file:
                json.dump(stats,file)
    elif fig=="Lanczos_k":
        if mode=="Load":
            with open(datapath, "rb") as file:
                dat=json.load(file)
            relative_error=np.array(dat["Relative error"])
            k=dat["k"]
            start=dat["start"]
            print("Relative error at k=100 (reference paper):",relative_error[100-start + 1])
        elif mode=="Run":
            k=500
            start=5
            relative_error=np.zeros(k)
            #Calculate preconditioner
            x = np.random.randn(A.shape[0])
            #Preconditioned A
            preconditioned = spsolve(G, spsolve(G, A).transpose()).transpose()
            U, alpha, beta = lanczos_decomposition(preconditioned, x, k)
            for i in tqdm(range(start,k)):
                est,_=lanczos_estimate(G,U,alpha,beta,i)
                relative_error[i]=np.linalg.norm(est-diagA)/np.linalg.norm(diagA)
        fig,ax=plt.subplots()
        ax.semilogy(np.arange(start,k),relative_error[start:],lw=2,label="Lanczos")
        ax.set_xlabel("Lanczos iterations: $k$")
        ax.set_ylabel("$||d_{Lanczos}^k - diag(A^{-1})||/||diag(A^{-1})||$")
        ax.grid()
        ax.legend()
        plt.savefig(figpath + ".eps")
        plt.savefig(figpath + ".png")
        if mode=="Run":
            stats={
                "Relative error" : relative_error.tolist(),
                "k" : k,
                "start": start
            }
            with open(datapath,'w') as file:
                json.dump(stats,file)
    elif fig=="Lanczos_MC":
        MC_filename=path + "/Data/MC_many.json"
        with open(MC_filename, "rb") as file:
                MC=json.load(file) 
        Lanczos_filename=path + "/Data/Lanczos_k.json"
        with open(Lanczos_filename, "rb") as file:
                Lanczos=json.load(file) 
        if mode=="Load":
            with open(datapath, "rb") as file:
                dat=json.load(file)
            mean=np.array(dat["Mean relative error"])
            quant_25=np.array(dat["2.5 percentile"])
            quant_975=np.array(dat["97.5 percentile"])
            N=dat["MC steps"]
            k=dat["Lanczos iterations"]
        elif mode=="Run":
            k=100
            N=960
            iterations=100
            #Calculate preconditioner
            x = np.random.randn(A.shape[0])
            #Preconditioned A
            preconditioned = spsolve(G, spsolve(G, A).transpose()).transpose()
            U, alpha, beta = lanczos_decomposition(preconditioned, x, k+1)
            Lan_MC_relative_error=np.zeros((iterations,N))
            for it in tqdm(range(iterations)):
                _,Lan_MC,_=lanczos_MC(A,G,U,alpha,beta,k,N)
                diff=Lan_MC - diagA[:,None].T
                Lan_MC_relative_error[it,:]=np.linalg.norm(diff,axis=1)/np.linalg.norm(diagA)
            mean=np.mean(Lan_MC_relative_error,axis=0)
            quant_25=np.quantile(Lan_MC_relative_error,0.025, axis=0)
            quant_975=np.quantile(Lan_MC_relative_error,0.975, axis=0)
        fig,ax=plt.subplots()
        ax.semilogy(np.arange(40,N),mean[40:],lw=2,color='b',label="Lanczos + MC (k=100)")
        ax.fill_between(np.arange(40,N),quant_25[40:],quant_975[40:],color='b',alpha=0.2)
        ax.semilogy(np.arange(40,N),MC["Mean relative error"][40:],lw=2,color='r',label="MC")
        ax.fill_between(np.arange(40,N),MC["2.5 percentile"][40:],MC["97.5 percentile"][40:],color='r',alpha=0.2)
        ax.hlines(Lanczos["Relative error"][Lanczos["k"]-Lanczos["start"]],xmin=40,xmax=N,lw=2,color='k',linestyle="--", label="Lanczos (k=100)")
        ax.set_xlabel("MC steps: $N$")
        ax.set_ylabel("$||d^{N,k} - diag(A^{-1})||/||diag(A^{-1})||$")
        ax.set_xlim([40,N])
        ax.set_ylim([10**-3,10])
        ax.grid()
        ax.legend()
        plt.savefig(figpath + ".eps")
        plt.savefig(figpath + ".png")
        if mode=="Run":
            stats={
                "Mean relative error" : mean.tolist(),
                "2.5 percentile" : quant_25.tolist(),
                "97.5 percentile" : quant_975.tolist(),
                "iterations" : iterations,
                "MC steps" : N,
                "Lanczos iterations" : k
            }
            with open(datapath,'w') as file:
                json.dump(stats,file)
    elif fig=="Optimal_alpha":
        if mode == "Load":
            with open(datapath, "rb") as file:
                dat=json.load(file)
            fixed_relative_error=np.array(dat["Fixed relative error"])
            optimal_relative_error=np.array(dat["Optimal relative error"])
            alphas=dat["Alphas"]
            N=dat["MC steps"]
            test_ks=dat["ks"]
        elif mode=="Run":
            N=200
            test_ks=np.arange(5,200)
            #Calculate preconditioner
            x = np.random.randn(A.shape[0])
            #Preconditioned A
            preconditioned = spsolve(G, spsolve(G, A).transpose()).transpose()
            U, alpha, beta = lanczos_decomposition(preconditioned, x, test_ks[-1]+1)
            fixed_relative_error=np.zeros(len(test_ks))
            optimal_relative_error=np.zeros(len(test_ks))
            alphas=np.zeros(len(test_ks))
            
            for it,kk in tqdm(enumerate(test_ks),total=len(test_ks)):
                optimal,fixed,aa=lanczos_MC(A,G,U,alpha,beta,kk,N)
                diff_optimal=optimal[-1,:] - diagA.T
                optimal_relative_error[it]=np.linalg.norm(diff_optimal)/np.linalg.norm(diagA)
                print(optimal_relative_error[it])
                diff_fixed=fixed[-1,:] - diagA.T
                fixed_relative_error[it]=np.linalg.norm(diff_fixed)/np.linalg.norm(diagA)
                print(fixed_relative_error[it])
                alphas[it]=aa[-1]
        fig,ax=plt.subplots()
        ax.semilogy(test_ks[:50],optimal_relative_error[:50], lw=2,label="Optimal alpha (N=200)")
        ax.semilogy(test_ks[:50],fixed_relative_error[:50], lw=2,label="Fixed alpha=1 (N=200)")
        ax.set_xlabel("Lanczos iterations: $k$")
        ax.set_ylabel("$||d^{N,k} - diag(A^{-1})||/||diag(A^{-1})||$")
        ax.grid()
        ax.legend()
        plt.savefig(figpath + ".eps")
        plt.savefig(figpath + ".png")
        if mode=="Run":
            print(test_ks)
            stats={
                "Fixed relative error" : fixed_relative_error.tolist(),
                "Optimal relative error" : optimal_relative_error.tolist(),
                "Alphas" : alphas.tolist(),
                "ks" : test_ks.tolist(),
                "MC steps" : N,
            }
            with open(datapath,'w') as file:
                json.dump(stats,file)
        
        
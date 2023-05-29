# CLA miniproject: Estimating the diagonal of the inverse

## In src:
    - cg.py: Implements the preconditioned CG method
    - ichol.py : Implements the incomplete Cholesky method
    - lanczos_MC.py : Implements the combined MC - Lanczos method for estimating the diagonal of the inverse
    - lanczos.py : Implements the lanczos method for estimating the diagonal of the inverse
    - Mc.py : Implements the MC method for estimating the diagonal of the inverse

## To reproduce the figures in the report:

The figures are generated using figs.py. The script can be ran using the "Run" option, meaning that the chosen experiments will be run, or "Load", which loads the data from .json files in the folder Data. Figures are saved to the folder Figs.

The .json files used to generate the figures will be provided.

The nos3 matrix should be dowloaded in a folder named nos3 in .mtx format 

Example: python figs.py MC_big Run
Will run the MC_big experiment. 

### Experiments:
    - MC_big : Runs the MC estimator for 10000 steps
    - MC_many : Runs the MC estimator for 960 steps, 100 times
    - Lanczos_k : Runs the Lanczos estimator for 5 to 500 iterations
    - Lanczos_MC : Runs the combined Lanczos - MC method using 100 Lanczos iterations and up to 960 MC steps
    - Optimal_alpha : Runs the equivalent method to Lanczos_MC using an estimated optimal control variate
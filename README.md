# Non-Asymptotic-Bounds-for-Zeroth-Order-Stochastic-Optimization

Contents
--------
1 Introduction                                                                                                  
2 Notes on usage 														    
3 References


                                                       
1 Introduction
--------------

To estimate the problem parameters, namely, L, $ \Lambda $, $ \sigma^2 $, and a bound, say $\alpha_0$, on the derivative of the objective function, we use an initial i.i.d. sample of size N_0 = 200. We compute the l_2-norm of the Hessian of the objective function at 200 randomly selected points, by averaging over N_0 samples, 
and then take the maximum l_2-norm of the Hessian over these points as an estimation of L, $ \Lambda $. A similar procedure has been employed by Ghadimi,2013 [5].  Similarly, 200 i.i.d. samples of the squared norm of stochastic gradient of the objective, and third derivative of the objective, respectively, are used to estimate $ \sigma^2 $ and $\alpha_0$.
For the SVM problem setting, the optima x* is unknown. However, using the fact that the objective has non-negative optimal values, i.e., f(x*) > 0, we infer that $ D_f  \le  f(x_1) $.
Using these estimates, we implement the ZRSG and ZRSQN algorithm with a stepsize chosen as mentioned in main paper for different settings. 


We implement following gradient and Hessian oracles:
1. Gaussian smoothing
2. 1SPSA and 2SPSA	
3. 1RDSA_Uniform and 2RDSA_Uniform
4. 1RDSA_AsymBer and 2RDSA_AsymBer
5. 1RDSA_Lex_DP and 2RDSA_Lex_DP 
6. 1RDSA_Perm_DP and 2RDSA_Perm_DP
7. 1RDSA_KW_DP and 2RDSA_KW_DP


2 Notes on usage of Matlab files
---------------------------------

2.1 The main files in the distribution are:
--------------------------------------------

First-order schemes:
-------------------
i) run_rsg_svm_bank.m --> this file contains the implementation to taking input parameters, loading Dataset, randomly shuffling dataset, performing min max normalization,  performing SVD to reduce dimension (if needed), splitting training and testing dataset, paramater estimation L, sigma, f(x1), etc and then running various first-order algorithms and generating plot for banknote authentication dataset.

ii) run_rsg_svm_heart.m --> this file contains the implementation to taking input parameters, loading Dataset, randomly shuffling dataset, performing min max normalization,  performing SVD to reduce dimension (if needed), splitting training and testing dataset, paramater estimation L, sigma, f(x1), etc and then running various first-order algorithms and generating plot for heart disease dataset.

iii) run_rsg_svm_synthetic.m --> this file contains the implementation to taking input parameters, generating synthetic Dataset, splitting training and testing dataset, paramater estimation L, sigma, f(x1), etc and then running various first-order algorithms and generating plot.

iv) run_rsg_multimodal.m --> this file contains the implementation to taking input parameters, paramater estimation L, sigma, f(x1), etc and then running various first-order algorithms and generating plot for the Multimodal function.


Second-order schemes:
---------------------
i) run_rsqn_svm_bank.m --> this file contains the implementation to taking input parameters, loading Dataset, randomly shuffling dataset, performing min max normalization,  performing SVD to reduce dimension (if needed), splitting training and testing dataset, paramater estimation L, sigma, f(x1), etc and then running various second-order algorithms and generating plot for banknote authentication dataset.

ii) run_rsqn_svm_heart.m --> this file contains the implementation to taking input parameters, loading Dataset, randomly shuffling dataset, performing min max normalization,  performing SVD to reduce dimension (if needed), splitting training and testing dataset, paramater estimation L, sigma, f(x1), etc and then running various second-order algorithms and generating plot for heart disease dataset.

iii) run_rsqn_svm_synthetic.m --> this file contains the implementation to taking input parameters, generating synthetic Dataset, splitting training and testing dataset, paramater estimation L, sigma, f(x1), etc and then running various second-order algorithms and generating plot.

iv) run_rsqn_multimodal.m --> this file contains the implementation to taking input parameters, paramater estimation L, sigma, f(x1), etc and then running various second-order algorithms and generating plot for the Multimodal function.



2.2 Supporting files:
---------------------
rsg.m - This file implements Zeroth-order Randomized Stochastic Gradient (ZRSG) algorithm with unbiased and biased gradient estimates for Multimodal function.
rsg_svm.m - This file implements Zeroth-order Randomized Stochastic Gradient (ZRSG) algorithm with unbiased and biased gradient estimates for non-convex support vector machine problem.
rsqn.m - This file implements Zeroth-order Randomized Stochastic quasi-Newton (ZRSQN) algorithm with unbiased and biased gradient estimates for Multimodal function.
rsqn_svm.m - This file implements Zeroth-order Randomized Stochastic Quasi-Newton (ZRSQN) algorithm with unbiased and biased gradient estimates for non-convex support vector machine problem.

------------------------------------------

SZO.m - This file contains the implementation of various Stochastic Zeroth Order Oracles
SFO.m - This file contains the implementation of various Stochastic First Order Oracles
SSO.m - This file contains the implementation of various Stochastic Second Order Oracles
SVM.m - This file contains the implementation of various SZO, SFO and SSO oracles and returns nonconvex support vector machine (SVM) sigmoid loss function value and gradient or Hessian depending on argument passed.

The following SZO, SFO and SSO are implemented in above mentioned files:
1. Unbiased gradient
2. Biased gradient/Hessian using Gaussian smoothing
3. Biased gradient/Hessian using 1SPSA
4. Biased gradient/Hessian using 1RDSA_Uniform
5. Biased gradient/Hessian using 1RDSA_AsymBer
6. Biased gradient/Hessian using 1RDSA_Lex_DP
7. Biased gradient/Hessian using 1RDSA_Perm_DP
8. Biased gradient/Hessian using 1RDSA_KW_DP



2.3 On input parameters: 
-------------------------

Most of the algorithms above take as input the following for non-convex SVM problem:
------------------------------------------------------------------------------------
algo -> ub - unbiased, spsa, rdsa_u, rdsa_ab, rdsa_perm, rdsa_lex, rdsa_kw
N -> Iteration limit
T -> Number of independent replications
Q -> Number of samples to find quality of solution at x_R
x_1 -> Initial point
gamma -> Step size (vector for all k = 1,...,N)
P_R -> Probability mass function (vector for all k = 1,...,N)
eta -> Perturbation constant
lambda -> Constant Lagrange multiplier
u -> Training features
v -> Training true lables 
test_u -> Testing features
test_v -> Testing true lables 



Most of the algorithms above take as input the following for Multimodal function:
---------------------------------------------------------------------------------
d -> Dimension of the problem
type -> 1 = Multimodal function
algo -> ub - unbiased, spsa, rdsa_u, rdsa_ab, rdsa_perm, rdsa_lex,
rdsa_kw
N -> Iteration limit
T -> Number of independent replications
Q -> Number of samples to find quality of solution at x_R
x_1 -> Initial point
x_star -> Optimal point
gamma -> Step size (vector for all k = 1,...,N)
P_R -> Probability mass function (vector for all k = 1,...,N)
eta -> Perturbation constant
sigma -> Noise in SFO or SZO




3 References
------------
[1] J. C. Spall, "Multivariate stochastic approximation using a simultaneous perturbation gradient approximation", IEEE Trans. Auto. Cont., vol. 37, no. 3, pp. 332-341, 1992.

[2] J. C. Spall, "Adaptive stochastic approximation by the simultaneous perturbation method", IEEE Trans. Autom. Contr., vol. 45, pp. 1839-1853, 2000.

[3] Prashanth L.A., S. Bhatnagar, Michael Fu and Steve Marcus, "Adaptive system optimization using (simultaneous) random directions stochastic approximation", arXiv:1502.05577, 2015.

[4] Prashanth L A, Shalabh Bhatnagar, Nirav Bhavsar, Michael Fu and Steven I. Marcus, "Random directions stochastic approximation with deterministic perturbations", 	arXiv:1808.02871, 2018.

[5] Ghadimi, Saeed, and Guanghui Lan. "Stochastic first-and zeroth-order methods for nonconvex stochastic programming." SIAM Journal on Optimization 23, no. 4 (2013): 2341-2368.


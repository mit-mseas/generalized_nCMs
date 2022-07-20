# Generalized Neural Closure Models with Interpretability

**Source code**: This directory contains the source code for Generalized Neural Closure Models with Interpretability, available on arXiv: http://arxiv.org/abs/xxxx.xxxxx

**Data sets**: Data is re-generated everytime before training for each of the experiments. However, all the runs used in the results and discussion section could be found here: http://mseas.mit.edu/download/guptaa/Generalized_nCM_paper_final_runs/

**Directory structure and scripts**

- `src/utilities/DDE_Solver.py`: A delay differential equation solver compatible with TensorFlow.
- `src/solvers/`: Contains all the functions/classes needed for adjoint equation and the main training loop for both nODE and distributed-nDDE implementations
- `src/findiff/`: Contains an explicit finite-difference solver
- Experiments - 1a 
	- `testcases/kdv_burgers_eqn/`: Contains the scripts corresponding to the neural closure model. Also contains scripts for plotting and analysis
	- `src/kdv_burgers_eqn_case/`: Script containing the RHS of the KdV Burgers' equations
    - `final_paper_scripts/Experiments1a/`: Contains scripts for plotting and analysis scripts for the results in the paper
- Experiments - 1b
	- `testcases/burgers_eqn_findiff/`: Contains the scripts corresponding to the different neural closure models. Also contains scripts for plotting and analysis
	- `src/burgers_eqn_case_findiff/`: Script containing the RHS of the classic Burgers' equations
    - `final_paper_scripts/Experiments1b/`: Contains scripts for plotting and analysis scripts for the results in the paper
- Experiments - 2a & 2b
	- `testcases/bio_eqn/`: Contains the scripts corresponding to the neural closure models. Also contains scripts for plotting and analysis
	- `src/bio_eqn_case/`: Script containing the RHS of the NPZ-OA and NPZD-OA equations
    - `final_paper_scripts/Experiments2*/`: Contains scripts for plotting and analysis scripts for the results in the paper
    

### Abstract

Complex dynamical systems are used for predictions in many domains. Because of computational costs, models are truncated, coarsened, or aggregated. As the neglected and unresolved terms become important, the utility of model predictions diminishes. In our recently published work \[1\], we developed a novel neural delay differential equations (nDDEs) based framework to learn closure parameterizations for known-physics/low-fidelity models using data from high-fidelity simulations and increase the long-term predictive capabilities of these models, called *neural closure models*. The need for using time-delays in closure parameterizations is deep rooted in the presence of inherent delays in real-world systems, and theoretical justification from the Mori-Zwanzig formulation. In the present study, we will extend our earlier framework and develop an unified approach based on neural partial delay differential equations (nPDDEs) which augments low-fidelity models in their partial differential equation (PDE) forms with both markovian and non-markovian closure parameterized with neural networks (NNs). The amalgamation of low-fidelity model and NNs in the continuous spatio-temporal space followed with numerical discretization, automatically allows for generalizability to computational grid resolution, boundary conditions, initial conditions, and provide interpretability. We will provide adjoint PDE derivations in the continuous form, thus allowing one to implement across differentiable and non-differentiable computational physics codes, different machine learning frameworks, and also allowing for nonuniformly-spaced spatio-temporal training data. We will demonstrate the ability of our new framework to discriminate and learn model ambiguity in the advecting shock problem governed by the KdV-Burgers PDE and a biogeochemical-physical ocean acidification model in an interpretable fashion. We will also learn the subgrid-scale processes and augment model simplification in those models, respectively. Finally, we will analyze computational advantages associated with our new framework.

#### References
\[1\] A. Gupta and P. F. J. Lermusiaux. Neural closure models for dynamical systems. *Proceedings of The
Royal Society A*, 477(2252):1–29, Aug. 2021. doi: 10.1098/rspa.2020.1004.


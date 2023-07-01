# Generalized Neural Closure Models with Interpretability

**Source code**: This directory contains the source code for Generalized Neural Closure Models with Interpretability, published in Nature Scientific Reports: [https://www.nature.com/articles/s41598-023-35319-w](https://www.nature.com/articles/s41598-023-35319-w)

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

Improving the predictive capability and computational cost of dynamical models is often at the heart of augmenting computational physics with machine learning (ML). However, most learning results are limited in interpretability and generalization over different computational grid resolutions, initial and boundary conditions, domain geometries, and physical or problem-specific parameters. In the present study, we simultaneously address all these challenges by developing the novel and versatile methodology of unified neural partial delay differential equations. We augment existing/low-fidelity dynamical models directly in their partial differential equation (PDE) forms with both Markovian and non-Markovian neural network (NN) closure parameterizations. The melding of the existing models with NNs in the continuous spatiotemporal space followed by numerical discretization automatically allows for the desired generalizability. The Markovian term is designed to enable extraction of its analytical form and thus provides interpretability. The non-Markovian terms allow accounting for inherently missing time delays needed to represent the real world. Our flexible modeling framework provides full autonomy for the design of the unknown closure terms such as using any linear-, shallow-, or deep-NN architectures, selecting the span of the input function libraries, and using either or both Markovian and non-Markovian closure terms, all in accord with prior knowledge. We obtain adjoint PDEs in the continuous form, thus enabling direct implementation across differentiable and non-differentiable computational physics codes, different ML frameworks, and treatment of nonuniformly-spaced spatiotemporal training data. We demonstrate the new generalized neural closure models (gnCMs) framework using four sets of experiments based on advecting nonlinear waves, shocks, and ocean acidification models. Our learned gnCMs discover missing physics, find leading numerical error terms, discriminate among candidate functional forms in an interpretable fashion, achieve generalization, and compensate for the lack of complexity in simpler models. Finally, we analyze the computational advantages of our new framework.

#### References
\[1\] A. Gupta and P. F. J. Lermusiaux. Neural closure models for dynamical systems. *Proceedings of The
Royal Society A*, 477(2252):1–29, Aug. 2021. doi: 10.1098/rspa.2020.1004.


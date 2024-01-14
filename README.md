# Optimal Experimental Design for Understanding Burn Injuries
This project investigates the optimal experimental design for measuring thermal proprieties in the layers of skin. The project’s main goal is to computationally and mathematically investigate various experimental designs which use phantom models for determining diffusitivity parameters from temperature measurements. This will be accomplished via the following associated goals:
1. Build a numerical PDE solver capable of solving both the basic model as well as being easily extendable to more complex cases (e.g. blood perfusion, inhomogeneous structure, multiple spatial dimensions, time-varying boundary conditions, etc)
2. Implement numerical methods for basic parameter estimation using nonlinear optimisation for fitting the forward model to data
3. Compare various experimental designs by analysing uncertainty, applicability, parameter identifiability, and complexity.
4. Provide a basis for future work regarding useful mathematical models and uncertainty quantification techniques

This project will apply appropriate mathematical models to the skin models produced in the first sub-project. The computational models will be written in Python using suitable numerical solvers. The first model will be a 1D, one-layer model solved using an explicit finite difference scheme and checked against a validated solution. Later models will adopt a finite element method to solve the model due to the robustness of the solver.
The mathematical model will then attempt to estimate the known thermal diffusivity from the model using optimisation techniques. Since the true value is known, the parameter value accuracy of each model will be known. Thus, this, in conjunction with uncertainty analysis, will indicate which design gives the most accurate parameter estimates.
Building on the Simpson (2018) study, the model will incorporate experimental variability through comprehensive uncertainty quantification.

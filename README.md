# Efficient-Plasticity-Camera-Ready
Camera ready code repo for the NeuRIPS 2020 paper: "Learning efficient task-dependent representations with synaptic plasticity"
Approximate training time on a personal computer: 5 hrs.

# normative_plasticity_cr.py
Code containing the PlasticNet class, including the main code for running simulations and generating plots

# pl_exp_params_cr.py
Contains all of the parameters for the multiple experiments we run with normative_plasticity_cr.py

# pl_support_functions.py
All of the supporting functions we need for normative_plasticity_cr.py and associated analysis
1. nonlinearities and their inverses
2. sampling functions for input to the network
3. generic and specific weight update rules
4. objective functions (MSE, logistic regression)
5. basic data analysis functions

# pl_plotters_cr.py
Basic code for generating plots and making them pretty

# pl_plot_generator_cr
Code for generating basic figures after training a network

# List of dependencies
1. numpy
2. time
3. copy
4. scipy
5. sklearn
6. os
7. seaborn
8. matplotlib

# Instructions
Experimental parameters are contained in the pl_exp_params_cr.py file.
Changing the objective from 'linear' to 'classifier will switch the training task from estimation to classification.
Changing 'trial_num_2' in params will affect training time, changing 'test_num' will affect the number of test trials.
To run a simulation, run the normative_plasticity_cr.py file. After running the simulation, you can plot results with the pl_plot_generator_cr.py file.

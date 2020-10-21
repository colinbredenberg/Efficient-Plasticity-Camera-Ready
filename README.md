# Learning efficient task-dependent representations with synaptic plasticity
Camera ready code repo for the NeuRIPS 2020 paper: "Learning efficient task-dependent representations with synaptic plasticity".

Approximate training time on a personal computer: 5 hrs.

Approximate runtime with the pre-trained network on a personal computer: 25 min.

This code trains a basic recurrent neural network to solve either the estimation or classification tasks outlined in the aforementioned paper, using a three-factor synaptic plasticity rule.

network_cr:

Contains a network pre-trained on the estimation task outlined in the paper.

normative_plasticity_cr.py:

Code containing the PlasticNet class, including the main code for running simulations and generating plots

pl_exp_params_cr.py:

Contains all of the parameters for the multiple experiments we run with normative_plasticity_cr.py

pl_support_functions.py:

All of the supporting functions we need for normative_plasticity_cr.py and associated analysis
1. nonlinearities and their inverses
2. sampling functions for input to the network
3. generic and specific weight update rules
4. objective functions (MSE, logistic regression)
5. basic data analysis functions

pl_plotters_cr.py:

Basic code for generating plots and making them pretty

pl_plot_generator_cr:

Code for generating basic figures after training a network

# Requirements
1. numpy
2. time
3. copy
4. scipy
5. sklearn
6. os
7. seaborn
8. matplotlib
9. dill

# Instructions
Experimental parameters are contained in the pl_exp_params_cr.py file.

Changing the objective from 'linear' to 'classifier' will switch the training task from estimation to classification.

Changing 'trial_num_2' in params will affect training time, changing 'test_num' will affect the number of test trials.

Training/Loading a Pre-trained model:

The file network_cr contains a model pre-trained on the estimation task.
The boolean variable network_load in pl_exp_params_cr.py determines whether the system loads a pre-trained network (on the estimation task) or trains the network.

Evaluation:

To run a simulation, run the normative_plasticity_cr.py file. After running the simulation, you can plot results with the pl_plot_generator_cr.py file.


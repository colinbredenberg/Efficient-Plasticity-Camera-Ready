# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 01:15:00 2018
pl_exp_params.py
File for storing the parameters used in multiple experiments using the
normative_plasticity.py code base.

1. L1-regularized regression of a Von Mises distribution
2. 2AFC classification of a Von Mises distribution
@author: Colin
"""
import numpy as np
from pl_support_functions_cr import *
import os
np.random.seed(seed = 120994)
#### CHECKLIST ###
#1. is the version set correctly?
#2. is the objective set correctly?
#3. have you made changes to 'standard' without making the changes to 'hpc'?
#4. have you set your parameters correctly?
#5. will the changes you just made mess with anything in the rest of the code?
#6  have you requested enough time on the HPC for your code to run?
version = 'standard'
hpc_control = False #causes all runs of the hpc to learn only the decoder
#%% The standard set of parameters, for running on the main computer
#%%
if (version == 'standard'):
    annealed = False #do simulated annealing
    input_switch_time = 20;
    
    #choose the objective to run either the 'classification' or 'linear' (reproduction) task
    objective = 'linear'
    #%% Parameters
    #detail all of the fixed parameters for the network
    decoder_only = False
    normalize_output = False
    input_noise = False;
    #%% Parameters
    #detail all of the fixed parameters for the network
    perturbed = False
    decoder_fixed = perturbed
    within_manifold = False
    params = {'input_dim': 12,
              'sample_dim': 1, #this is only relevant if there is a change in dimensionality from samples -> inputs. For Von Mises sampling, 1 theta value is turned into an x and a y value.
              'T': 1000,
              'dt': 0.01,
              'trial_num':0,
              'trial_num_2': 260,
              'test_num': 100,
              'record_period': 1, #number of trials to wait before recording 1 trial
              'R': 1,
              'sigma': 0.2,
              'low_r_threshold': 0.001,
              'high_r_threshold': 100,
              'transient': 40,
              'decoder_update_freq':5,
              'decoder_window':5,
              #parameters for the nonlinearity
              'alpha': 0,
              'beta': 1/30,
              'gamma': 30,
              'record_res':20} #number of steps to skip in a given trial before taking a record of r or o
    #theta: weighting for the various sliding averages in the simulation
    #theta_obj: weighting for the sliding average of the objective function through time
    #lambda_l1: Lagrange multiplier for the l1 regularizer
    params.update(theta = params['dt']* 3/ input_switch_time, theta_obj = 5e-5, weight_l2 = 0.0001)
    #detail all of the function parameters for the network
    param_fns = {'nl': relu_diff,#linear, #nonlinearity
                 'inv_nl': inv_relu_diff,#inv_linear, #inverse nonlinearity
                 'plasticity_rule': MSE_l2_weight, #rule for updating recurrent weights
                 'input_rule': MSE_l2_input, #rule for updating the input weights
                 'bias_rule': MSE_l2_bias,#rule for updating the biases
                 'decoder_rule': MSE_decoder_rule, #rule for updating the decoder
                 'objective': MSE_l2,#function to calculate the error throughout time
                 'target_generator': circular_classifier, #function to determine when the output is supposed to be 1 vs. 0
                 'decoder_fn': standard_decoder,
                 'input_generator': mv_vonmises, #function for generating the input
                 'input_f': mv_vonmises_update, #function for generating the drift in a stochastic sampler of the target SS distribution
                 'input_sampler': mv_vonmises_sampler #generates samples from input distribution
                 } 
    #detail the parameters for generating input data
    input_params = {'T': params['T'],
                    'dt': params['dt'],
                    'switch_period': 200,
                    'mu_in': 0,
                    'sigma_in': 0.1,
                    'sigma_sampler': 0.06, #not to be confused with sigma_in, sigma_sampler determines variance of the MCMC sampling
                    'stim_gain': 0.2,
                    'offset': 0.2,
                    'input_dim': params['input_dim'],
                    'sample_dim': params['sample_dim'],
                    'trial_num': params['trial_num'],
                    'test_num': params['test_num']}
    if(objective == 'classifier'):
        param_fns.update(target_generator = circular_classifier,
                         decoder_fn = logistic_decoder,
                         decoder_rule = log_likelihood_decoder_rule,
                         objective = classifier_vonmises_l12,
                         input_rule = classifier_input,
                         bias_rule = classifier_bias,
                         plasticity_rule = classifier_weight,
                         );
        input_params.update(kappa_in = 0.75);
        params.update(N = 25);
        params.update(theta_thresh = 0,
                      theta_thresh_rev = -np.pi, #the reverse threshold is when there are two classification
                      target_dim = 1,
                      theta_obj = 5e-5,
                      lambda_l2 = 1.5/params['N'],
                      lambda_l1 = 0/params['N'],
                      learning_rate = 1e-6, #learning rates for all of the different parameters in the network
                      learning_rate_bias = 1e-6,
                      learning_rate_input = 1e-6,
                      learning_rate_decoder = 1e-8,
                      weight_l2 = 3/40**2,
                      bias_weight_l2 = 2/40,
                      i_weight_l2 = 25/40/10)
        decoder = np.random.uniform(-1/params['N'], 1/params['N'], size = (params['target_dim'], params['N']));
    elif(objective=='linear'):
        input_params.update(kappa_in = 0.75)
        param_fns.update(target_generator = autoencoder,
                         decoder_fn = standard_decoder,
                         objective = MSE_l2)
        params.update(N = 40);
        params.update(target_dim = 2,
                      lambda_l2 = 1.5/params['N'],
                      lambda_l1 = 0/params['N'],
                      learning_rate = 5e-6, #learning rates for all of the different parameters in the network
                      learning_rate_bias = 5e-6,
                      learning_rate_input = 5e-6,
                      learning_rate_decoder = 5e-6,
                      weight_l2 = 0.5/25**2,
                      bias_weight_l2 = 3/25**2,#2/25,
                      i_weight_l2 = 3/25**2)#12/25/params['input_dim'])
        decoder = np.random.uniform(-1/params['N'], 1/params['N'], size = (params['target_dim'], params['N']))

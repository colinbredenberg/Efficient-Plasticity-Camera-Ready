# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 01:32:16 2018
pl_support_functions.py
File for storing all of the miscellaneous functions used by the normative
plasticity code base. This includes:
1. nonlinearities and their inverses
2. sampling functions
3. generic weight update rules (not specially crafted for specific experiments)
4. objective functions
5. basic data analysis functions
@author: Colin
"""

import numpy as np
import pl_exp_params_cr as exp_params
import matplotlib.pyplot as plt
from scipy.stats import vonmises
from scipy.stats import multivariate_normal
from sklearn.decomposition import PCA
#%% Nonlinearities

#basic sigmoidal nonlinearity

def nl(params, x):
    alpha = params['alpha'];
    beta = params['beta'];
    gamma = params['gamma']
    return 1/(alpha + beta * np.exp(-1*gamma*x));

#derivative of the basic sigmoidal nonlinearity
def nl_prime(params, x):
    alpha = params['alpha'];
    beta = params['beta'];
    gamma = params['gamma'];
    return gamma * beta * np.exp(-1*gamma*x) / (alpha + beta*np.exp(-1*gamma*x))**2

#inverse of the basic sigmoidal nonlinearity

def inv_nl(params, x):
    alpha = params['alpha'];
    beta = params['beta'];
    gamma = params['gamma'];
    return -1/gamma * np.log(1/(beta*x) - alpha);

#basic linear F-I curve
def linear(params, x):
    alpha = params['alpha'] #offset
    gamma = params['gamma'] #slope
    return alpha + gamma * x

def inv_linear(params, x):
    alpha = params['alpha']
    gamma = params['gamma']
    return (x - alpha)/gamma

# differentiable ReLU F-I curve
def relu_diff(params, x):
    alpha = params['alpha']
    gamma = params['gamma'] #gamma multiplicatively increases the slope and the sharpness of the relu
    beta = params['beta'] #beta multiplicatively increases only the slope
    return np.log(1 + np.exp(alpha + gamma*x)) * beta; 

def inv_relu_diff(params, x):
    alpha = params['alpha']
    gamma = params['gamma']
    beta = params['beta']
    return (np.log(np.exp(x/beta) - 1) - alpha)/gamma

def inv_relu_diff_prime(params, x):
    alpha = params['alpha']
    gamma = params['gamma']
    beta = params['beta']
    return np.exp(x/beta)/(beta * gamma * (np.exp(x/beta) - 1))
#nondifferentiable ReLU
    
def relu(params, x):
    alpha = params['alpha']
    gamma = params['gamma']
    return np.maximum(0, gamma*x + alpha)

#approximate energy calculation--assumes R = 1, and the ReLU is the rectified identity. Assumes all tested r values are nonzero
def energy_relu(r, s, network):
    return -1/2 * np.sum(network.weights * np.outer(r,r)) + np.sum(1/2 * r**2) - np.dot(network.biases, r) - np.sum(network.i_weights * np.outer(r,s))

def differential_energy_relu(r, s, network):
    return -np.sum(network.i_weights * np.outer(r,s));

#plot the energy as a function of the distance from the mean
def stimulus_dependent_energy(r_record, theta_record, input_record, network, axes, axes_2, axes_3):
    downsample = 200;
    N = r_record.shape[1]
    N = len(r_record[0,:,0]);
    r_reshape = np.reshape(np.transpose(r_record, (1, 0, 2)), (N, r_record.shape[0]*r_record.shape[2]));
    theta_reshape = np.reshape(np.transpose(theta_record, (1, 0, 2)), (1, theta_record.shape[0]*theta_record.shape[2]));
    input_reshape = np.reshape(np.transpose(input_record, (1, 0, 2)), (input_record.shape[1], input_record.shape[0]*input_record.shape[2]));
    theta_range = np.unique(theta_record);
    distance_total = []
    energy_total = []
    #plt.figure()
    energy_means = np.zeros((len(theta_range,)))

    for jj in range(0, len(theta_range)):
        idx = np.where(theta_reshape == theta_range[jj])[1]
        r_trim = r_reshape[:,idx]
        r_trim = r_trim[:,::downsample]
        input_trim = input_reshape[:,idx]
        input_trim = input_trim[:,::downsample]
        energy = np.zeros((len(r_trim[0,:],)))
        theta_trim = theta_reshape[:,idx]
        theta_trim = np.ndarray.flatten(theta_trim[:,::downsample])
        for ii in range(0, len(r_trim[0,:])):
            energy[ii] = energy_relu(r_trim[:,ii], input_trim[:,0], network)
        #plt.figure()
        energy_means[jj] = np.mean(energy)
    
    axes.plot(theta_range, (energy_means))#/np.linalg.norm(energy_means-np.mean(energy_means)))
    print(np.linalg.norm(energy_means-np.mean(energy_means)))
    centered_energy_means = energy_means - np.mean(energy_means)
    axes_2.scatter(network.params['sigma']**2, np.linalg.norm(energy_means- np.mean(energy_means)))
    #axes_3.scatter(network.params['sigma']**2, centered_energy_means[0]/centered_energy_means[6], c = 'k')
    #axes_3.scatter(network.params['sigma']**2, centered_energy_means[7]/-0.03818894278052164, c = 'r')
    print(centered_energy_means[0])
    print(centered_energy_means[6])

    """
    for ii in np.arange(0, len(theta_range)):
        idx = np.where(theta_reshape == theta_range[ii])[1]
        r_trim = r_reshape[:,idx]
        r_trim = r_trim[:,::downsample]
        input_trim = input_reshape[:,idx]
        input_trim = input_trim[:,::downsample]
        theta_trim = theta_reshape[:,idx]
        theta_trim = theta_trim[:,::downsample]
        energy = np.zeros((len(r_trim[0,:],)))
        distance = np.zeros((len(r_trim[0,:],)))
        r_mean = np.mean(r_trim[:,:],axis = 1)
        energy_mean = energy_relu(r_mean, input_trim[:,0], network)
        for jj in range(0, len(r_trim[0,:])):
            energy[jj] = (energy_relu(r_trim[:,jj], input_trim[:,jj], network) - energy_mean)
            distance[jj] = np.linalg.norm(network.decoder @ (r_trim[:,jj]-r_mean)) #- np.array([np.cos(theta_trim[:,jj]), np.sin(theta_trim[:,jj])]))
        energy_total.append(energy)
        distance_total.append(distance)
    plt.figure()

    for kk in range(0, len(distance_total)):
        #if (kk == 7 or kk == 0):
        if (1 == 1):
            plt.scatter(distance_total[kk], energy_total[kk], alpha = 0.1)
            #plt.ylim((-10, 100))
    
    plt.figure()
    for ii in range(0, len(theta_range)):
        plt.scatter(theta_range[ii], np.mean(distance_total[ii]))
    """
    #print(energy)
    return energy_means, distance_total

#estimate the average signal-to-noise ratio of a neuron in the RNN as a function of input test stimuli.
def stimulus_dependent_s2n(r_record, theta_record, input_record, network, axes, axes_2):
    downsample = 20;
    N = r_record.shape[1]
    N = len(r_record[0,:,0]);
    r_reshape = np.reshape(np.transpose(r_record, (1, 0, 2)), (N, r_record.shape[0]*r_record.shape[2]));
    theta_reshape = np.reshape(np.transpose(theta_record, (1, 0, 2)), (1, theta_record.shape[0]*theta_record.shape[2]));
    input_reshape = np.reshape(np.transpose(input_record, (1, 0, 2)), (input_record.shape[1], input_record.shape[0]*input_record.shape[2]));
    theta_range = np.unique(theta_record);
    s2n = np.zeros((len(theta_range,)))
    s2n_decoder = np.zeros((len(theta_range,)))
    U_dec, S_dec, V_T_dec = np.linalg.svd(network.decoder)
    for jj in range(0, len(theta_range)):
        idx = np.where(theta_reshape == theta_range[jj])[1]
        r_trim = r_reshape[:,idx]
        r_mean = np.mean(r_trim, axis = 1)
        r_cov = np.cov(r_trim)
        r_mean_dec = np.mean(V_T_dec[0:2,:] @ r_trim, axis = 1)
        r_cov_dec = np.cov(V_T_dec[0:2,:] @ r_trim)
        r_mean_orth = np.mean(V_T_dec[2::,:] @ r_trim, axis = 1)
        r_cov_orth = np.cov(V_T_dec[2::,:] @ r_trim)
        r_trim = r_trim[:,::downsample]
        input_trim = input_reshape[:,idx]
        input_trim = input_trim[:,::downsample]
        theta_trim = theta_reshape[:,idx]
        theta_trim = np.ndarray.flatten(theta_trim[:,::downsample])

        s2n[jj] = np.dot(r_mean_orth, r_mean_orth)/np.trace(r_cov_orth)
        s2n_decoder[jj] = np.dot(r_mean_dec, r_mean_dec)/np.trace(r_cov_dec)
    axes.plot(theta_range, s2n)
    axes_2.plot(theta_range, s2n_decoder)
    
    return s2n, s2n_decoder
#%% Decoder functions

#takes a network response and outputs a sigmoid using a trained decoder
def logistic_decoder(decoder, r):
    output = nl({'alpha': 1, 'beta': 1, 'gamma': 100}, np.dot(decoder, r));
    return output

#takes a network response and applies a linear decoder
def standard_decoder(decoder, r):
    return np.dot(decoder, r);

#takes a network response and applies both a logistic and nonlinear decoder
def MSE_LL_decoder(decoder, r):
    decoder_LL = decoder[-1,:]
    decoder_MSE = decoder[0:(len(decoder[:,0])-1),:]
    output_LL = nl({'alpha': 1, 'beta': 1, 'gamma': 100}, np.dot(decoder_LL, r))
    output_MSE = np.dot(decoder_MSE, r)
    return np.append(output_MSE, output_LL)

#basic classifier for the input
def circular_classifier(params, theta):
    #return np.greater_equal(theta, params['theta_thresh']).astype(float)
    return np.logical_or(np.greater_equal(theta, params['theta_thresh']).astype(float), np.less_equal(theta, params['theta_thresh_rev']).astype(float))


#autoencoder identity function
def autoencoder(params, inputs):
    return inputs

#combination of the autoencoder and the basic classifier for input
def MSE_LL_target(params, theta_total):
    theta_LL = theta_total[-1]
    inputs = theta_total[0:len(theta_total)-1]
    target_MSE = inputs
    target_LL = np.logical_or(np.greater_equal(theta_LL, params['theta_thresh']).astype(float), np.less_equal(theta_LL, params['theta_thresh_rev']).astype(float))
    target_total = np.append(target_MSE, target_LL)
    return target_total

def BCI_target(params):
    return params['bci_threshold']
#%% Synaptic weight updates

#updates that are just zero
def null_rule(r, i, rr_avg):
    return np.zeros((len(r),len(r)));


def null_input_rule(r, i, ri_avg):
    return np.zeros((len(r), len(i)));


def null_bias_rule(r, i, rb_avg, b = 0):
    return np.zeros((len(r),));


#a generic update rule, flexibly determined by the 'objective' and 'rule_id' arguments.
def generic_rule(objective, rule_id, r, i, target, rx_avg, b = 0, weights = None, weight_l2 = 0):
    #objective is a scalar reward function indicating the performance of the network
    #r is the firing rates for the network
    #i is the inputs to the network
    #rx_avg is the necessary average coactivation, be it the biases, the inputs, or the rates
    #b is an optional argument for bias updates.
    #target_extra is for combining objectives
    if (rule_id == 'weight'):
        coactivation = np.outer(r,r);
    elif(rule_id == 'input'):
        coactivation = np.outer(r, i);
    elif(rule_id == 'bias'):
        coactivation = r;
    obj = objective(r, target) 
    if (not(weights is None)):
        update = obj * (coactivation - rx_avg) - weight_l2 * weights #optionally add in an L2 regularizer to the weight update
    else:
        update = obj * (coactivation - rx_avg);
    return update

#define the MSE weight updates in terms of the generic update rule
MSE_weight = lambda r, i, target, rx_avg, decoder, params: generic_rule(lambda r, target: MSE_objective(params, decoder, r, target), 'weight', r, i, target, rx_avg);
MSE_bias = lambda r, i, target, rx_avg, b, decoder, params: generic_rule(lambda r, target: MSE_objective(params, decoder, r, target), 'bias', r, i, target, rx_avg, b=b);
MSE_input = lambda r, i, target, rx_avg, decoder, params: generic_rule(lambda r, target: MSE_objective(params, decoder, r, target), 'input', r, i, target, rx_avg);

#define the MSE l1 weight updates in terms of the generic update rule
MSE_l1_weight = lambda r, i, target, rx_avg, weights, decoder, params: generic_rule(lambda r, target: MSE_l1(params, decoder, r, target), 'weight', r, i, target, rx_avg, weights = weights);
MSE_l1_bias = lambda r, i, target, rx_avg, b, weights, decoder, params: generic_rule(lambda r, target: MSE_l1(params, decoder, r, target), 'bias', r, i, target, rx_avg, b=b, weights = weights, weight_l2 = params['weight_l2']);
MSE_l1_input = lambda r, i, target, rx_avg, weights, decoder, params: generic_rule(lambda r, target: MSE_l1(params, decoder, r, target), 'input', r, i, target, rx_avg, weights = weights, weight_l2 = params['weight_l2']);

#define the MSE l2 weight updates in terms of the generic update rule
MSE_l2_weight = lambda r, i, target, rx_avg, weights, decoder, params: generic_rule(lambda r, target: MSE_l2(params, decoder, r, target), 'weight', r, i, target, rx_avg, weights = weights, weight_l2 = params['weight_l2']);
MSE_l2_bias = lambda r, i, target, rx_avg, b, weights, decoder, params: generic_rule(lambda r, target: MSE_l2(params, decoder, r, target), 'bias', r, i, target, rx_avg, b=b, weights = weights, weight_l2 = params['bias_weight_l2']);
MSE_l2_input = lambda r, i, target, rx_avg, weights, decoder, params: generic_rule(lambda r, target: MSE_l2(params, decoder, r, target), 'input', r, i, target, rx_avg, weights = weights, weight_l2 = params['i_weight_l2']);

#define the classifier weight updates
classifier_weight = lambda r, i, target, rx_avg, weights, decoder, params: generic_rule(lambda r, target: classifier_vonmises_l12(params, decoder, r, target), 'weight', r, i, target, rx_avg, weights = weights, weight_l2 = params['weight_l2']);
classifier_bias = lambda r, i, target, rx_avg, b, weights, decoder, params: generic_rule(lambda r, target: classifier_vonmises_l12(params, decoder, r, target), 'bias', r, i, target, rx_avg, b = b, weights = weights, weight_l2 = params['bias_weight_l2']);
classifier_input = lambda r, i, target, rx_avg, weights, decoder, params: generic_rule(lambda r, target: classifier_vonmises_l12(params, decoder, r, target), 'input', r, i, target, rx_avg, weights = weights, weight_l2 = params['i_weight_l2']);

#define the weight update for the MSE/classifier combo
MSE_LL_weight = lambda r, i, target, rx_avg, weights, decoder, params: generic_rule(lambda r, target: MSE_LL_l2(params, decoder, r, target), 'weight', r, i, target, rx_avg, weights = weights, weight_l2 = params['weight_l2'])
MSE_LL_bias = lambda r, i, target, rx_avg, b, weights, decoder, params: generic_rule(lambda r, target: MSE_LL_l2(params, decoder, r, target), 'bias', r, i, target, rx_avg, b = b, weights = weights, weight_l2 = params['bias_weight_l2'])
MSE_LL_input = lambda r, i, target, rx_avg, weights, decoder, params: generic_rule(lambda r, target: MSE_LL_l2(params, decoder, r, target), 'input', r, i, target, rx_avg, weights = weights, weight_l2 = params['i_weight_l2'])

#define the weight update for the BCI paradigm
BCI_weight = lambda r, i, target, rx_avg, weights, decoder, params: generic_rule(lambda r, target: BCI_objective(params, decoder, r, target), 'weight', r, i, target, rx_avg, weights = weights, weight_l2 = params['weight_l2'])
BCI_bias = lambda r, i, target, rx_avg, b, weights, decoder, params: generic_rule(lambda r, target: BCI_objective(params, decoder, r, target), 'bias', r, i, target, rx_avg, b = b, weights = weights, weight_l2 = params['bias_weight_l2'])
BCI_input = lambda r, i, target, rx_avg, weights, decoder, params: generic_rule(lambda r, target: BCI_objective(params, decoder, r, target), 'input', r, i, target, rx_avg, weights = weights, weight_l2 = params['i_weight_l2'])


#learning rule for the linear decoder weights
def MSE_decoder_rule(decoder, r, i, target, weight_l2 = 0):
    r_biased = np.concatenate((r, -1 * np.ones((1,))), axis = 0)
    r_biased = r;
    if (exp_params.normalize_output):
        r_biased = r_biased / np.linalg.norm(r)
    return -1 * np.outer((np.dot(decoder, r_biased) - target), r_biased) - weight_l2 * decoder

#learning rule for the logistic decoder weights
def log_likelihood_decoder_rule(decoder, r, i, target, weight_l2 = 0):
    return (target - logistic_decoder(decoder, r)) * r * 100 - weight_l2 * decoder

#learning rule for the mixed decoder weights
def MSE_LL_decoder_rule(decoder, r, i, target, weight_l2 = 0):
    #the 'decoder' contains 2 different decoders, one for each task. This update rule splits the decoders apart,
    #updates them separately, and then puts them back together
    decoder_MSE = decoder[0:(len(target)-1),:]
    decoder_LL = decoder[-1,:]
    MSE_update = exp_params.params['lambda_MSE'] * -1 * np.outer((np.dot(decoder_MSE, r) - target[0:(len(target)-1)]), r) - weight_l2 * decoder_MSE
    LL_update = exp_params.params['lambda_LL'] * np.reshape((target[-1] - logistic_decoder(decoder_LL, r)) * r * 100 - weight_l2 * decoder_LL, (1, len(r)))
    return np.concatenate((MSE_update, LL_update), axis = 0)

#%% Objective functions

def null_objective(params, decoder, r, i):
    return 0;

#standard MSE objective function
def MSE_objective(params, decoder, r, i):
    r_biased = np.concatenate((r, -1* np.ones((1,))), axis = 0);
    r_biased = r
    network_output = np.dot(decoder, r_biased);
    return -1 * np.dot((i - network_output).T, i-network_output);

#MSE objective function with additional optional l1 regularization
def MSE_l1(params, decoder, r, i):
    r_bias = r;
    network_output = np.dot(decoder, r_bias);
    lambda_l1 = params['lambda_l1'];
    return -1 * np.dot((i - network_output).T, i-network_output) - lambda_l1 * np.sum(np.abs(r));

#MSE objective function with additional optional l2 regularization
def MSE_l2(params, decoder, r, i):
    r_bias = r;
    if (exp_params.normalize_output):
        r_bias = r_bias/ np.linalg.norm(r)
    network_output = np.dot(decoder, r_bias);
    lambda_l2 = params['lambda_l2'];
    return -1 * np.dot((i - network_output).T, i-network_output) #- lambda_l2 * np.sum(r**2);

#objective: require the output of a predefined decoder to be greater than a threshold
def BCI_objective(params, decoder, r, i):
    network_output = decoder @ r;
    return int((network_output > params['bci_threshold']) and (network_output < params['bci_upper_bd']))

#objective function for classifying a vonmises input into two categories
def classifier_vonmises(params, decoder, r, target):
    network_output = logistic_decoder(decoder, r);
    return -1 * np.dot((target - network_output).T, target - network_output);

#same as above, but used the log-likelihood objective function and include l1 and l2 regularization on the firing rates
def classifier_vonmises_l12(params, decoder, r, target):
    tolerance = 1e-4
    lambda_l1 = params['lambda_l1']
    lambda_l2 = params['lambda_l2']
    network_output = logistic_decoder(decoder, r);
    if (network_output < tolerance): #prevent log(0) error
        network_output = tolerance;
    elif (1-network_output < tolerance):
        network_output = 1 - tolerance;
    return target * np.log(network_output) + (1-target) * np.log(1 - network_output) #- lambda_l1 * np.sum(np.abs(r)) - lambda_l2 * np.sum(r**2);

#MSE_LL: a combination of the classifier and mean-squared error objectives
def MSE_LL_l2(params, decoder, r, target, classify_only = False):
    target_MSE = target[0:(len(target)-1)] #split the objective up into both the classification and the representation
    target_LL = target[-1]
    tolerance = 1e-4
    network_output_LL = logistic_decoder(decoder[-1,:], r)
    network_output_MSE = np.dot(decoder[0:(len(decoder[:,0])-1),:], r)
    if (network_output_LL < tolerance): #prevent log(0) error
        network_output_LL = tolerance;
    elif (1-network_output_LL < tolerance):
        network_output_LL = 1 - tolerance;
    MSE_obj = -1 * np.dot((target_MSE - network_output_MSE).T, target_MSE -network_output_MSE)
    LL_obj = target_LL * np.log(network_output_LL) + (1-target_LL) * np.log(1 - network_output_LL)
    if (classify_only):
        return params['lambda_LL'] * LL_obj
    else:
        return params['lambda_MSE'] * MSE_obj + params['lambda_LL'] * LL_obj
#%% Input generators

#each distribution needs a 'input', for generating stimuli that switch sharply in step functions
#an 'input_sampler' for generating a large number of samples from the distribution
#and an 'update' which takes steps according to a Gibbs sampler.
    
# Normal Distribution
def basic_input(input_params):
    #input params contains: T, dt, switch_period, mu_in, sigma_in, input_dim, trial_num
    #generates normally-distributed stimuli that stay constant w/ frequency switch_period

    stim_num = int(input_params['T']/input_params['switch_period']);
    stims = np.random.normal(loc = input_params['mu_in'], scale = input_params['sigma_in'], 
                             size = (input_params['trial_num'],input_params['input_dim'], stim_num))
    inputs = np.repeat(stims, int(input_params['switch_period']/input_params['dt']), axis = 2);

    return inputs

def basic_input_sampler(input_params, size):
    return np.random.normal(loc = input_params['mu_in'], scale = input_params['sigma_in'], size = size);

def normal_f(params, input_params, i_prev, rand_var):
    di = -(i_prev - input_params['mu_in'])/input_params['sigma_in']**2 * input_params['sigma_sampler']**2
    i_current = i_prev + di * params['dt'] + input_params['sigma_sampler'] * rand_var
    return i_current

# Von Mises Distribution -- projected onto 'input_dim' unit vectors tiling the sphere
def basic_vonmises(input_params):
    stim_num = int(input_params['T']/input_params['switch_period']);
    stim_angles = np.random.vonmises(input_params['mu_in'], input_params['kappa_in'], 
                               size = (input_params['trial_num'], stim_num));
    x_y_pos = np.array([np.cos(stim_angles), np.sin(stim_angles)])*input_params['stim_gain'];
    return x_y_pos

def basic_vonmises_sampler(input_params, size):
    stim_angles = np.random.vonmises(input_params['mu_in'], input_params['kappa_in'], size = size);
    x_y_pos = np.array([np.cos(stim_angles), np.sin(stim_angles)]);
    return x_y_pos, stim_angles

#function to calculate a new position on a unit circle from an old one based on a vonmises MCMC sampler
def vonmises_update(params, input_params, i_prev, theta_prev, rand_var):
    dtheta = (input_params['kappa_in'] * -1 * np.sin(theta_prev - input_params['mu_in'])) * input_params['sigma_sampler']**2
    theta_new = theta_prev + dtheta * params['dt'] + input_params['sigma_sampler'] * rand_var;
    if (theta_new < -1 * np.pi):
        theta_new = np.pi + (theta_new + np.pi)
    if (theta_new > np.pi):
        theta_new = -1 * np.pi + (theta_new - np.pi);
    x_y_pos = np.array([np.cos(theta_new), np.sin(theta_new)]) #* input_params['stim_gain'] + input_params['offset'];
    i_new = np.ndarray.flatten(x_y_pos)
    return i_new, theta_new

# Von Mises Distribution -- projected onto 'input_dim' ReLU's on the unit circle
# fixed_samples: for determining the statistics of trained neurons, where samples are drawn from a fixed input distribution
def mv_vonmises(input_params, fixed_samples = False, test = False):
    if (test):
        num = input_params['test_num']
    else:
        num = input_params['trial_num']
    theta = np.arange(-1*np.pi, np.pi, 2*np.pi/input_params['input_dim']);
    projection_mat = np.array([np.cos(theta), np.sin(theta)]).T
    stim_num = int(input_params['T']/input_params['switch_period']);
    if (fixed_samples):
        stim_angles = np.random.choice(theta, size = (num, stim_num))
    else:
        stim_angles = np.random.vonmises(input_params['mu_in'], input_params['kappa_in'], 
                               size = (num, stim_num));
    x_y_pos = np.array([np.cos(stim_angles), np.sin(stim_angles)]);
    stims = np.zeros((input_params['input_dim'], num, stim_num));
    for ii in np.arange(0, num):
        for jj in np.arange(0, stim_num):
            stims[:,ii,jj] = np.dot(projection_mat, x_y_pos[:,ii,jj]);
    inputs = np.repeat(stims, int(input_params['switch_period']/input_params['dt']), axis = 2);
    inputs = np.transpose(inputs, axes = (1,0,2));
    inputs = np.maximum(0, inputs);
    stim_angles = np.reshape(stim_angles, (num, 1, stim_num))
    theta_i = np.repeat(stim_angles, int(input_params['switch_period']/input_params['dt']), axis = 2)
    return inputs, theta_i

def mv_vonmises_sampler(input_params, size):
    theta = np.arange(-1*np.pi, np.pi, 2*np.pi/input_params['input_dim']);
    projection_mat = np.array([np.cos(theta), np.sin(theta)]).T
    stim_angles = np.random.vonmises(input_params['mu_in'], input_params['kappa_in'], size = size);
    x_y_pos = np.array([np.cos(stim_angles), np.sin(stim_angles)]);
    samples = np.dot(projection_mat, x_y_pos);
    samples = np.maximum(0, samples);
    return samples, stim_angles

#Generates a Langevin sampling update for the mv_vonmises distribution
def mv_vonmises_update(params, input_params, i_prev, theta_prev, rand_var):
    angles = np.arange(-1*np.pi, np.pi, 2*np.pi/input_params['input_dim']);
    projection_mat = np.array([np.cos(angles), np.sin(angles)]).T
    dtheta = (input_params['kappa_in'] * -1 * np.sin(theta_prev - input_params['mu_in'])) * input_params['sigma_sampler']**2
    theta_new = theta_prev + dtheta * params['dt'] + input_params['sigma_sampler'] * rand_var;
    if (theta_new < -1 * np.pi):
        theta_new = np.pi + (theta_new + np.pi)
    if (theta_new > np.pi):
        theta_new = -1 * np.pi + (theta_new - np.pi);
    x_y_pos = np.array([np.cos(theta_new), np.sin(theta_new)]) #* input_params['stim_gain'] + input_params['offset'];
    x_y_pos = np.ndarray.flatten(x_y_pos)
    i_new = np.dot(projection_mat, x_y_pos);
    i_new = np.maximum(0, i_new);
    return i_new, theta_new

#code for the schematics. The point is literally to be able to plot bump-shaped curves
def bump_fn(x, center, sigma):
    response = np.exp(-(x - center)**2/ sigma**2)
    response = response/ np.amax(response)
    return response

#a Gibbs sampler for the Ornstein-Uhlenbeck process
def OU_process(params):
    dt = 0.01 #time step
    T = 10 #total time per trial
    x = np.zeros((int(T/dt)))
    t = np.arange(0, T, dt);
    x_0 = params['mean'][0];
    x[0] = x_0;
    for ii in np.arange(1, int(T/dt), 1):
        dx = -(x[ii-1] - params['mean'][ii])
        x[ii] = x[ii-1] + dx * dt + params['sigma'][ii] * np.random.normal(0, np.sqrt(dt));
    return x, t

#%% Basic Data Analysis Functions
    
#calculate the sliding mean of a 1D vector
#var: the variable over which to perform the mean
#window: window size for computing the average
#downsample: amount to downsample by
def sliding_mean(var, window, downsample):
    avg_mask = np.ones(int(window/downsample))/(int(window/downsample))
    sm = np.convolve(var[::downsample], avg_mask, 'same')
    #pad untraversed values in sliding_mean with the closest mean value
    #sliding_mean[0:(half_window-1)] = sliding_mean[half_window];
    #sliding_mean[(len(sliding_mean)-half_window)+1::] = sliding_mean[(len(sliding_mean)-half_window)]
    return sm

#calculate the sliding standard deviation of a 1D vector
#var: the variable over which to calculate the standard deviation.
def sliding_std(var, window):
    half_window = int(window/2);
    std_slide = np.zeros(var.shape);
    for ii in range(0, len(std_slide)):
        std_slide[ii] = np.std(var[np.maximum(0,ii-half_window):np.minimum(ii+half_window,len(std_slide))])

    #pad untraversed values in sliding_std with the closest STD value
    #sliding_std[0:(half_window-1)] = sliding_std[half_window];
    #sliding_std[(len(sliding_std)-half_window)+1::] = sliding_std[(len(sliding_std)-half_window)]
    return std_slide


#calculate noise correlations as a function of stimulus
#theta_record: the stimulus value for each moment in time
#theta_range: the list of stimuli used in the test set
#r_record: the firing rate activity
#returns a len(theta_range) x N x N array containing the noise correlations as a function of stimulus
def noise_correlations(theta_record, theta_range, r_record, normalize = True):
    N = len(r_record[0,:,0]);
    noise_correlations = np.zeros((len(theta_range), N,N));
    r_reshape = np.reshape(np.transpose(r_record, (1, 0, 2)), (N, r_record.shape[0]*r_record.shape[2]));
    theta_reshape = np.reshape(np.transpose(theta_record, (1,0,2)), (1, theta_record.shape[0]*theta_record.shape[2]));
    for ii in range(0, len(theta_range)):
        idx = np.where(theta_reshape == theta_range[ii])[1]
        idx = idx[np.where(np.mod(idx, 2000) > 40)[0]]
        if (normalize):
            noise_correlations[ii,:,:] = np.cov(r_reshape[:,idx])/np.outer(np.std(r_reshape[:,idx], axis = 1),np.std(r_reshape[:,idx], axis = 1)) #calculate an actual correlation
        else:
            noise_correlations[ii,:,:] = np.cov(r_reshape[:,idx]) #calculate only a covariance instead
    return noise_correlations    

#calculate the signal correlation between different neurons across all stimuli
#noise correlations fix a particular stimulus, signal correlations are across all stimuli.
def signal_correlations(r_record):
    N = len(r_record[0,:,0]);
    r_reshape = np.reshape(np.transpose(r_record, (1, 0, 2)), (N, r_record.shape[0]*r_record.shape[2]));
    return np.cov(r_reshape)/np.outer(np.std(r_reshape, axis = 1), np.std(r_reshape, axis = 1))

#calculate the stimulus-conditioned variance
#pcas: the matrix of eigenvectors
def stimulus_variance_pca(theta_record, theta_range, r_record, pcas, decoder):
    N = len(r_record[0,:,0]);
    r_reshape = np.reshape(np.transpose(r_record, (1, 0, 2)), (N, r_record.shape[0]*r_record.shape[2]));
    theta_reshape = np.reshape(np.transpose(theta_record, (1, 0, 2)), (1, theta_record.shape[0]*theta_record.shape[2]));
    r_projected = pcas @ r_reshape;
    pca_projected = decoder @ pcas.T;
    pca_variance = np.zeros((len(pcas[:,0]), len(theta_range)));
    decoder_variance = np.zeros((len(decoder[:,0]), len(pcas[:,0]),len(theta_range)))
    for ii in np.arange(0, len(theta_range)):
        idx = np.where(theta_reshape == theta_range[ii])[1]
        pca_variance[:,ii] = np.var(r_projected[:,idx], axis = 1)
        for jj in range(0,2):
            decoder_projected = pca_projected[jj,:].reshape(10,1) * r_projected[:,idx]
            decoder_variance[jj,:,ii] = np.var(decoder_projected, axis = 1)
    return pca_variance, decoder_variance

#calculate the cosine angle between two vectors or matrices
def cosine_angle(mat_1, mat_2):
    mat_1_flat = np.ndarray.flatten(mat_1)
    mat_2_flat = np.ndarray.flatten(mat_2)
    return np.dot(mat_1_flat, mat_2_flat)/np.linalg.norm(mat_1_flat)/np.linalg.norm(mat_2_flat)
#calculate the entropy (or total variance) of the stimulus-conditioned fluctuations projected onto the decoder's axes, divided by the total stimulus-conditioned entropy
#calculate across stimuli and return a vector of the fractions
def noise_volume(theta_record, theta_range, r_record, decoder, entropy = True):
    N = r_record.shape[1]
    U, S, V_T = np.linalg.svd(decoder);
    decoder_dim_num = decoder.shape[0];
    N = len(r_record[0,:,0]);
    decoder_variance_overlap = np.zeros((len(theta_range),decoder_dim_num, N))
    e_vals = np.zeros((len(theta_range), N))
    r_reshape = np.reshape(np.transpose(r_record, (1, 0, 2)), (N, r_record.shape[0]*r_record.shape[2]));
    theta_reshape = np.reshape(np.transpose(theta_record, (1, 0, 2)), (1, theta_record.shape[0]*theta_record.shape[2]));
    volume_fraction = np.zeros((len(theta_range)))
    total_volume = np.zeros((len(theta_range)))
    projected_volume = np.zeros((len(theta_range)))
    for ii in np.arange(0, len(theta_range)):
        idx = np.where(theta_reshape == theta_range[ii])[1]
        r_stim_conditioned = r_reshape[:,idx]
        cov_conditioned = np.cov(r_stim_conditioned)
        U_cov, S_cov, V_T_cov = np.linalg.svd(cov_conditioned)
        cov_conditioned_max = np.cov(V_T_cov[0:decoder_dim_num,:] @ r_stim_conditioned)
        if (entropy):
            total_volume[ii] = cov_conditioned_max.shape[0]/2*(1 + np.log(2*np.pi)) + 1/2 * np.log(np.linalg.det(cov_conditioned_max))
        else: #use volume of covariance ellipsoid
            if (decoder_dim_num > 1):
                total_volume[ii] = np.linalg.det(cov_conditioned_max)
            else:
                total_volume[ii] = cov_conditioned_max
        r_stim_conditioned_projected = V_T[0:decoder_dim_num,:] @ r_stim_conditioned
        cov_projected_conditioned = np.cov(r_stim_conditioned_projected) #approximate the entropy with a Gaussian
        if (entropy):
            projected_volume[ii] = cov_projected_conditioned.shape[0]/2*(1 + np.log(2*np.pi)) + 1/2 * np.log(np.linalg.det(cov_projected_conditioned))
        else: #use volume of covariance ellipsoid
            if (decoder_dim_num > 1):
                projected_volume[ii] = np.linalg.det(cov_projected_conditioned)
            else:
                projected_volume[ii] = cov_projected_conditioned
        volume_fraction[ii] = projected_volume[ii]/total_volume[ii]
        decoder_variance_overlap[ii,:,:] = V_T[0:decoder_dim_num,:] @ V_T_cov.T
        e_vals[ii,:] = S_cov
    return volume_fraction, total_volume, projected_volume, decoder_variance_overlap, e_vals

#calculates the cosine angle between the 1st 2 PCA axes and the decoder axes for 
def noise_angle(theta_record, theta_range, r_record, decoder):
    N = r_record.shape[1]
    U, S, V_T = np.linalg.svd(decoder);
    decoder_dim_num = decoder.shape[0];
    N = len(r_record[0,:,0]);
    angle = np.zeros((len(theta_range),))
    r_reshape = np.reshape(np.transpose(r_record, (1, 0, 2)), (N, r_record.shape[0]*r_record.shape[2]));
    theta_reshape = np.reshape(np.transpose(theta_record, (1, 0, 2)), (1, theta_record.shape[0]*theta_record.shape[2]));
    for ii in np.arange(0, len(theta_range)):
        idx = np.where(theta_reshape == theta_range[ii])[1]
        r_stim_conditioned = r_reshape[:,idx]
        cov_conditioned = np.cov(r_stim_conditioned)
        U_cov, S_cov, V_T_cov = np.linalg.svd(cov_conditioned)
        pca_mat = V_T_cov[0:decoder_dim_num,:]
        decoder_mat = V_T[0:decoder_dim_num,:]
        angle[ii] = cosine_angle(pca_mat, decoder_mat)
    return angle

def noise_volume_bci(r_record, decoder):
    U, S, V_T = np.linalg.svd(decoder);
    decoder_dim_num = decoder.shape[0];
    N = len(r_record[:,0]);
    decoder_variance_overlap = np.zeros((decoder_dim_num, N))
    e_vals = np.zeros((N,))


    cov_conditioned = np.cov(r_record)
    U_cov, S_cov, V_T_cov = np.linalg.svd(cov_conditioned)
    cov_conditioned_max = np.cov(V_T_cov[0:decoder_dim_num,:] @ r_record)

    if (decoder_dim_num > 1):
        total_volume = np.linalg.det(cov_conditioned_max)
    else:
        total_volume = cov_conditioned_max
    r_stim_conditioned_projected = V_T[0:decoder_dim_num,:] @ r_record
    cov_projected_conditioned = np.cov(r_stim_conditioned_projected) #approximate the entropy with a Gaussian

    if (decoder_dim_num > 1):
        projected_volume = np.linalg.det(cov_projected_conditioned)
    else:
        projected_volume = cov_projected_conditioned
    volume_fraction = projected_volume/total_volume
    decoder_variance_overlap[:,:] = V_T[0:decoder_dim_num,:] @ V_T_cov.T
    e_vals[:] = S_cov
    return volume_fraction, total_volume, projected_volume, decoder_variance_overlap, e_vals
#calculate the d' for the two distributions above/below the classification boundary (here set to be 0)
#note: the intention is for r_record to be calculated from time-varying test stimuli, NOT from fixed test stimuli, b/c this biases results weirdly
def d_prime_calc(r_record, decoder, theta_record):
    N = r_record.shape[1]
    r_reshape = np.reshape(np.transpose(r_record, (1, 0, 2)), (N, r_record.shape[0]*r_record.shape[2]));
    theta_reshape = np.reshape(np.transpose(theta_record, (1, 0, 2)), (1, theta_record.shape[0]*theta_record.shape[2]));
    idx_up = np.where(theta_reshape > 0)[1]
    idx_down = np.where(theta_reshape < 0)[1]
    #calculate the output mean and variance projected onto the linear decoder
    r_mean_up = np.mean( decoder @ r_reshape[:,idx_up])
    r_mean_down = np.mean(decoder @ r_reshape[:,idx_down])
    r_var_up = np.var(decoder @ r_reshape[:,idx_up])
    r_var_down = np.var(decoder @ r_reshape[:,idx_down])
    mean_diff = np.abs(np.mean(r_mean_up) - np.mean(r_mean_down))
    mean_std = np.sqrt(1/2 * (r_var_up + r_var_down))
    d_prime = mean_diff/mean_std
    return d_prime, mean_diff, mean_std
    
#break down the MSE into bias and variance terms between the output and the target
def stimulus_bias_variance(theta_record, theta_range, o_record, target_record):
    N = len(o_record[0,:,0]);
    o_reshape = np.reshape(np.transpose(o_record, (1, 0, 2)), (N, o_record.shape[0]*o_record.shape[2]));
    target_reshape = np.reshape(np.transpose(target_record, (1, 0, 2)), (N, target_record.shape[0]*target_record.shape[2]))
    theta_reshape = np.reshape(np.transpose(theta_record, (1, 0, 2)), (1, theta_record.shape[0]*theta_record.shape[2]));
    bias = np.zeros((N, len(theta_range)));
    variance = np.zeros((N, len(theta_range)));
    for ii in np.arange(0, len(theta_range)):
        idx = np.where(theta_reshape == theta_range[ii])[1]
        bias[:,ii] = np.mean(target_reshape[:,idx] - o_reshape[:,idx], axis = 1)
        variance[:,ii] = np.var(o_reshape[:,idx], axis = 1)
    return bias, variance

#calculate the error as a function of stimulus angle for either MSE or log-likelihood
def stimulus_error(theta_record, theta_range, o_record, target_record, objective = 'represent'):
    N = len(o_record[0,:,0]);
    o_reshape = np.reshape(np.transpose(o_record, (1, 0, 2)), (N, o_record.shape[0]*o_record.shape[2]));
    target_reshape = np.reshape(np.transpose(target_record, (1, 0, 2)), (N, target_record.shape[0]*target_record.shape[2]))
    theta_reshape = np.reshape(np.transpose(theta_record, (1, 0, 2)), (1, theta_record.shape[0]*theta_record.shape[2]));
    error = np.zeros((1, len(theta_range)));
    for ii in np.arange(0, len(theta_range)):
        idx = np.where(theta_reshape == theta_range[ii])[1]
        if objective == 'represent': #if represent, calculate error according to MSE
            error[:,ii] = np.mean(np.sum((target_reshape[:,idx] - o_reshape[:,idx])**2, axis = 0))
        elif objective =='classify': #if classifier, calculate error according to log-likelihood
            error[:,ii] = target_reshape[:,idx] * np.log(o_reshape[:,idx]) + (1-target_reshape[:,idx]) * np.log(1 - o_reshape[:,idx])
    return error


#calculate the mean excitatory current and the mean recurrent current each neuron receives across stimuli
#this is part of the analysis trying to see the amount of recurrent activation as a function of parameters
def mean_current_breakdown(r_record, i_record, biases, weights, i_weights, theta_record, theta_range):
    N = len(r_record[0,:,0])
    input_dim = len(i_record[0,:,0])
    r_reshape = np.reshape(np.transpose(r_record, (1, 0, 2)), (N, r_record.shape[0]*r_record.shape[2]));
    i_reshape = np.reshape(np.transpose(i_record, (1, 0, 2)), (input_dim, i_record.shape[0]*i_record.shape[2]));
    theta_reshape = np.reshape(np.transpose(theta_record, (1, 0, 2)), (1, theta_record.shape[0]*theta_record.shape[2]));
    bias_current = np.zeros((N, len(theta_range)));
    r_current = np.zeros((N, len(theta_range)));
    i_current = np.zeros((N, len(theta_range)));
    for ii in np.arange(0, len(theta_range)):
        idx = np.where(theta_reshape==theta_range[ii])[1]
        bias_current[:,ii] = np.abs(biases)
        #r_current[:,ii] = np.mean(np.abs(weights @ r_reshape[:,idx]), axis = 1)
        #i_current[:,ii] = np.mean(np.abs(i_weights @ i_reshape[:,idx]), axis = 1)
        r_current[:,ii] = np.mean(weights @ r_reshape[:,idx], axis = 1)
        i_current[:,ii] = np.mean(i_weights @ i_reshape[:,idx], axis = 1)
    return r_current, i_current, bias_current

#calculate the mean error in the test set, weighting by the probability of the error actually occurring
def mean_error(obj_test, theta_test, mu, kappa):
    theta = np.ndarray.flatten(theta_test)
    K = len(np.unique(theta))
    obj = np.ndarray.flatten(obj_test)
    weight = vonmises.pdf(theta, kappa, loc = mu)
    mean = 1/len(obj) * np.sum(obj * weight * K) #importance sampling of objective
    return mean
    
#calculate the basic covariance for the r_record output from normative_plasticity
def covariance(r_record):
    N = len(r_record[0,:,0]);
    r_reshape = np.reshape(np.transpose(r_record, (1, 0, 2)), (N, r_record.shape[0]*r_record.shape[2]));
    return np.cov(r_reshape)
#calculate the basic mean for the r_record output from normative plasticity
def fr_mean(r_record):
    N = len(r_record[0,:,0]);
    r_reshape = np.reshape(np.transpose(r_record, (1, 0, 2)), (N, r_record.shape[0]*r_record.shape[2]));
    return np.mean(r_reshape, axis = 1)

#divide each variable by its mean, and then calculate the covariance
def normalized_covariance(r_record):
    return covariance(r_record)/np.outer(np.abs(r_record), np.abs(fr_mean(r_record)))

#calculate variance based on the output, rather than the input neurons themselves
def decoder_variance(theta_record, theta_range, r_record, decoder):
    N = len(r_record[0,:,0]);
    I = len(decoder[:,0]);
    covariance = np.zeros((len(theta_range), I,I));
    r_reshape = np.reshape(np.transpose(r_record, (1, 0, 2)), (N, r_record.shape[0]*r_record.shape[2]));
    output = np.dot(decoder, r_reshape);
    theta_reshape = np.reshape(np.transpose(theta_record, (1,0,2)), (1, theta_record.shape[0]*theta_record.shape[2]));
    for ii in range(0, len(theta_range)):
        idx = np.where(theta_reshape == theta_range[ii])[1]
        idx = idx[np.where(np.mod(idx, 2000) > 40)[0]]
        covariance[ii,:,:] = np.cov(output[:,idx])
    return covariance

#calculate the variance of the noise correlations in the direction of the decoder, compared to the variance in the other dimensions
def nc_decoder_variance(nc, decoder):
    unit_decoder = decoder/np.linalg.norm(decoder);
    decoder_var = np.zeros((nc.shape[0],))
    for ii in np.arange(0, len(decoder_var)):
        decoder_var[ii] = np.linalg.norm(unit_decoder @ nc[ii,:,:])**2
    return decoder_var

#check the direction the recurrent inputs to neurons biases the network in
#also check if the biases can be matched by theoretical predictions
def recurrent_bias(r_record, input_record, theta_record, theta_range, weights, i_weights, decoder):
    N = len(r_record[0,:,0])
    I = len(input_record[0,:,0])
    r_reshape = np.reshape(np.transpose(r_record, (1, 0, 2)), (N, r_record.shape[0]*r_record.shape[2]));
    input_reshape = np.reshape(np.transpose(input_record, (1, 0, 2)), (I, input_record.shape[0]*input_record.shape[2]))
    theta_reshape = np.reshape(np.transpose(theta_record, (1,0,2)), (1, theta_record.shape[0]*theta_record.shape[2]));
    avg_recurrent_inputs = np.zeros((len(theta_range), N))
    avg_r = np.zeros((len(theta_range), N))
    avg_inputs = np.zeros((len(theta_range),N))
    avg_output = np.zeros((len(theta_range),decoder.shape[0]))
    var_output = np.zeros((len(theta_range),))
    
    decoded_bias_recurrent = np.zeros((len(theta_range), decoder.shape[0]))
    decoded_bias_input = np.zeros((len(theta_range), decoder.shape[0]))
    
    for ii in np.arange(0, len(theta_range)):
        idx = np.where(theta_reshape == theta_range[ii])[1]
        recurrent_inputs = weights @ r_reshape[:,idx] #calculate the current contribution of the recurrent weights
        inputs = i_weights @ input_reshape[:,idx] #calculate the current contribution of the inputs
        avg_inputs[ii,:] = np.mean(inputs, axis = 1) #calculate the average input current to each neuron (for the test case, the inputs should be constant)
        avg_r[ii,:] = np.mean(r_reshape[:,idx], axis = 1)
        avg_recurrent_inputs[ii,:] = np.mean(recurrent_inputs, axis = 1) #calculate the average recurrent input to each neuron
        decoded_bias_recurrent[ii,:] = decoder @ avg_recurrent_inputs[ii,:] #project the recurrent input onto the decoder
        decoded_bias_input[ii,:] = decoder @ avg_inputs[ii,:] #project the input current onto the decoder
        avg_output[ii,:] = decoder @ np.mean(r_reshape[:,idx], axis = 1)
        output_mean_subtracted = decoder @ r_reshape[:,idx] - avg_output[[ii],:].T
        var_output[ii] = np.trace(np.cov(output_mean_subtracted))
        
    return decoded_bias_recurrent, decoded_bias_input, avg_output, var_output, avg_r
        
        
# extract tuning curves from a test data set
def tuning_curves(theta_record, theta_range, r_record):
    N = len(r_record[0,:,0])
    tuning_curves = np.zeros((len(theta_range), N))
    tuning_curves_std = np.zeros((len(theta_range), N))
    r_reshape = np.reshape(np.transpose(r_record, (1, 0, 2)), (N, r_record.shape[0]*r_record.shape[2]));
    theta_reshape = np.reshape(np.transpose(theta_record, (1,0,2)), (1, theta_record.shape[0]*theta_record.shape[2]));
    for ii in range(0, len(theta_range)):
        idx = np.where(theta_reshape == theta_range[ii])[1]
        tuning_curves[ii,:] = np.mean(r_reshape[:,idx], axis = 1)
        tuning_curves_std[ii,:] = np.std(r_reshape[:,idx], axis = 1)
    return tuning_curves, tuning_curves_std


#sort the noise correlation array by the maximum or minimum angle response
def sorted_noise_correlations(nc, tc, max_min = 'max'):
    if (max_min == 'max'):
        tc_ext = np.argmax(tc, axis = 0);
    elif (max_min == 'min'):
        tc_ext = np.argmin(tc, axis = 0);
    sorted_idx = np.argsort(tc_ext);
    sorted_nc = nc[:,sorted_idx, :];
    sorted_nc = sorted_nc[:,:, sorted_idx]
    return sorted_nc

#calculate the coefficient of variation for each neuron, for each stimulus, and sort the neurons by their greatest influence on the decoder
def sorted_stimulus_variance(r_record, tc, theta_record, theta_range, decoder):
    N = len(r_record[0,:,0])
    CV = np.zeros((N, len(theta_range)))
    r_reshape = np.reshape(np.transpose(r_record, (1, 0, 2)), (N, r_record.shape[0]*r_record.shape[2]));
    theta_reshape = np.reshape(np.transpose(theta_record, (1,0,2)), (1, theta_record.shape[0]*theta_record.shape[2]));
    tc_ext = np.argmax(tc, axis = 0);
    sorted_idx = np.argsort(tc_ext);
    r_sorted = r_reshape[sorted_idx, :]
    for ii in range(0, len(theta_range)):
        idx = np.where(theta_reshape == theta_range[ii])[1]
        CV[:,ii] = np.std(r_sorted[:,idx], axis = 1)/ np.abs(np.mean(r_sorted[:,idx], axis = 1))
    return CV
#calculate the confusion matrix for the classifier
def confusion_matrix(o_test, target_test):
    over_threshold = o_test > 0.5;
    under_threshold = np.logical_not(over_threshold);
    total = target_test.size
    target_flipped = (target_test - 1) * -1
    TP = np.sum(target_test[over_threshold])/total #true positives
    TN = np.sum(target_flipped[under_threshold])/total #true negatives
    FP = np.sum(target_flipped[over_threshold])/total #false positives
    FN = np.sum(target_test[under_threshold])/total #false negatives
    return np.array([[FN, TP], [TN, FP]])

#energy,distance = stimulus_dependent_energy(test_reload_2['r_test_2'], test_reload_2['theta_test_2'], test_reload_2['i_test_2'], network)
#stim_var, decoder_var = stimulus_variance_pca(theta_test, theta_range, r_test, network.pcas[0].components_, decoder_new)
#bias, variance = stimulus_bias_variance(theta_test, theta_range, o_test, target_test)
#r_current, i_current, bias_current = mean_current_breakdown(r_test, i_test, biases, weights, i_weights, theta_test, theta_range)
#tc, tc_std = tuning_curves(theta_test, np.unique(theta_test), r_test)   
#ssv = sorted_stimulus_variance(r_test, tc, theta_test, np.unique(theta_test), decoder)
#plt.imshow(np.log(np.mean(nc_sorted, axis = 0)), origin = 'lower')
#nc = noise_correlations(theta_test, np.unique(theta_test), r_test)
#nc_sorted = sorted_noise_correlations(nc, tc, max_min = 'max')
#dv = decoder_variance(theta_test, np.unique(theta_test), r_test, network.decoder)
#samples, stim_angles = mv_vonmises_orientation_update(params, input_params, b, a, 0.5)
# -*- coding: utf-8 -*-
"""
pl_plot_generator
A file to generate a single plot of a network training session for easy viewing

@author: Colin
"""
import numpy as np
import matplotlib.pyplot as plt
import time
import pl_exp_params_cr as exp_params
import pl_plotters_cr as plotters
import pl_support_functions_cr as sf

#%% vanilla representation and classification plots

#classifier
if (exp_params.objective == 'classifier'):
    #objective plot
    fig, axes = plt.subplots(1, figsize = (5, 5));
    plotters.objective_plot(np.concatenate((obj_record[0:-1,:], obj_record_2[0:-1,:]), axis = 0), axes)
    plt.title('objective')
    
    
    #in_out_compare
    fig, axes = plt.subplots(1, figsize = (5, 5));
    trial_num = 247
    test_num = 9
    test_range = np.arange(test_num, test_num + 1);
    trial_range = np.arange(trial_num, trial_num + 1);
    dim_range = np.arange(0,1);
    time_vec = np.arange(0, params['T'], params['dt']*params['record_res'])
    plotters.in_out_compare(time_vec, o_record_2, target_record_2, trial_range, dim_range, axes);
    
    
    #tuning curves
    fig, axes = plt.subplots(1, figsize = (5,5));
    tc_2, tc_std_2 = sf.tuning_curves(theta_test_2, np.unique(theta_test_2), r_test_2)
    tc_compare_classifier = tc_2
    theta_compare_classifier = theta_test_2
    obj_compare_classifier = obj_test_2
    plotters.tuning_curves_plotter(np.unique(theta_test_2), tc_2, axes, discrim_threshold = 0)

if (exp_params.objective == 'linear'):
    plt.show()
    
    if (not(exp_params.network_load)):
        fig, axes = plt.subplots(1, figsize = (5, 5));
        #objective plot
        plotters.objective_plot(np.concatenate((obj_record[0:-1,:], obj_record_2[0:-1,:]), axis = 0), axes)
    
        #in_out_compare
        fig, axes = plt.subplots(1, figsize = (5, 5));
        trial_num = 246#245
        test_num = 9
        test_range = np.arange(test_num, test_num + 1);
        trial_range = np.arange(trial_num, trial_num + 1);
        dim_range = np.arange(0,1);
        time_vec = np.arange(0, params['T'], params['dt']*network.params['record_res'])
        plotters.in_out_compare(time_vec, o_record_2, target_record_2, trial_range, dim_range, axes);
    
    #output prior to learning
    fig, axes = plt.subplots(1, figsize = (5,5))
    total_variance_pre = plotters.circle_compare(o_test_2, target_test_2, 450, axes)
    
    #tuning curves
    fig, axes = plt.subplots(1, figsize = (5,5));
    tc_2, tc_std_2 = sf.tuning_curves(theta_test_2, np.unique(theta_test_2), r_test_2)
    tc_compare_represent = tc_2 #save the tuning curves for comparison
    theta_compare_represent = theta_test_2
    obj_compare_represent = obj_test_2
    plotters.tuning_curves_plotter(np.unique(theta_test_2), tc_2, axes, discrim_threshold = 0)

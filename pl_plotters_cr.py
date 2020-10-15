# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 02:10:33 2018
pl_plotters.py

collection of functions for plotting figures for the normative_plasticity
code base.

Includes:
1. Decorators to make plots pretty
2. Core plotting functions
@author: Colin
"""
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pl_exp_params_cr as exp_params
from matplotlib.patches import Polygon
import pl_support_functions_cr as sf
from scipy.stats import vonmises
from scipy.interpolate import UnivariateSpline
from sklearn.linear_model import LinearRegression
import matplotlib.cm as mplcm
import matplotlib.colors as colors
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KernelDensity
import matplotlib as mpl
from scipy import linalg
from scipy import stats

#%% Decorators    
#decorator to adjust the parameters of a plot to give it more professional style
def pretty(plotter):
    def func_wrapper(*args, **kwargs):
        font = {'family' :'normal',
                'weight': 'bold',
                'size': 30}
        plt.rc('font', **font)
        plt.rc('legend', fontsize = 30)
        plt.rc('axes', labelsize = 30)
        plt.rc('xtick', labelsize = 10)
        plt.rc('ytick', labelsize = 10)
        #plt.rcParams.update({'font.size': 40})
        #sns.set(font_scale = 1)
        sns.set_style("ticks")
        a = plotter(*args, **kwargs)
        #sns.despine()
        
        return a
    return func_wrapper

def fontify():
    plt.rc('xtick', labelsize = 10)
    plt.rc('ytick', labelsize = 10)
    font = {'family' :'normal',
            'weight': 'bold',
            'size': 30}
    plt.rc('font', **font)
    plt.rc('legend', fontsize = 20)
    plt.rc('axes', labelsize = 30)
    plt.rc('xtick', labelsize = 20)
    plt.rc('ytick', labelsize = 20)
    
#decorator specifically for cosyne plots
def cosyne(plotter):
   def func_wrapper(*args, **kwargs):
        plt.rc('xtick', labelsize = 10)
        plt.rc('ytick', labelsize = 10)
        font = {'family' :'normal',
                'weight': 'bold',
                'size': 40}
        plt.rc('font', **font)
        #plt.rcParams.update({'font.size': 40})
        sns.set(font_scale = 10)
        sns.set_style("ticks")
        plotter(*args, **kwargs)
        sns.despine()
        
        return
   return func_wrapper
#%% Basic Plotting functions
#plot the correspondence between the output and the input records
@pretty
def in_out_compare(time, o_record, input_record, trial_range, dim_range, ax):
    downsample = 25
    o_truncated = np.ndarray.flatten(o_record[trial_range,dim_range,0:-1:downsample]).T
    i_truncated = np.ndarray.flatten(input_record[trial_range,dim_range,0:-1:downsample]).T
    time = time[0:-1:downsample]
    ax.plot(time, o_truncated, color = 'slateblue', zorder = 1, lw = 1);
    ax.plot(time, i_truncated, color = 'black', zorder = 2, lw = 3);

#plot the correspondence between the output and the input records with a std around the mean of the output
#for now, dim_range is required to contain only 1 number
@pretty
def in_out_std(time, o_record, input_record, trial_range, dim_range):
    ax = plt.gca()
    o_truncated = o_record[trial_range,dim_range,:].T
    i_truncated = input_record[trial_range,dim_range,:].T
    plt.plot(time, i_truncated, color = 'black');
    plt.xlabel('time')
    plt.ylabel('input/output')
    plt.title('Comparing network input and output')
    #add in standard deviation bars
    window_size = 200;
    o_mean = sf.sliding_mean(o_truncated, window_size)
    plt.plot(time, o_mean, color = 'blue')
    o_std = sf.sliding_std(o_truncated, window_size)
    upper_vtx = o_mean + o_std;
    lower_vtx = o_mean - o_std;
    verts = [ *zip(time, lower_vtx), *zip(time[::-1], upper_vtx[::-1]), ]
    poly = Polygon(verts, facecolor = '0.9', edgecolor = '0.5')
    ax.add_patch(poly)

 
#plot the objective through time to confirm convergence
@pretty
def objective_plot(obj_record, ax):
    sns.despine(ax = ax)
    obj_record_flat = np.ndarray.flatten(obj_record[:,:]);
    obj_record_flat_short = np.ndarray.flatten(obj_record[0:100,:])
    downsample = 10000
    ax.scatter(np.arange(0, len(obj_record_flat),downsample), obj_record_flat[::downsample], c = 'slateblue', alpha = 0.1)
    ax.set_xlabel('time (cumulative across trials)')
    ax.set_ylabel('objective')

#plot the performance pre-averaged over several test trials
def perform_plot(perform, perform_sem, ax):
    x = np.arange(0,len(perform))
    ax.plot(x, perform)
    #ax.plot(x, perform, alpha = 1, linewidth = 3);
    upper_vtx = perform + perform_sem;
    lower_vtx = perform - perform_sem;
    verts = [ *zip(x, lower_vtx), *zip(x[::-1], upper_vtx[::-1]), ]
    poly = Polygon(verts, alpha = 0.5)
    ax.add_patch(poly)
    
#plot the inputs vs. outputs in a scatterplot
@pretty
def scatter_compare(o_record, input_record, trial_range, dim_range, transient):
    fontsize = 140
    plt.rc('xtick', labelsize = fontsize)
    plt.rc('ytick', labelsize = fontsize)
    fig, ax = plt.subplots(figsize = (15,15))
    output = np.ndarray.flatten(o_record[trial_range, dim_range, transient:-1])
    input_var = np.ndarray.flatten(input_record[trial_range, dim_range, transient:-1])
    lr = LinearRegression()
    lr.fit(output[:, np.newaxis], input_var);
    x = np.arange(0, 1, 0.01);
    plt.scatter(output[0:-1:30], input_var[0:-1:30],  s = 300, c = 'green', alpha = 0.3)
    y_min = np.amin(input_record[trial_range, dim_range, transient:-1]);
    y_max = np.amax(input_record[trial_range, dim_range, transient:-1]);
    y_range = np.arange(y_min, y_max, 0.01);
    plt.plot(y_range, y_range, color = 'red', linestyle = 'dashed', linewidth = 15)
    plt.plot(x, lr.predict(x[:, np.newaxis]), color = 'sienna', linestyle = 'dashed', linewidth = 15)
    
@pretty
#plot the input theta along a circle, and overlay the output
def circle_compare(o_record, target_record, downsample, ax):
    sns.despine(ax = ax)
    o_reshape = np.reshape(np.transpose(o_record, (1, 0, 2)), (2, o_record.shape[0]*o_record.shape[2]));
    o_reshape = o_reshape[:, ::downsample]
    target_reshape = np.reshape(np.transpose(target_record, (1,0,2)), (2, target_record.shape[0]*target_record.shape[2]));
    target_pts = np.unique(target_reshape, axis = 1)
    target_pts = np.hstack((target_pts[:,0:6], target_pts[:,7::]))
    target_reshape = target_reshape[:, ::downsample]
    total_variance = np.zeros(len(target_pts[0,:]))
    for ii in np.arange(0, len(target_pts[0,:])):
        data_cloud = o_reshape[:, target_reshape[0,:] == target_pts[0,ii]];
        data_cov = np.cov(data_cloud);
        total_variance[ii] = np.linalg.det(data_cov)
        v,w = linalg.eigh(data_cov)
        u = w[0]/ linalg.norm(w[0])
        angle = np.arctan(u[1]/u[0])
        angle = 180 * angle/ np.pi
        data_mean = np.mean(data_cloud, axis = 1)
        ell = mpl.patches.Ellipse(np.mean(data_cloud, axis = 1), 2 * v[0] ** 0.5, 2 * v[1] ** 0.5, 180 + angle, edgecolor = 'black', linewidth = 2, zorder = 0)
        ax.add_artist(ell)
        plt.plot([target_pts[0,ii], data_mean[0]], [target_pts[1,ii], data_mean[1]], 'k')
        
    ax.scatter(target_pts[0,:], target_pts[1,:], marker = 'X', linewidths = 1.5, c = 'k', zorder = 10)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlim(-1.5, 1.5)

    #plot the line of most probable input
    x = np.array([0, 1.5])
    y = np.array([0, 0])
    ax.plot(x, y, 'k--', linewidth = 3)
    return total_variance

#@pretty
#compare the classifier boundary for the high probability and the low probability boundaries
def boundary_compare(r_record, target_record, decoder, theta_record, theta_range, ax1, ax2):
    #generate 2 plots, one for the high probability boundary, and one for the low probability boundary
    N = len(r_record[0,:,0]);
    #assume theta_value has 10 values
    r_reshape = np.reshape(np.transpose(r_record, (1, 0, 2)), (N, r_record.shape[0]*r_record.shape[2]));
    theta_reshape = np.reshape(np.transpose(theta_record, (1,0,2)), (1, theta_record.shape[0] * theta_record.shape[2]))
    theta_high_pos = theta_range[7]
    theta_high_neg = theta_range[5]
    theta_low_pos = theta_range[11]
    theta_low_neg = theta_range[1]
    idx_high_pos = np.where(theta_reshape == theta_high_pos)[1]
    idx_high_neg = np.where(theta_reshape == theta_high_neg)[1]
    idx_low_pos = np.where(theta_reshape == theta_low_pos)[1]
    idx_low_neg = np.where(theta_reshape == theta_low_neg)[1]
    
    o_high_pos = decoder @ r_reshape[:, idx_high_pos];
    o_high_neg = decoder @ r_reshape[:, idx_high_neg];
    o_low_pos = decoder @ r_reshape[:, idx_low_pos];
    o_low_neg = decoder @ r_reshape[:, idx_low_neg];
    sns.distplot(o_high_pos, hist = False, ax = ax1)
    sns.distplot(o_high_neg, hist = False, ax = ax1)
    ax1.set_xlim((-0.08, 0.08))
    ax2.set_xlim((-0.08, 0.08))
    sns.distplot(o_low_pos, hist = False, ax = ax2)
    sns.distplot(o_low_neg, hist = False, ax = ax2)
    d_prime_high = np.abs(np.mean(o_high_pos) - np.mean(o_high_neg))/np.sqrt(1/2 * (np.var(o_high_pos) + np.var(o_high_neg)))
    d_prime_low = np.abs(np.mean(o_low_pos) - np.mean(o_low_neg))/np.sqrt(1/2 * (np.var(o_low_pos) + np.var(o_low_neg)))
    return d_prime_high, d_prime_low

#perform linear discriminant analysis on data from the classifier
def classifier_lda(r_record, target, theta_record, ax):
    N = len(r_record[0,:,0])
    r_reshape = np.reshape(np.transpose(r_record, (1, 0, 2)), (N, r_record.shape[0]*r_record.shape[2]));
    target_reshape = np.reshape(np.transpose(target, (1,0,2)), (1, target.shape[0] * target.shape[2]));
    theta_reshape = np.reshape(np.transpose(theta_record, (1,0,2)), (1, theta_record.shape[0] * theta_record.shape[2]))
    classes = np.zeros(theta_reshape.shape)
    classes[np.where(theta_reshape > 0)] = 1
    classes = np.ndarray.flatten(classes)
    r_reshape = r_reshape.T
    lda = LinearDiscriminantAnalysis(n_components = 2)
    r_transformed = lda.fit(r_reshape, classes).transform(r_reshape)
    d_prime = np.abs(np.mean(r_transformed[classes == 0]) - np.mean(r_transformed[classes == 1]))/np.sqrt(1/2 * (np.var(r_transformed[classes == 1]) + np.var(r_transformed[classes == 0])))
    sns.distplot(r_transformed[classes == 0])
    sns.distplot(r_transformed[classes == 1])
    return d_prime

#plot the distribution of the 'up' trials compared to the 'down' trials throughout learning
def discriminability_plot(r_record, decoder, theta_record, ax):
    N = r_record.shape[1]
    r_reshape = np.reshape(np.transpose(r_record, (1, 0, 2)), (N, r_record.shape[0]*r_record.shape[2]));
    theta_reshape = np.reshape(np.transpose(theta_record, (1, 0, 2)), (1, theta_record.shape[0]*theta_record.shape[2]));
    idx_up = np.where(theta_reshape > 0)[1]
    idx_down = np.where(theta_reshape < 0)[1]
    r_up = decoder @ r_reshape[:, idx_up]
    r_down = decoder @ r_reshape[:, idx_down]
    sns.distplot(r_up, ax = ax, hist = False)
    sns.distplot(r_down, ax = ax, hist = False)
    ax.set_xlim((-0.08, 0.08))
    ax.set_ylim((0,30))
#%% Analysis plotting functions
#plot a polar representation of error
#sampling: number of samples to skip
#@pretty
        
def obj_polar(theta_record, theta_range, obj_record, trial_range):
    obj_record = np.abs(obj_record[:,0:-1:10]);
    ax = plt.subplot(111, projection = 'polar')
    mean_obj = np.zeros(theta_range.shape);
    obj_slice = obj_record[trial_range,:]
    for ii in range(0,len(theta_range)):
        idx = np.where(theta_range[ii] == theta_record[trial_range,0,:]);
        mean_obj[ii] = np.mean(obj_slice[idx])
    plt.rc('xtick', labelsize = 40)
    plt.rc('ytick', labelsize = 40)
    #colors = obj_record[trial_range,transient:-1:sampling]
    bars = ax.bar(theta_range, mean_obj, width = np.abs(theta_range[0] - theta_range[1]))
    xT=plt.xticks()[0]
    xL=['0',r'',r'$\frac{\pi}{2}$',r'',\
    r'$\pi$',r'',r'$\frac{3\pi}{2}$',r'']
    plt.xticks(xT, xL)
    #ax.set_title('polar plot of error')
    #ax.set_ylabel('MSE', rotation = 45, size = 11)
    for r, bar in zip(mean_obj, bars):
        bar.set_facecolor('k')
        bar.set_alpha(0.7)

#plot the mean performance on the objective function as a function of stimulus angle
#plotted onto rectilinear coordinates rather than circular     
@pretty
def obj_rect(theta_record, theta_range, obj_record, trial_range, ax, discrim_threshold = None, theta_compare = None, obj_compare = None, splines = True):
    sns.despine(ax = ax, left = False, right = True)
    obj_record = np.abs(obj_record[:,0:-1:10]);
    mean_obj = np.zeros(theta_range.shape);
    obj_slice = obj_record[trial_range,:]
    for ii in range(0,len(theta_range)):
        idx = np.where(theta_range[ii] == theta_record[trial_range,0,:]);
        mean_obj[ii] = np.mean(obj_slice[idx])
    font_size = 10
    plt.rc('xtick', labelsize = font_size)
    plt.rc('ytick', labelsize = font_size)
    
    x = theta_range
    y = mean_obj
    spl = UnivariateSpline(x,y, k=4, s = 0) #interpolate with quartic splines to make the figure prettier
    xs = np.linspace(np.amin(theta_range), np.amax(theta_range), 500)
    if (splines):
        ax.plot(xs, np.maximum(0, spl(xs)), 'k', alpha = 1, linewidth = 3);
    else:
        ax.plot(x,y, 'k', alpha = 1, linewidth = 3)
    #plt.title('Neuron Tuning curves')
    ax.set_ylabel('error', fontsize = font_size*1.5)
    ax.set_xlabel(r'$\theta$', fontsize = font_size * 1.5)
    xT=plt.xticks()[0]
    xL=[r'0',r'',r'$\frac{\pi}{4}$',r'',\
    r'$\frac{\pi}{2}$',r'',r'$\pi$',r'']
    #plt.xticks(xT, xL)
    if (not(discrim_threshold == None)):
        x = np.array([discrim_threshold, discrim_threshold])
        y = np.array([0, np.amax(mean_obj)])
        ax.plot(x, y, 'k--', linewidth = 8)
    if (not(theta_compare is None) and not(obj_compare is None)):
        obj_compare = np.abs(obj_compare[:,0:-1:10]);
        mean_obj_compare = np.zeros(theta_range.shape);
        obj_slice = obj_compare[trial_range,:]
        for ii in range(0,len(theta_range)):
            idx = np.where(theta_range[ii] == theta_compare[trial_range,0,:]);
            mean_obj_compare[ii] = np.mean(obj_slice[idx])
        
        y_compare = mean_obj_compare
        spl_compare = UnivariateSpline(x,y_compare, k=4, s = 0) #interpolate with quartic splines to make the figure prettier
        if (splines):
            ax.plot(xs, np.maximum(0, spl_compare(xs)), 'maroon', alpha = 1, linewidth = 3);
        else:
            ax.plot(x, y_compare, 'maroon', alpha = 1, linewidth = 3)

#plot the average noise correlations as a function of stimulus angle
@pretty
def avg_noise_correlations(theta_range, noise_correlations, ax):
    nc_mean = np.zeros((len(theta_range),));
    nc_std = np.zeros((len(theta_range),));
    idx = np.ones(noise_correlations.shape[1:2], dtype = 'int32') - np.eye(noise_correlations.shape[1], noise_correlations.shape[1], dtype = 'int32')
    for ii in range(0, len(nc_mean)):
        nc = noise_correlations[ii,:,:];
        nc = np.abs(nc[idx]);
        nc_mean[ii] = np.mean(nc);
        nc_std[ii] = np.std(nc);
    ax.errorbar(theta_range, nc_mean, yerr = nc_std, fmt = '.')
    ax.set_title('average noise correlations as a function of stimulus angle')
    ax.set_xlabel('theta')
    ax.set_ylabel('average noise correlation')

#simple plot of noise correlations between neurons, averaged across stimuli
def noise_correlations_plot(noise_correlations, color_range, ax):
    nc_mean = np.mean(noise_correlations, axis = 0) - np.eye(noise_correlations.shape[1], noise_correlations.shape[2])
    sns.heatmap(nc_mean, ax = ax, vmin = color_range[0], vmax = color_range[1])

#plot the relationship between signal correlations and mean noise correlations
@pretty
def signal_vs_noise_correlations(sc, nc, ax):
    sns.despine(ax = ax)
    N = nc.shape[1]
    nc_mean = np.mean(nc, axis = 0)
    nc_mean = np.ndarray.flatten(nc_mean - np.eye(N,N))
    sc = np.ndarray.flatten(sc - np.eye(N,N))
    ax.scatter(sc, nc_mean);
    ax.set_xlabel('signal correlation')
    ax.set_ylabel('noise correlation')
    
#plot tuning curves and a discrimination threshold (optionally)
@pretty
def tuning_curves_plotter(theta_range, tuning_curves, ax, discrim_threshold = None, obj_record = None, theta_record = None, classifier = True):
    sns.despine(ax = ax, right = False)
    font_size = 10
    trial_range = range(0,100);
    plt.rc('xtick', labelsize = font_size)
    plt.rc('ytick', labelsize = font_size)
    #optionally add in the error too
    if ((not(obj_record is None)) and not(theta_record is None)):
        obj_record = np.abs(obj_record[:,0:-1:10]);
        mean_obj = np.zeros(theta_range.shape);
        obj_slice = obj_record[trial_range,:]
        for ii in range(0,len(theta_range)):
            idx = np.where(theta_range[ii] == theta_record[trial_range,0,:]);
            mean_obj[ii] = np.mean(obj_slice[idx])
        
        x = theta_range
        y = mean_obj
        spl = UnivariateSpline(x,y, k=4, s = 0)
        xs = np.linspace(np.amin(theta_range), np.amax(theta_range), 500)
        if (classifier):
            offset = 0.2;
            scale = 1;
        else:
            offset = -0;
            scale = 0.3
        plt.plot(xs, scale * (np.maximum(0, spl(xs)) - offset), 'salmon', alpha = 1, linewidth = 12);

    if (classifier):
        color = 'slateblue'
    else:
        color = 'green'
    tuning_curves_sparse = tuning_curves[:,0:-1:2] #pick out only a few tuning curves
    #plot each tuning curve
    for ii in range(0, len(tuning_curves_sparse[0,:])):
        x = theta_range
        y = tuning_curves_sparse[:,ii]
        spl = UnivariateSpline(x,y, k=4, s = 0)
        xs = np.linspace(np.amin(theta_range), np.amax(theta_range), 500)
        ax.plot(xs, np.maximum(0, spl(xs)), color, alpha = np.random.uniform(low = 0.3, high = 0.8), linewidth = 3);
    ax.set_ylabel('mean firing rate', fontsize = font_size*1.5)
    ax.set_xlabel(r'$\theta$', fontsize = font_size * 1.5)
    if (not(discrim_threshold == None)):
        x = np.array([discrim_threshold, discrim_threshold])
        y = np.array([0, np.amax(tuning_curves)])
        ax.plot(x, y, 'k--', linewidth = 3)
        
#plot the distribution of the maximum response stimulus for the neurons in the network
def tuning_curve_dist(theta_range, tuning_curves, ax, ax_2, classify = False):
    if (not(classify)):
        threshold = 0
        tuning_curves = tuning_curves[:, np.where(np.sum(tuning_curves >= threshold, axis = 0) > 0)[0]];
        max_theta = theta_range[np.argmax(tuning_curves, axis = 0)]
        max_theta_padded = np.concatenate((max_theta, max_theta - 2* np.pi, max_theta + 2* np.pi))
        sns.distplot(max_theta, bins = 12, ax = ax, kde = False, norm_hist = True)
        kde = KernelDensity(kernel = 'gaussian', bandwidth = 0.4).fit(max_theta_padded.reshape([len(max_theta_padded),1]))
        theta_range_2 = np.arange(np.min(theta_range), np.max(theta_range), 0.01)
        log_dens = kde.score_samples(theta_range_2.reshape([len(theta_range_2),1]))
        ax.plot(theta_range_2, 3* np.exp(log_dens))
        ax.set_ylim((0,0.4))
        ax.set_xlim((np.min(theta_range), np.max(theta_range)))

        ax_2.plot(theta_range, np.mean(tuning_curves, axis = 1))
        tc_mean = np.mean(tuning_curves, axis = 1)
        tc_sem = np.sqrt(np.var(tuning_curves, axis = 1)/tuning_curves.shape[1])
        upper_vtx = tc_mean + tc_sem;
        lower_vtx = tc_mean - tc_sem;
        verts = [ *zip(theta_range, lower_vtx), *zip(theta_range[::-1], upper_vtx[::-1]), ]
        poly = Polygon(verts, alpha = 0.5)
        ax_2.add_patch(poly)

    else:
        idx = np.argmax(tuning_curves, axis = 0)
        idx_pos = idx[np.where(theta_range[idx] > 0)]
        idx_neg = idx[np.where(theta_range[idx] <= 0)]
        max_theta_pos = theta_range[idx_pos]
        max_theta_neg = theta_range[idx_neg]
        sns.distplot(max_theta_pos, bins = 6, ax = ax, kde = True, norm_hist = True)
        sns.distplot(max_theta_neg, bins = 6, ax = ax, kde = True, norm_hist = True)
        ax.set_ylim((0,0.4))

        ax_2.plot(theta_range, np.mean(tuning_curves[:,idx_neg], axis = 1))
        ax_2.plot(theta_range, np.mean(tuning_curves[:,idx_pos], axis = 1))
        

        
    

#compare mean tuning curves
def tuning_curve_compare(theta_range, tuning_curves, tc_compare, ax, discrim_threshold = None):
    sns.despine(ax = ax, right = False)
    font_size = 10
    plt.rc('xtick', labelsize = font_size)
    plt.rc('ytick', labelsize = font_size)

    color = 'slateblue'
    color_compare = 'salmon'
    tuning_curves_mean = np.mean(tuning_curves, axis = 1) #average the tuning curves
    tuning_curves_sem = np.std(tuning_curves, axis = 1)/np.sqrt(tuning_curves.shape[1])
    tc_compare_mean = np.mean(tc_compare, axis = 1)
    tc_compare_sem = np.std(tc_compare, axis = 1)/np.sqrt(tc_compare.shape[1])

    #plot the mean tuning curve
    x = theta_range
    ax.plot(x, tuning_curves_mean, color, alpha = 1, linewidth = 3);
    # add in error bars
    upper_vtx = tuning_curves_mean + tuning_curves_sem;
    lower_vtx = tuning_curves_mean - tuning_curves_sem;
    verts = [ *zip(x, lower_vtx), *zip(x[::-1], upper_vtx[::-1]), ]
    poly = Polygon(verts, color = color, alpha = 0.5)
    ax.add_patch(poly)
    #plot the mean tuning curve comparison
    x = theta_range
    ax.plot(x, tc_compare_mean, color_compare, alpha = 1, linewidth = 3);
    upper_vtx = tc_compare_mean + tc_compare_sem;
    lower_vtx = tc_compare_mean - tc_compare_sem;
    verts = [ *zip(x, lower_vtx), *zip(x[::-1], upper_vtx[::-1]), ]
    poly = Polygon(verts, color = color_compare, alpha = 0.5)
    ax.add_patch(poly)
    
    #plt.title('Neuron Tuning curves')
    ax.set_ylabel('mean firing rate', fontsize = font_size*1.5)
    ax.set_xlabel(r'$\theta$', fontsize = font_size * 1.5)
    if (not(discrim_threshold == None)):
        x = np.array([discrim_threshold, discrim_threshold])
        y = np.array([0, np.amax(tuning_curves)])
        ax.plot(x, y, 'k--', linewidth = 3)
    
# plot the coactivity of two target neurons across a range of stimulus values from a test set
def coactivity_plotter(theta_record, theta_value, r_record, neuron_ids, decoder):
    N = len(r_record[0,:,0]);
    #assume theta_value has 10 values
    fig, axes = plt.subplots(2,5, figsize = (40, 20));
    theta_idx = -1
    r_reshape = np.reshape(np.transpose(r_record, (1, 0, 2)), (N, r_record.shape[0]*r_record.shape[2]));
    theta_reshape = np.reshape(np.transpose(theta_record, (1,0,2)), (1, theta_record.shape[0]*theta_record.shape[2]));
    for ii in range(0,2):
        for jj in range(0, 5):
            theta_idx = theta_idx + 1
            idx = np.where(theta_value[theta_idx] == theta_reshape)[1]
            idx = idx[np.where(np.mod(idx, 2000) > 40)[0]]
            n_1 = r_reshape[neuron_ids[0],idx]
            n_2 = r_reshape[neuron_ids[1],idx]
            axes[ii,jj].scatter(n_1, n_2);
            axes[ii,jj].set_xlabel('neuron ' + str(neuron_ids[0]))
            axes[ii,jj].set_ylabel('neuron ' + str(neuron_ids[1]))
            axes[ii,jj].set_title('theta = ' + str(theta_value[theta_idx]))
@pretty
#plot the covariance of neurons (sorted by maximum tuning), stratified by the stimulus angle
def mass_cov_plotter(covariance_sorted, theta_range):
    plt.rc('image', cmap = 'gnuplot')
    fig, axes = plt.subplots(2,5, figsize = (40, 20));
    counter = -1
    N = len(covariance_sorted[0,:,0])
    for ii in range(0,2):
        for jj in range(0,5):
            counter = counter + 1
            diagonal_terms = np.eye(N,N)*covariance_sorted
            im = axes[ii,jj].imshow(covariance_sorted[counter,:,:] - diagonal_terms[counter,:,:],
                vmin = -0.0002, vmax = 0.0002)
            axes[ii,jj].set_title('theta = ' + str(theta_range[counter]))
            fig.colorbar(im, ax = axes[ii,jj], orientation = 'horizontal')

@pretty
#just a plot of a covariance matrix
def covariance_plot(covariance):
    fig, ax = plt.subplots(figsize = (20,10))
    plt.imshow(covariance, origin = 'lower')
    ax.set_xlabel('neuron number')
    ax.set_ylabel('neuron number')
    plt.colorbar(ax = ax)

#given a neurons who have been sorted by the angle they are most responsive to, plot the covariance structure    
def sorted_stimulus_variance_plot(CV, theta_range):
    fig, ax = plt.subplots(figsize = (7,10))
    plt.imshow(CV, origin = 'lower')
    ax.set_xlabel('theta')
    ax.set_ylabel('neuron number')
    plt.colorbar(ax = ax)

#plot the systematic bias of the network output compared to the input, as a function of stimulus angle

def bias_plot(bias, theta_range, ax, linear = False):

    if(linear):
        bias_sem = np.sqrt(np.var(np.sum(bias**2, axis = 0), axis = 1)/bias.shape[2])
        bias = np.mean(np.sum(bias **2, axis = 0), axis = 1)
        #bias = np.sum(bias **2, axis = 0)
    else:
        bias = bias.reshape((bias.size,1))
    ax.plot(theta_range, bias)
    ax.set_xlabel('angle')
    ax.set_ylabel('output bias')
    upper_vtx = bias + bias_sem;
    lower_vtx = bias - bias_sem;
    verts = [ *zip(theta_range, lower_vtx), *zip(theta_range[::-1], upper_vtx[::-1]), ]
    poly = Polygon(verts, facecolor = '0.9', edgecolor = '0.5')
    ax.add_patch(poly)

def error_plot(error, theta_range, ax, linear = False):

    if(linear):
        error_sem = stats.sem(error, axis = 0)
        error = np.mean(error, axis = 0)
    ax.plot(theta_range, error)
    ax.set_xlabel('angle')
    ax.set_ylabel('error')
    upper_vtx = error + error_sem;
    lower_vtx = error - error_sem;
    verts = [ *zip(theta_range, lower_vtx), *zip(theta_range[::-1], upper_vtx[::-1]), ]
    poly = Polygon(verts, facecolor = '0.9', edgecolor = '0.5')
    ax.add_patch(poly)
    
@pretty
#plot on a circle the inputs vs. the average biased output
def radial_bias_plot(avg_error, theta_range, ax):
    inputs = np.array([np.cos(theta_range), np.sin(theta_range)]).T
    for ii in np.arange(0, len(theta_range)):
        ax.plot(np.array([inputs[ii,0], inputs[ii,0]+avg_error[ii,0]]), np.array([inputs[ii,1], inputs[ii,1] + avg_error[ii,1]]))

#plot the 'volume_fraction' variable as a function of stimulus angle. See pl_support_functions.py for details about the volume_fraction
@pretty
def volume_fraction_plot(theta_range, volume_fraction, volume_fraction_original, ax):
    sns.despine(ax = ax)
    ax.plot(theta_range, volume_fraction)
    ax.plot(theta_range, volume_fraction_original)
    ax.set_xlabel('angle')
    ax.set_ylabel('volume fraction')
    ax.legend(('after learning', 'before learning'))
    
def volume_fraction_total_plot(theta_range, volume_fraction_total, volume_fraction_total_original, ax):
    volume_sem = np.sqrt(np.var(volume_fraction_total, axis = 0)/volume_fraction_total.shape[0])
    volume_mean = np.mean(volume_fraction_total, axis = 0)
    
    volume_original_sem = np.sqrt(np.var(volume_fraction_total_original, axis = 0)/volume_fraction_total_original.shape[0])
    volume_original_mean = np.mean(volume_fraction_total_original, axis = 0)
    
    ax.plot(theta_range, volume_mean)
    ax.plot(theta_range, volume_original_mean)
    ax.set_xlabel('angle')
    ax.set_ylabel('volume_fraction')
    upper_vtx = volume_mean + volume_sem;
    lower_vtx = volume_mean - volume_sem;
    upper_vtx_original = volume_original_mean + volume_original_sem;
    lower_vtx_original = volume_original_mean - volume_original_sem;
    verts = [ *zip(theta_range, lower_vtx), *zip(theta_range[::-1], upper_vtx[::-1]), ]
    poly = Polygon(verts, facecolor = '0.9', edgecolor = '0.5')
    ax.add_patch(poly)
    
    verts = [ *zip(theta_range, lower_vtx_original), *zip(theta_range[::-1], upper_vtx_original[::-1]), ]
    poly = Polygon(verts, facecolor = '0.9', edgecolor = '0.5')
    ax.add_patch(poly)
#plot the variance of the decoder compared to the input as a function of stimulus angle
def variance_plot(variance, theta_range, ax, linear = False):
    
    if(linear):
        variance_sem = np.sqrt(np.var(np.sum(variance, axis = 0), axis = 1)/variance.shape[2])
        variance = np.mean(np.sum(variance, axis = 0), axis = 1)
        #variance = np.sum(variance, axis = 0)

    else:
        variance = variance.reshape((variance.size,1))
    ax.plot(theta_range, variance)
    ax.set_xlabel('angle')
    ax.set_ylabel('output_variance')
    upper_vtx = variance + variance_sem;
    lower_vtx = variance - variance_sem;
    verts = [ *zip(theta_range, lower_vtx), *zip(theta_range[::-1], upper_vtx[::-1]), ]
    poly = Polygon(verts, facecolor = '0.9', edgecolor = '0.5')
    ax.add_patch(poly)
#plot the relative contributions of the recurrent and input currents as a function of stimulus angle
@pretty
def current_plot(r_current, i_current, bias_current, theta_range, ax):
    sns.despine(ax = ax, right = False, top = True)
    r_current = np.sum(r_current, axis = 0)
    i_current = np.sum(i_current, axis = 0)
    bias_current = np.sum(bias_current, axis = 0)
    
    ax.plot(theta_range, r_current, label = 'recurrent current')
    ax_2 = ax.twinx()
    sns.despine(ax = ax_2, right = False, top = True)
    ax_2.plot(theta_range, i_current, color = 'g', label = 'input current')
    
#plot the ratio of currents in two different situations (usually low noise vs. high noise)
@pretty
def current_ratio_plot(r_high, r_low, i_high, i_low, ax):
    sns.despine()
    low_ratio = np.mean(np.abs(r_low))/np.mean(np.abs(i_low))
    high_ratio = np.mean(np.abs(r_high))/np.mean(np.abs(i_high))
    ax.bar([0,1], [low_ratio, high_ratio])
    
#plot the dimensionality of the network activity
@pretty
def pca_plot(pca, ax, cumulative = False):
    sns.despine(ax = ax)
    if (cumulative):
        cumulative_var = np.cumsum(pca.explained_variance_ratio_[0:6])
        ax.plot(cumulative_var, linewidth = 3, solid_capstyle = 'round')
    else:
        var = pca.explained_variance_ratio_[0:6]
        ax.plot(var, linewidth = 3, solid_capstyle = 'round')
    ax.set_xlabel('principal component')
    ax.set_ylabel('percentage explained variance')

#plot mean error as a function of the test sigma
@pretty
def mean_error_noise_plot(sigmas, mean_error, ax, log = False):
    sns.despine(ax = ax)
    ax.plot(sigmas, mean_error)
    if (log):
        ax.set_xscale('log')
    ax.set_xlabel('sigma')
    ax.set_ylabel('mean error')
    
#plot the input distribution
@pretty
def input_distribution_plot(theta_record, ax):
    sns.despine(ax = ax)
    ax.hist(np.ndarray.flatten(theta_record), normed = True)

#%% Construct the elements of the schematic
@pretty
#a prettified plot of any arbitrary FI curve
def fi_plot(params, fi_fn):
    fig, ax = plt.subplots(figsize = (4,4))
    x = np.arange(-2, 2, 0.01);
    y = fi_fn(params, x);
    lw = 15
    ax.plot(x, y,'k', linewidth = lw)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.spines['left'].set_position('center')
    ax.spines['left'].set_linewidth(lw)
    ax.spines['bottom'].set_linewidth(lw)

#plot how distributions change in different experiments
@pretty
def two_distribution_schematic(kappa, mu, kappa_new, mu_new, ax, discrim_threshold = None):
    sns.despine()
    x = np.arange(-np.pi, np.pi, 0.01)
    y_1 = vonmises.pdf(x, kappa, loc = mu)
    ax.plot(x, y_1, linewidth = 3)
    ax.plot(x, vonmises.pdf(x, kappa_new, loc = mu_new), linewidth = 3)
    if (not(discrim_threshold == None)):
        x = np.array([discrim_threshold, discrim_threshold])
        y = np.array([0, np.amax(y_1)])
        ax.plot(x, y, 'k--', linewidth = 3)

def tuning_curve_fn(x, center):
    response = np.exp(-(x - center)**2)
    response = response/ np.amax(response)
    return response

#generate a schematic for the 3-factor synaptic plasticity rule
#shows how the plasticity rule performs gradient ascent by using local covariance information
@pretty
def plasticity_schematic():
    plt.rc('xtick', labelsize = 60)
    plt.rc('ytick', labelsize = 60)
    fig, ax = plt.subplots(figsize = (8.4,7.2))
    lw = 20;
    rr = np.arange(-1, 1, 0.01)
    error = -rr**4 + 1
    plt.plot(rr, error, 'k', linewidth = lw, zorder = 1)
    points = np.array([-0.75, 0, 0.75]);
    cov = -4 * points**3;
    size_a = 1.5 * np.array([800, 1000, 800]);
    for ii in range(0, len(points)):
        #generate data clouds at 3 points along the objective function (a quadratic concave-down function)
        data = np.random.multivariate_normal(np.array([points[ii],-points[ii]**4 + 1]), cov = 0.003 * np.array([[3, cov[ii]], [cov[ii], 3]]), size = 50)
        data[:,1] = np.maximum(0,data[:,1])
        idx_pos = np.where(data[:,0] - points[ii] > 0)[0];
        idx_neg = np.where(data[:,0] - points[ii] < 0)[0];
        data_pos = data[idx_pos,:]
        data_neg = data[idx_neg,:]
        mod_size = np.array([0.3, 1])
        if ii == 0:
            mod_neg = mod_size[1]
            mod_pos = mod_size[0]
        elif ii == 1:
            mod_neg = mod_size[1]
            mod_pos = mod_size[1]
        else:
            mod_neg = mod_size[0]
            mod_pos = mod_size[1]
        #plot the size of the data points so that larger points correlate to larger rewards
        #color the data points based on whether LTD or LTP is being performed
        plt.scatter(data_pos[:,0], data_pos[:,1], zorder = 2, s = size_a[ii]/mod_pos * np.abs(data_pos[:,0] - points[ii]) * ((data_pos[:,1] + np.abs(points[ii]))**4 + 1), color = 'royalblue', alpha = .9)
        plt.scatter(data_neg[:,0], data_neg[:,1], zorder = 2, s = size_a[ii]/mod_neg * np.abs(data_neg[:,0] - points[ii]) * ((data_neg[:,1] + np.abs(points[ii]))**4 + 1), color = 'indianred', alpha = .9)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

#like plasticity_schematic(), but only generates a data cloud. x_mean specifies where locally on the objective function you are
@pretty
def plasticity_schematic_2(x_mean):
    lw = 10
    plt.rc('xtick', labelsize = 60)
    plt.rc('ytick', labelsize = 60)
    fig, ax = plt.subplots(figsize = (5,3))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.spines['left'].set_position('zero')
    ax.set_axisbelow(True)
    cov = -2 * x_mean
    data = np.random.multivariate_normal(np.array([0,0]), cov = 0.03 * np.array([[2.5, cov], [cov, 2.5]]), size = 100)
    idx_pos = np.where(data[:,0] > 0)[0];
    idx_neg = np.where(data[:,0] < 0)[0];
    data_pos = data[idx_pos,:]
    data_neg = data[idx_neg,:]
    plt.scatter(data_pos[:,0], data_pos[:,1], zorder = 2, s = 1000* np.abs(data_pos[:,0]) * ((data_pos[:,1] + np.abs(x_mean)+ 1)**4), color = 'royalblue', alpha = .9)
    plt.scatter(data_neg[:,0], data_neg[:,1], zorder = 2, s = 1000* np.abs(data_neg[:,0]) * ((data_neg[:,1] + np.abs(x_mean) + 1)**4), color = 'indianred', alpha = .9)


#circle_compare(o_test_2, target_test_2, downsample = 1000)
#fig, axes = plt.subplots(1)
#objective_plot(np.concatenate((obj_record, obj_record_2), axis = 0), axes)
#scatter_compare(o_record, target_record, range(199,200), range(3,6), 100)
#tuning_curves_plotter(np.unique(theta_test), tc, discrim_threshold = np.pi/4, obj_record = obj_test, theta_record = theta_test, classifier = True)  
#time_vec = np.arange(0, params['T'], params['dt']*params['record_res'])
#in_out_compare(time_vec, o_record, target_record, range(200,201), range(0,1))
#obj_polar(theta_test, np.unique(theta_test), obj_test, range(0,params['test_num']))
#sorted_stimulus_variance_plot(ssv, np.unique(theta_test))
#nc_sorted = sf.sorted_noise_correlations(nc, tc, max_min = 'min')
#cv_sorted = sf.sorted_covariance(covariance(r_test), tc, max_min = 'min')
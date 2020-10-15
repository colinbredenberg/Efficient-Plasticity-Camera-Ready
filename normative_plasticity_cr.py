# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 09:53:57 2018
Code base for studying stochastic RNNs with local plasticity rules
Defines a saveable/trainable network class
@author: Colin
"""

import numpy as np
import time
import pl_exp_params_cr as exp_params
import copy
from scipy import stats
from sklearn.decomposition import PCA


#%% Main Class definition
class PlasticNet():
    #initialize a recurrent neural network with a plastic architecture
    #params: the free parameters for the network (noise level, time constant, etc)
    #plasticity_rule: a function dictating how the network updates its recurrent weights
    #input_rule: a function dictating how the network updates its input weights
    #bias_rule: a function dicating how the network updates its biases
    #param_fns: a dictionary of functions parameterizing the network
    #includes plasticity rules for the input weights, recurrent weights and biases
    #includes a definition of the vector nonlinearity and inverse nonlinearity (need not be the same for each neuron)
    def __init__(self, params, param_fns, input_params, decoder, weight_init, bias_init, i_weight_init):
        self.weights = weight_init; #initialize the recurrent weights
        self.i_weights = i_weight_init; #initialize the input weights
        self.biases = bias_init
        self.decoder = decoder; #initialize the decoder
        self.params = params; #initialize free network parameters. Dictionary containing basic parameters.
        self.param_fns = param_fns; #initialize the network functions. Dictionary of functions.
        self.input_params = input_params #parameters necessary for train_run to generate inputs
        
        self.biases_old = None
        self.weights_old = None
        self.i_weights_old = None
        self.decoder_old = None
        
        #store PCA data from the test set
        self.pcas = [] #principal components for network activity
        self.pca_trajectories = [] #projected trajectories of population activity
        self.decoders_past = []
    
    @classmethod #a new constructor for copying the existing data from an object
    #the object must contain: weights, biases, decoder, params, param_fns, and input_params
    #this constructor is valuable b/c it allows us to update the methods for an old saved PlasticNet by copying to a new object with all of the same data
    def create_copy(cls, obj, params, param_fns, input_params):
        return cls(params, param_fns, input_params, obj.decoder, obj.weights, obj.biases, obj.i_weights)
    
    #train_run: run the network and produce outputs. If train = True (default) the network will update its weights as well
    #T = total time per trial (dt is a parameter)
    #inputs: num_trials rows x input_dim x T/dt columns containing the input stimulus at every time step
    #r_0: the initial firing rates
    #by default, the input is NOT generated in real time. Set rt_input = True to generate inputs via a stochastic process
    #Note: it might be wise to add functionality so that the inputs can also be a function, so that the inputs are generated in real time
    def train_run(self, T, r_0, train = True, train_2 = False, tuning= False, perform = False, rt_input = False, decoder_fixed = False):
        print('input_noise:')
        print(exp_params.input_noise)
        beta_1 = 0.9
        beta_2 = 0.999
        epsilon = 10e-5
        d_momentum = np.zeros(self.decoder.shape)
        d_var = np.zeros(self.decoder.shape)
        #determine how many trials there should be
        if(train):
            if(train_2):
                if(perform):
                    session_num = self.params['perform_period']
                else:
                    session_num = self.params['trial_num_2']
            else:
                session_num = self.params['trial_num']         
        elif(perform):
            session_num = self.params['perform_num']
        else:
            session_num = self.params['test_num']
        #initialize variables for recording data
        if (not(rt_input)):
            inputs, theta_i = self.param_fns['input_generator'](self.input_params, fixed_samples = tuning, test = not(train)); #if inputs are generated in advance, generate them
            target = np.zeros((session_num, self.params['target_dim'], int(self.params['T']/self.params['dt'])))
        else:
            #if inputs are going to be made from a stochastic process, do that.
            inputs = np.zeros((session_num,self.params['input_dim'], int(self.params['T']/self.params['dt'])))
            theta_i = np.zeros((session_num,self.params['sample_dim'], int(self.params['T']/self.params['dt'])))
            target = np.zeros((session_num, self.params['target_dim'], int(self.params['T']/self.params['dt'])))
        obj_record = np.zeros((inputs.shape[0], inputs.shape[2])); #record of the smoothed objective function through time
        param_l2_record = np.zeros((inputs.shape[0], inputs.shape[2]));
        r_record = np.zeros((int(session_num/self.params['record_period']),self.params['N'],int(self.params['T']/self.params['dt']/self.params['record_res'])));
        o_record = np.zeros((int(session_num/self.params['record_period']), self.decoder.shape[0], int(self.params['T']/self.params['dt']/self.params['record_res'])));
        i_record = np.zeros((int(session_num/self.params['record_period']), self.params['input_dim'], int(self.params['T']/self.params['dt']/self.params['record_res'])));
        th_record = np.zeros((int(session_num/self.params['record_period']), self.params['sample_dim'], int(self.params['T']/self.params['dt']/self.params['record_res'])));
        target_record = np.zeros((int(session_num/self.params['record_period']), self.params['target_dim'], int(self.params['T']/self.params['dt']/self.params['record_res'])));
        for ii in np.arange(0, inputs.shape[0]): #loop over all trials
            #initialize all variables for a single trial
            r = np.zeros((self.params['N'],inputs.shape[2]));
            delta_w_short = np.zeros((self.params['N'], self.params['N'], inputs.shape[2]));
            o = np.zeros((self.decoder.shape[0],inputs.shape[2]));
            average_correlation = np.zeros((self.params['N'],self.params['N'])); # sliding average of correlation
            average_correlation_bias = np.zeros((self.params['N'],)); #sliding average of correlation between bias and firing rate
            average_correlation_input = np.zeros((self.params['N'], self.params['input_dim'])); #sliding average of the correlation between input and neuron FR
            average_r = np.zeros((self.params['N'],)); #record of the average firing rates throughout time
            random_vars = np.random.normal(0, np.sqrt(self.params['dt']), size = (self.params['N'],int(T/self.params['dt'])));
            if (exp_params.input_noise):
                input_random_vars = np.random.normal(0, self.input_params['input_noise_sigma'], size = (self.params['input_dim'],int(T/self.params['dt'])))
            if (rt_input):
                if (exp_params.objective == 'bci'):
                    i_0 = np.zeros(self.input_params['sample_dim'],)
                else:
                    i_0, theta_0 = self.param_fns['input_sampler'](self.input_params, (self.input_params['sample_dim'],));
                    random_vars_i = np.random.normal(0, np.sqrt(self.params['dt']), size = (self.input_params['sample_dim'],int(T/self.params['dt'])));
                    inputs[ii,:,0] = np.ndarray.flatten(i_0);
                    theta_i[ii,:,0] = np.ndarray.flatten(theta_0);
            r[:,0] = r_0; #set initial condition
            if (ii == 0):
                obj_record[ii,0] = 0;
                param_l2_record[ii,0] = 0;
            else:
                obj_record[ii,0] = obj_record[ii-1,-1]; #establish continuity across trials for the MSE measure
                param_l2_record[ii,0] = param_l2_record[ii-1, -1];
            if (np.mod(ii, self.params['record_period'])==0):
                w_init = self.weights;
                i_w_init = self.i_weights;
                b_init = self.biases;
                d_init = self.decoder;
#%% BEGIN CORE CODE - the basic code for the simulation itself
            for jj in np.arange(1, int(T/self.params['dt']), 1):
                #numerically solve the recurrent network stochastic differential equation (SDE)
                if (rt_input):
                    #rt_input = True means that the input is generated stochastically in real time along with the firing rates.
                    if(exp_params.objective == 'bci'):
                        #feed into the network the difference in activation between the two
                        inputs[ii,:,jj] = self.decoder @ r[:,jj-1]
                    else:
                        inputs[ii,:,jj], theta_i[ii,:,jj] = self.param_fns['input_f'](params, input_params, inputs[ii,:,jj-1], theta_i[ii,:,jj-1], random_vars_i[:,jj])

                r_prev = r[:,jj-1];
                nl_term = -1*self.param_fns['inv_nl'](self.params, r_prev)/self.params['R']; #the nonlinearity term for dr_dt
                recurrent_term = np.dot(self.weights, r_prev) #the recurrent connectivity term for dr_dt
                if (exp_params.input_noise):
                    input_term = np.dot(self.i_weights, inputs[ii,:,jj-1] + input_random_vars[:,jj]); #the input connectivity term for dr_dt
                else:
                    input_term = np.dot(self.i_weights, inputs[ii,:,jj-1]); #the input connectivity term for dr_dt
                dr_dt =  nl_term + recurrent_term + input_term + self.biases; #the derivative of the firing rate equations
                noise_term = self.params['sigma'] * random_vars[:,jj]; #turns into an SDE
                r[:,jj] = r_prev + dr_dt * self.params['dt'] + noise_term; #add noise to the differential
                r_biased = np.concatenate((r[:,jj], -1 * np.ones((1,))), axis = 0)
                r_biased = r[:,jj];
                o[:,jj] = self.param_fns['decoder_fn'](self.decoder, r_biased); #store the output of the network
                if (exp_params.objective == 'classifier'):
                    theta_target = theta_i[ii,:,jj]
                    target[ii,:,jj] = self.param_fns['target_generator'](self.params, theta_target);
                elif(exp_params.objective == 'mixed'):
                    theta_target = theta_i[ii,:,jj]
                    theta_2d = theta_i[ii,:,jj]
                    theta_2d = np.ndarray.flatten(np.array([np.cos(theta_2d), np.sin(theta_2d)]))
                    theta_total = np.append(theta_2d, theta_target) #bundle up the necessary input information to feed into the target generator function
                    target[ii,:,jj] = self.param_fns['target_generator'](self.params, theta_total)
                elif(exp_params.objective == 'bci'):
                    target[ii,:,jj] = self.param_fns['target_generator'](self.params)
                else:
                    theta_2d = theta_i[ii,:,jj]
                    theta_2d = np.ndarray.flatten(np.array([np.cos(theta_2d), np.sin(theta_2d)]))
                    target[ii,:,jj] = self.param_fns['target_generator'](self.params, theta_2d);
                #include the l2 regularizer on parameters in the weight update
                param_l2_record[ii,jj] = self.params['weight_l2'] * np.sum(np.ravel(self.weights**2)) + self.params['bias_weight_l2'] * np.sum(np.ravel(self.biases**2)) + self.params['i_weight_l2'] * np.sum(np.ravel(self.i_weights**2))    
                if (exp_params.objective == 'mixed'):
                    obj_record[ii,jj] = (self.param_fns['objective'](self.params, self.decoder, r[:,jj],target[ii,:,jj], classify_only = True) - param_l2_record[ii,jj]) #option: don't smooth the objective function

                else:
                    obj_record[ii,jj] = (self.param_fns['objective'](self.params, self.decoder, r[:,jj],target[ii,:,jj]))# - param_l2_record[ii,jj]) #option: don't smooth the objective function
                # prevent the activities from producing a NaN on the next iteration when passed through the inverse nonlinearity
                # Note: this NaN is caused by numerical imprecision, not by a failure of the model.
                r[np.where(r[:,jj] <= self.params['low_r_threshold']),jj] = self.params['low_r_threshold'];
                r[np.where(r[:,jj] >= self.params['high_r_threshold']),jj] = self.params['high_r_threshold'];
                correlation = np.outer(r[:,jj], r[:,jj]);
                correlation_bias = r[:,jj];
                correlation_input = np.outer(r[:,jj], inputs[ii,:,jj]);
                average_correlation = (1-self.params['theta']) * average_correlation + self.params['theta']*correlation;
                average_correlation_bias = (1-self.params['theta']) * average_correlation_bias + self.params['theta']*correlation_bias;
                average_correlation_input = (1-self.params['theta']) * average_correlation_input + self.params['theta'] * correlation_input;
                average_r = (1-self.params['theta']) * average_r + self.params['theta'] * r[:,jj];
                #ac_record[jj,:] = average_correlation_bias
                #perform the parameter updates
                #annealing puts the weight updates on a learning schedule that decreases the learning rate steadily through time.
                if ((jj*self.params['dt'] > self.params['transient']) and train): #give an initial transient for the average correlation to be accurate
                    if (exp_params.objective == 'mixed'):
                        anneal_const = 10;
                    else:
                        anneal_const = 100
                    if (not(exp_params.decoder_only)):
                        
                        weight_gradient = self.param_fns['plasticity_rule'](r[:,jj], inputs[ii,:,jj], target[ii,:,jj], average_correlation, self.weights, self.decoder, self.params);
                        weight_update = weight_gradient
                        delta_w_short[:,:,jj] = weight_update
                        if (exp_params.annealed):
                            self.weights = self.weights + self.params['dt'] * (anneal_const*np.exp(-1 *ii/150) + 1) * self.params['learning_rate'] * weight_update;
                        else:
                            self.weights = self.weights + self.params['dt'] * anneal_const*self.params['learning_rate'] * weight_update;
                        
                        bias_gradient = self.param_fns['bias_rule'](r[:,jj], inputs[ii,:,jj], target[ii,:,jj], average_correlation_bias, self.biases, self.biases, self.decoder, self.params);
                        bias_update = bias_gradient
                        if (exp_params.annealed):
                            self.biases = self.biases + self.params['dt'] * (anneal_const*np.exp(-1 *ii/150) + 1) * self.params['learning_rate_bias']*bias_update;
                        else:
                            self.biases = self.biases + self.params['dt'] * anneal_const*self.params['learning_rate_bias']*bias_update;
                        
                        i_weight_gradient = self.param_fns['input_rule'](r[:,jj], inputs[ii,:,jj], target[ii,:,jj], average_correlation_input, self.i_weights, self.decoder, self.params);
                        i_weight_update = i_weight_gradient
                        if (exp_params.annealed):
                            self.i_weights = self.i_weights + self.params['dt'] * (anneal_const*np.exp(-1 *ii/150) + 1) * self.params['learning_rate_input'] * i_weight_update;
                        else:
                            self.i_weights = self.i_weights + self.params['dt'] * anneal_const*self.params['learning_rate_input'] * i_weight_update;
                    if (exp_params.objective == 'bci'):
                        decoder_gradient = 0;
                    else:
                        decoder_gradient = self.param_fns['decoder_rule'](self.decoder, r[:,jj], inputs[ii,:,jj], target[ii,:,jj], weight_l2 = self.params['weight_l2'])
                    d_momentum = beta_1 * d_momentum + (1-beta_1) * decoder_gradient
                    d_var = beta_2 * d_var + (1-beta_2) * decoder_gradient**2
                    decoder_update = d_momentum / (epsilon + np.sqrt(d_var))
                    decoder_update = decoder_gradient
                    if (not(decoder_fixed)):
                        if (exp_params.annealed):
                            if (exp_params.objective == 'linear'):
                                self.decoder = self.decoder + self.params['dt'] * (anneal_const*np.exp(-1 *ii/150) + 1) * self.params['learning_rate_decoder'] * decoder_update;
                                
                            elif(exp_params.objective == 'classifier'):                        
                                self.decoder = self.decoder + self.params['dt'] * (anneal_const * np.exp(-1 * ii/150) + 1) * self.params['learning_rate_decoder'] * decoder_update;
                            else:
                                self.decoder = self.decoder + self.params['dt'] * (anneal_const * np.exp(-1 * ii/150) + 1) * self.params['learning_rate_decoder'] * decoder_update;
                        else:
                            if (exp_params.objective == 'linear'):
                                self.decoder = self.decoder + self.params['dt'] * anneal_const * self.params['learning_rate_decoder'] * decoder_update;
                                
                            elif(exp_params.objective == 'classifier'):                        
                                self.decoder = self.decoder + self.params['dt'] * anneal_const * self.params['learning_rate_decoder'] * decoder_update;
                            else:
                                self.decoder = self.decoder + self.params['dt'] * anneal_const * self.params['learning_rate_decoder'] * decoder_update;

#%% END CORE CODE
            #record input, output, firing rate and weight update information through time
            if (np.mod(ii, self.params['record_period'])==0 and (int(session_num/self.params['record_period']) > 0)):
                if(not(train)):
                    r_record[int(ii/self.params['record_period']),:,:] = r[:,np.arange(0,inputs.shape[2],self.params['record_res'])];
                else:
                    r_record = np.array([0])
                o_record[int(ii/self.params['record_period']),:,:] = o[:,np.arange(0,inputs.shape[2],self.params['record_res'])];
                i_record[int(ii/self.params['record_period']),:,:] = inputs[ii,:,np.arange(0,inputs.shape[2],self.params['record_res'])].T;
                th_record[int(ii/self.params['record_period']),:,:] = theta_i[ii,:,np.arange(0,inputs.shape[2],self.params['record_res'])].T;
                target_record[int(ii/self.params['record_period']),:,:] = target[ii,:,np.arange(0,inputs.shape[2],self.params['record_res'])].T;
            
        #record the old weights
        #for the initial training session
        if (not(train_2) and train):
            self.biases_old = self.biases
            self.weights_old = self.weights
            self.i_weights_old = self.i_weights
            self.decoder_old = self.decoder
        return r_record, o_record, i_record, th_record, target_record, obj_record
#%% Record a PCA of firing rate data
    def Net_PCA(self, fr_data):
        #perform a PCA on the outputs of each population in the network for some set of firing_rate data
        #if 'lesion' is true, then the data is stored under pcas_lesion rather than pcas
        component_number = 10; #extract 10 principle components
        pca = PCA(n_components = component_number);
        test_num = fr_data.shape[0]
        fr_data = fr_data.transpose((1,0,2)) #switch trial_num with neuron_num
        fr_data = fr_data.reshape((fr_data.shape[0], fr_data.shape[1]*fr_data.shape[2])).T
        pca.fit(fr_data);
        dur_max = int(params['T']/params['dt']/params['record_res'])
        #project the data onto the first few principle components
        num_comp_interesting = 4
        projection_mat = pca.components_[0:num_comp_interesting,:];
        projected_trajectories = np.dot(projection_mat, fr_data.T - np.reshape(np.mean(fr_data.T, axis = 1), (self.params['N'], 1)));
        projected_trajectories = np.reshape(projected_trajectories, (test_num, num_comp_interesting,dur_max)); #middle index is the trial number, and the last index is the time point
        self.pcas.append(pca) #store in the normal area if the data is not from a lesion study
        self.pca_trajectories.append(projected_trajectories)
if __name__ == '__main__':
    np.random.seed(seed = 120994)
    #import experiment parameters from the parameter definition file
    params = copy.deepcopy(exp_params.params) #the deepcopy prevents Python from continually having to reference another file, saving considerable time.
    decoder = copy.deepcopy(exp_params.decoder)
    param_fns = copy.deepcopy(exp_params.param_fns)
    input_params = copy.deepcopy(exp_params.input_params)
    #%% Generate data and run simulations
    w_0 = np.random.uniform(-0.3/np.sqrt(params['N']), 0.3/np.sqrt(params['N']), (params['N'], params['N']));
    i_w_0 = np.random.uniform(-0.5/np.sqrt(params['input_dim']), 0.5/np.sqrt(params['input_dim']), (params['N'], params['input_dim']))
    
    print(i_w_0[0,:])
    if (exp_params.objective == 'linear' or exp_params.objective == 'mixed'):
        initial_bias_offset = 0.;
    elif(exp_params.objective == 'bci'):
        initial_bias_offset = 0.;
    else:
        initial_bias_offset = 0.7;
    bias_0 = np.ones((params['N'],)) * initial_bias_offset;
    print(bias_0)
    network = PlasticNet(params, param_fns, input_params, decoder, w_0, bias_0, i_w_0)
    t_start = time.time()
    #run the initial training session
    r_record, o_record, input_record, theta_record, target_record, obj_record = network.train_run(params['T'], 0.1* np.ones((params['N'],)), train = True, rt_input = True);
    #trim the objective_record
    obj_record = obj_record[::params['record_period']]
    runtime = time.time() - t_start
    print('training time is: ' + str(runtime))
    t_start = time.time()
    #run the first test
    if (params['test_num'] > 0):
        r_test, o_test, i_test, theta_test, target_test, obj_test = network.train_run(params['T'], 0.1* np.ones((params['N'],)), train = False, tuning = True, rt_input = False)
        obj_test = obj_test[::params['record_period']]
        if (r_test.shape[0] > 0):
            network.Net_PCA(r_test)
    runtime = time.time() - t_start
    print('test time is: ' + str(runtime))
    t_start = time.time()
    
    #run the second training session
    r_record_2, o_record_2, input_record_2, theta_record_2, target_record_2, obj_record_2 = network.train_run(params['T'], 0.1* np.ones((params['N'],)), train = True, train_2 = True, rt_input = True);

    obj_record_2 = obj_record_2[::params['record_period']]
    runtime = time.time() - t_start
    print('training time 2 is: ' + str(runtime))
    
    #run the second test
    if (params['test_num'] > 0):
        r_test_2, o_test_2, i_test_2, theta_test_2, target_test_2, obj_test_2 = network.train_run(params['T'], 0.1* np.ones((params['N'],)), train = False, tuning = True, rt_input = False)
        obj_test_2 = obj_test_2[::params['record_period']]
        if (r_test_2.shape[0] > 0):
            network.Net_PCA(r_test_2)
    runtime = time.time() - t_start
    print('test time is: ' + str(runtime))
    
    
    decoder_new = network.decoder
    weights = network.weights
    i_weights = network.i_weights
    biases = network.biases






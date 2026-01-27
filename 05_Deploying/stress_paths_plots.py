# Plots and predictions for stress paths
# bav, 30.7.2025

import os, sys, pickle
import numpy as np
import matplotlib.pyplot as plt

base_dir = os.path.dirname(os.path.abspath(__file__))
shellsim_dir = os.path.dirname(base_dir)
training_dir = os.path.join(shellsim_dir, '04_Training')
sys.path.insert(0, training_dir)
from sampler_utils import Sampler_utils_vb              # the error shown here can be ignored, it still works below.
from sampler_utils import samplers
from dict_CC import dict_CC
from NN_call import predict_sig_D
from data_work_depl import transf_units
import scipy.io


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def plot_stress_paths(idx_eps, geom, idx_sig, idx_D, model_path, epnum, model_dim = 'ALLDIM', save_path = None, 
                      multirow = False, maxmin_lims = False, ax_quantiles_sig = [0,1], ax_quantiles_D = [0,1], 
                      NN_comp = None, exp_dict = None, rho_sublayer = False, allcols = False, train_points = False):
    '''
    idx_eps         (list)      desired dimension for input epsilon - if len(idx_eps) >1: the first eps_idx will be varied, the second will be in three steps.
    geom            (list)      geometrical input parameters (t, rho, CC)
    idx_sig         (list)      desired dimension for output sigma
    idx_D           (list)      desired dimension for output D
    model_path      (str)       path to trained model and underlying data
    epnum           (str)       episode number of trained model
    save_path       (str)       location to save the figure
    multirow        (bool)      if True: plots 8 rows with all sig_i and D_i corresponding to the selected eps_i
    maxmin_lims     (bool)      if True: considers ax_quantiles for the limits in multirow plot
                                if False: adjusts the axis according to the individual scales of each plot
    ax_quantiles_sig(list)      to determine the quantiles where the axis limits are cut off for sig
    ax_quantiles_D  (list)      to determine the quantiles where the axis limits are cut off for D
    NN_comp         (str-list)  if not None: Contains the path and ep number to a second NN which shall be 
                                compared to the first NN in model_path (e.g. ['04_Training\\new_data\\_simple_logs\\v_3xx', '_xx', 'TWODIM'])
    exp_dict        (dict)      experimental data that can potentially be added in this graph.
    rho_sublayer    (bool)      if True: rho only in 8 sublayers, not in every layer
    model_dim       (str)       can be "ONEDIM_x", "ONEDIM_y", "TWODIM" or "ALLDIM", depending on the NN architecture 
    allcols         (bool)      if True, prints all predictions of all stiffness matrix entries, not just the ones related to the varied epsilon
    train_points    (bool)      if True, will plot training points of the corresponding geometry and stress / stiffness 
                                in addition to the predicted and NLFEA curves


    main function, calls all functions below step-wise
    creates a new set of data for epsilon in just one direction (idx_eps) and calculates from it: 
        - sig_NN, sig_NLFEA
        - D_NN, D_NLFEA    
    Plots these values against each other for selected idx_sig and idx_D. This is like one single iteration in the FEA loop.
    Requires generation of new "predictions" with NN / calculations with NLFEA
    '''


    # Step 1: Sample a meaningful vector for idx_eps
    range_factor = 1
    inp_vector = sample_idx_eps(idx_eps, geom, model_path, model_dim, range_factor)
    inp_vector_star = sample_idx_eps(idx_eps, geom, model_path, model_dim, range_factor, small_value = 1e-9)
    print(f'Sampled eps values in [min, {range_factor}*max] of entire sampled epsilon range.')

    # Step 2a: Calculate all NLFEA values for given eps input
    sig_D_NLFEA = calculate_sig_D_NLFEA(inp_vector, rho_sublayer = rho_sublayer)
    print('Calculated NLFEA values')

    # Step 2a*: Calculate additional NLFEA value to understand where the large values in D come from
    sig_D_NLFEA_star = calculate_sig_D_NLFEA(inp_vector_star, rho_sublayer=rho_sublayer)
    print('Calculated NLFEA* values')

    # Step 2b: Calculate all LFEA values for given eps input
    sig_D_linel = calculate_sig_D_linel(inp_vector)
    print('Calculated LFEA values')

    # Step 2c: Calculate all NN values for given eps input
    sig_D_NN = predict_sig_D_NN(inp_vector, model_path, epnum, NN_comp, model_dim)
    print('Calculated NN values')

    # Step 2d: (Optional) Fetch training points to plot based on given geometry
    if train_points:
        sig_D_train = get_sig_D_train(inp_vector, geom, model_path, NN_comp, idx_eps)
        print('Fetched training data points')
    else: 
        sig_D_train = None

    # Step 3: Plot the figures
    if not multirow:
        fig1 = plot_singlerow(idx_eps, idx_sig, idx_D, inp_vector, inp_vector_star, sig_D_NLFEA, sig_D_NLFEA_star, sig_D_NN, sig_D_linel, sig_D_train, 
                              model_path, maxmin_lims, ax_quantiles_sig, ax_quantiles_D, NN_comp, exp_dict, model_dim, train_points)
    else: 
        # does not require idx_sig or idx_D anymore
        fig2 = plot_multirow(idx_eps, inp_vector, inp_vector_star, sig_D_NLFEA, sig_D_NLFEA_star, sig_D_NN, sig_D_linel, sig_D_train, 
                             model_path, maxmin_lims, ax_quantiles_sig, ax_quantiles_D, NN_comp, exp_dict, model_dim, allcols, train_points)
        
    # Step 4: Save the figures
    if save_path is not None and not multirow: 
        filename = 'stress_path_'+str(idx_eps)+'_'+str(idx_D)
        fig1.savefig(os.path.join(save_path, filename))
        print('Saved stress path figure ', filename, ' at ', save_path)
    elif multirow: 
        filename = 'stress_path_'+str(idx_eps)+'_multirow.png'
        fig2.savefig(os.path.join(save_path, filename))
        print('Saved stress path multirow figure at ', save_path)




################ sub-functions #####################

def plot_singlerow(idx_eps, idx_sig, idx_D, inp_vector, inp_vector_star, sig_D_NLFEA, sig_D_NLFEA_star, sig_D_NN, sig_D_linel, sig_D_train, model_path, maxmin_lims, 
                   ax_quantiles_sig, ax_quantiles_D, NN_comp, exp_dict, model_dim, train_points):
    
    model_name_1 = model_path[-5:]
    if NN_comp is not None: 
        model_name_2 = NN_comp[0][-5:]

    for key in inp_vector.keys():
        if inp_vector[key] is not None:
            # Plot histograms
            fig1, axs = plt.subplots(1, 3, figsize = [21, 7])
            axs[0].hist(inp_vector[key][:,idx_eps[0]], color = 'grey', alpha = 0.5, label = 'NLFEA = NN = linel')
            axs[0].set_xlabel('$\epsilon_{'+str(idx_eps[0])+'} [-] $')

            # Plot predictions vs NLFEA vs LFEA for sigma
            axs[1].plot(inp_vector[key][:,idx_eps[0]], sig_D_NLFEA[key]['sh_NLFEA'][:,idx_sig[0],0], color = 'black', label = 'NLFEA')
            axs[1].plot(inp_vector_star[key][:,idx_eps[0]], sig_D_NLFEA_star[key]['sh_NLFEA'][:,idx_sig[0],0], color = 'black', linestyle = '--', label = 'NLFEA*')
            axs[1].plot(inp_vector[key][:,idx_eps[0]], sig_D_NN[key]['sh_NN'][:,idx_sig[0]], color = 'lightblue', linestyle = '--', label = model_name_1)
            if NN_comp is not None: 
                axs[1].plot(inp_vector[key][:,idx_eps[0]], sig_D_NN[key]['sh_NN_1'][:,idx_sig[0]], color = 'coral', linestyle = '--', label = model_name_2)
            axs[1].plot(inp_vector[key][:,idx_eps[0]], sig_D_linel[key]['sh_linel'][:,idx_sig[0]], color = 'lightgrey', linestyle = ':', label = 'Lin.El.')
            if exp_dict is not None:
                axs[1].plot(exp_dict['eps_exp'], exp_dict['sig_exp'], color = 'black', linestyle = ':', label = 'Experiment')
            if train_points: 
                axs[1].scatter(sig_D_train[key]['eh_train'][:, idx_eps[0]], sig_D_train[key]['sh_train'][:, idx_sig[0]], color = 'lightblue', alpha = 0.3, 
                               marker = 'x', label = "Training data " + model_name_1)
                if NN_comp is not None: 
                    axs[1].scatter(sig_D_train[key]['eh_train1'][:, idx_eps[0]], sig_D_train[key]['sh_train1'][:, idx_sig[0]], color = 'coral', alpha = 0.3,
                               marker = 'x', label = "Training data " + model_name_2)
            axs[1].set_xlabel('$\epsilon_{'+str(idx_eps[0])+'} [-] $')
            axs[1].set_ylabel('$\sigma_{'+str(idx_sig[0])+'} [N,mm] $')
            

            # Plot predictions vs NLFEA vs LFEA for D
            axs[2].plot(inp_vector[key][:,idx_eps[0]], sig_D_NLFEA[key]['D_NLFEA'][:,idx_D[0],idx_D[1]], color = 'black', label = 'NLFEA')
            axs[2].plot(inp_vector_star[key][:,idx_eps[0]], sig_D_NLFEA_star[key]['D_NLFEA'][:,idx_D[0],idx_D[1]], color = 'black', linestyle = '--', label = 'NLFEA*')
            axs[2].plot(inp_vector[key][:,idx_eps[0]], sig_D_NN[key]['D_NN'][:,idx_D[0], idx_D[1]], color = 'lightblue', linestyle = '--',  label = model_name_1)
            if NN_comp is not None: 
                axs[2].plot(inp_vector[key][:,idx_eps[0]], sig_D_NN[key]['D_NN_1'][:,idx_D[0], idx_D[1]], color = 'coral', linestyle = '--',  label = model_name_2)
            axs[2].plot(inp_vector[key][:,idx_eps[0]], sig_D_linel[key]['D_linel'][:,idx_D[0], idx_D[1]], color = 'lightgrey', linestyle = ':', label = 'Lin.El.')
            if exp_dict is not None:
                axs[2].plot(exp_dict['eps_exp'], exp_dict['De_exp'], color = 'black', linestyle = ':', label = 'Experiment')
            if train_points: 
                axs[2].scatter(sig_D_train[key]['eh_train'][:, idx_eps[0]], sig_D_train[key]['D_train'][:, idx_D[0], idx_D[1]], color = 'lightblue', alpha = 0.3,
                               marker = 'x', label = "Training data " + model_name_1)
                if NN_comp is not None: 
                    axs[2].scatter(sig_D_train[key]['eh_train1'][:, idx_eps[0]], sig_D_train[key]['D_train1'][:, idx_D[0], idx_D[1]], color = 'coral', alpha = 0.3, 
                               marker = 'x', label = "Training data " + model_name_2)
            axs[2].set_xlabel('$\epsilon_{'+str(idx_eps[0])+'} [-]$')
            axs[2].set_ylabel('$D_{'+str(idx_D[0])+','+str(idx_D[1])+'} [N,mm]$')


            axs[0].legend()
            axs[2].legend()

            if maxmin_lims:
                lims_sig = get_min_max_lims(idx_eps[0],idx_sig, model_path, model_dim, ax_quantiles_sig, tag = 'sig')
                lims_D = get_min_max_lims(idx_eps[0],idx_sig, model_path, model_dim, ax_quantiles_D, tag = 'D')
                axs[1].set_ylim(lims_sig['y_lim_0'])
                axs[2].set_ylim(lims_D['y_lim_1'])


    return fig1

def plot_multirow(idx_eps, inp_vector, inp_vector_star, sig_D_NLFEA, sig_D_NLFEA_star, sig_D_NN, sig_D_linel, sig_D_train, model_path, maxmin_lims, 
                  ax_quantiles_sig, ax_quantiles_D, NN_comp, exp_dict, model_dim, allcols, train_points):
    if allcols: 
        num_cols = 5
    else: 
        num_cols = 3

    if model_dim == 'TWODIM': 
        num_rows = 3
    else: 
        num_rows = 8
    fig2, axs = plt.subplots(num_rows, num_cols, figsize = [7*num_cols, 7*num_rows])

    model_name_1 = model_path[-5:]
    if NN_comp is not None: 
        model_name_2 = NN_comp[0][-5:]

    colors1, colors2, colors3 = get_colors_from_map(inp_vector)

    if exp_dict is not None: 
        raise UserWarning('This functionality has not yet been implemented. Please implement or plot without experimental data.')
    
    for key in inp_vector.keys():
        if inp_vector[key] is not None:
            for i in range(num_rows):
                # Plot histograms
                if key == '0':
                    if len(idx_eps) > 1 and i == idx_eps[1]:
                        ns = inp_vector[key].shape[0]
                        inp_vec_hist = np.concatenate((inp_vector['0'][:int(ns/3),idx_eps[1]], inp_vector['min'][:int(ns/3),idx_eps[1]], inp_vector['max'][:int(ns/3),idx_eps[1]]), axis = 0)
                        axs[i,0].hist(inp_vec_hist, color = 'grey', alpha = 0.5, label = 'NLFEA = NN = linel')
                        axs[i,0].set_xlabel('$\epsilon_{'+str(i)+'} [-] $')
                    else:
                        axs[i,0].hist(inp_vector[key][:,i], color = 'grey', alpha = 0.5, label = 'NLFEA = NN = linel')
                        axs[i,0].set_xlabel('$\epsilon_{'+str(i)+'} [-] $')
                        
                # Plot predictions vs NLFEA vs LFEA for sigma
                if len(idx_eps) > 1:
                    labels = ['NLFEA, $\epsilon_'+ str(idx_eps[1]) + ' = $'+str(np.round(inp_vector[key][0, idx_eps[1]], 3))+' [-]',
                              'NN, $\epsilon_'+ str(idx_eps[1]) + ' = $'+str(np.round(inp_vector[key][0, idx_eps[1]], 3))+' [-]', 
                              'NLFEA*']
                else: 
                    labels = ['NLFEA', 'NN', 'NLFEA*']
                axs[i,1].plot(inp_vector[key][:,idx_eps[0]], sig_D_NLFEA[key]['sh_NLFEA'][:,i,0], color = colors1[key], label = labels[0])
                axs[i,1].plot(inp_vector_star[key][:,idx_eps[0]], sig_D_NLFEA_star[key]['sh_NLFEA'][:,i,0], color = colors1[key], linestyle = '--', label = labels[2])
                axs[i,1].plot(inp_vector[key][:,idx_eps[0]], sig_D_NN[key]['sh_NN'][:,i], color = colors2[key], linestyle = '--', label = labels[1]+ ' ' + model_name_1)
                if NN_comp is not None: 
                    axs[i,1].plot(inp_vector[key][:,idx_eps[0]], sig_D_NN[key]['sh_NN_1'][:,i], color = colors3[key], linestyle = '--', label = labels[1] + ' ' + model_name_2)
                if key == '0':
                    axs[i,1].plot(inp_vector[key][:,idx_eps[0]], sig_D_linel[key]['sh_linel'][:,i], color = 'lightgrey', linestyle = ':', 
                              label = 'Lin.El.')
                    axs[i,1].set_xlabel('$\epsilon_{'+str(idx_eps[0])+'} [-] $')
                    axs[i,1].set_ylabel('$\sigma_{'+str(i)+'} [N,mm] $')
                if train_points:
                    axs[i,1].scatter(sig_D_train[key]['eh_train'][:, idx_eps[0]], sig_D_train[key]['sh_train'][:, i], color = colors2[key], alpha = 0.3, 
                            marker = 'x', label = "Training data " + model_name_1)
                    if NN_comp is not None: 
                        axs[i,1].scatter(sig_D_train[key]['eh_train1'][:, idx_eps[0]], sig_D_train[key]['sh_train1'][:, i], color = colors3[key], alpha = 0.3,
                                marker = 'x', label = "Training data " + model_name_2)

                # Plot predictions vs NLFEA vs LFEA for D
                if not allcols:
                    axs[i,2].plot(inp_vector[key][:,idx_eps[0]], sig_D_NLFEA[key]['D_NLFEA'][:,i, idx_eps[0]], color = colors1[key], label = labels[0])
                    axs[i,2].plot(inp_vector_star[key][:,idx_eps[0]], sig_D_NLFEA_star[key]['D_NLFEA'][:,i, idx_eps[0]], color = colors1[key], linestyle = '--', label = labels[2])
                    axs[i,2].plot(inp_vector[key][:,idx_eps[0]], sig_D_NN[key]['D_NN'][:,i, idx_eps[0]], color = colors2[key], linestyle = '--', label = labels[1]+ ' ' + model_name_1)
                    if NN_comp is not None: 
                        axs[i,2].plot(inp_vector[key][:,idx_eps[0]], sig_D_NN[key]['D_NN_1'][:,i, idx_eps[0]], color = colors3[key], linestyle = '--', label = labels[1]+ ' ' + model_name_2)
                    if key == '0':
                        axs[i,2].plot(inp_vector[key][:,idx_eps[0]], sig_D_linel[key]['D_linel'][:,i,idx_eps[0]], color = 'lightgrey', linestyle = ':', 
                                    label = 'Lin.El.')
                        axs[i,2].set_xlabel('$\epsilon_{'+str(idx_eps[0])+'} [-]$')
                        axs[i,2].set_ylabel('$D_{'+str(i)+','+str(idx_eps[0])+'} [N,mm]$')
                    if train_points:
                        axs[i,2].scatter(sig_D_train[key]['eh_train'][:, idx_eps[0]], sig_D_train[key]['D_train'][:, i, idx_eps[0]], color = colors2[key], alpha = 0.3, 
                                marker = 'x', label = "Training data " + model_name_1)
                        if NN_comp is not None: 
                            axs[i,2].scatter(sig_D_train[key]['eh_train1'][:, idx_eps[0]], sig_D_train[key]['D_train1'][:, i, idx_eps[0]], color = colors3[key], alpha = 0.3, 
                                    marker = 'x', label = "Training data " + model_name_2)
                    

                # Plot predictions vs NLFEA vs LFEA for all D (if allcols = True)
                if allcols:
                    for j in [2,3,4]:
                        axs[i,j].plot(inp_vector[key][:,idx_eps[0]], sig_D_NLFEA[key]['D_NLFEA'][:,i, j-2], color = colors1[key], label = labels[0])
                        axs[i,j].plot(inp_vector_star[key][:,idx_eps[0]], sig_D_NLFEA_star[key]['D_NLFEA'][:,i, j-2], color = colors1[key], linestyle = '--', label = labels[2])
                        axs[i,j].plot(inp_vector[key][:,idx_eps[0]], sig_D_NN[key]['D_NN'][:,i, j-2], color = colors2[key], linestyle = '--', label = labels[1]+ ' ' + model_name_1)
                        if NN_comp is not None: 
                            axs[i,j].plot(inp_vector[key][:,idx_eps[0]], sig_D_NN[key]['D_NN_1'][:,i, j-2], color = colors3[key], linestyle = '--', label = labels[1]+ ' ' + model_name_2)
                        if key == '0':
                            axs[i,j].plot(inp_vector[key][:,idx_eps[0]], sig_D_linel[key]['D_linel'][:,i,j-2], color = 'lightgrey', linestyle = ':', 
                                        label = 'Lin.El.')
                            axs[i,j].set_xlabel('$\epsilon_{'+str(idx_eps[0])+'} [-]$')
                            axs[i,j].set_ylabel('$D_{'+str(i)+','+str(j-2)+'} [N,mm]$')
                        if train_points:
                            axs[i,j].scatter(sig_D_train[key]['eh_train'][:, idx_eps[0]], sig_D_train[key]['D_train'][:, i, j-2], color = colors2[key], alpha = 0.3, 
                                    marker = 'x', label = "Training data " + model_name_1)
                            if NN_comp is not None: 
                                axs[i,j].scatter(sig_D_train[key]['eh_train1'][:, idx_eps[0]], sig_D_train[key]['D_train1'][:, i, j-2], color = colors3[key], alpha = 0.3, 
                                        marker = 'x', label = "Training data " + model_name_2)
                    

        else:
            pass

    if maxmin_lims:
        for i in range(num_rows):
            lims_sig = get_min_max_lims(idx_eps[0],i, model_path, model_dim, ax_quantiles_sig, tag = 'sig')
            axs[i,1].set_ylim(lims_sig['y_lim_0'])
            if allcols: 
                j_vec = [2,3,4] 
            else: 
                j_vec = [2]
            for j in j_vec:
                lims_D = get_min_max_lims(j-2,i, model_path, model_dim, ax_quantiles_D, tag = 'D')
                axs[i,j].set_ylim(lims_D['y_lim_1'])

    axs[0,0].legend()
    axs[0,num_cols-1].legend()

    return fig2


def sample_idx_eps(idx_eps, geom, model_path, model_dim, range_factor = 1, num_samples = 100, small_value = 1e-20):
    '''
    samples eps_inp only for the dimension given in idx_eps  
    
    idx_eps      (list)       desired dimension for input epsilon
    geom         (list)       geometrical input parameters (t, rho, CC)
    model_path   (str)        path to sampled data
    model_dim    (str)        architecture type of the model.
    range_factor (float)      to reduce the max. range of epsilons in the input vector
    num_samples  (int)        amount of values to be sampled in eps

    '''
    
    with open(os.path.join(model_path, 'mat_data_np_TrainEvalTest.pkl'),'rb') as handle:
                mat_data_np = pickle.load(handle)

    if len(idx_eps) < 2: 
        eps_vec = small_value*np.ones((num_samples, 8))                  # other values are set to "zero", i.e. small_value here
        if model_dim == 'ONEDIM_y':
            max_idx_eps = np.max(mat_data_np['X_train'][:,0])
            min_idx_eps = np.min(mat_data_np['X_train'][:,0])
        elif model_dim == 'TWODIM': 
            max_idx_eps = np.max(mat_data_np['X_train'][:,idx_eps[0]])
            min_idx_eps = np.min(mat_data_np['X_train'][:,idx_eps[0]])
        else: 
            # works for ONEDIM_x or ALLDIM
            max_idx_eps = np.max(mat_data_np['X_train'][:,idx_eps[0]])
            min_idx_eps = np.min(mat_data_np['X_train'][:,idx_eps[0]])
        idx_eps_vec = np.linspace(min_idx_eps, range_factor*max_idx_eps, num_samples)
        if idx_eps[0] > 2 and idx_eps[0] < 6:
            idx_eps_vec = idx_eps_vec/10                           # convert to 1/mm
        eps_vec[:,idx_eps[0]] = idx_eps_vec
        
        t_vec = np.tile(np.array(geom), (num_samples, 1))

        inp_vec = {
            'min': None, 
            '0': np.concatenate((eps_vec, t_vec), axis = 1),                    # this now has the desired shape
            'max': None,
        }

    else: 
        if model_dim == 'ONEDIM_x' or model_dim == 'ONEDIM_y': 
            raise UserWarning('Onedim model, it does not make sense to plot multiple epsilon. Please change idx_eps to len(idx_eps) = 1')
        elif model_dim == 'TWODIM':
            pass
        
        eps_vec = small_value*np.ones((num_samples, 8))                  # other values are set to "zero", i.e. small_value here
        max_idx_eps = np.max(mat_data_np['X_train'][:,idx_eps[0]])
        min_idx_eps = np.min(mat_data_np['X_train'][:,idx_eps[0]])
        idx_eps_vec = np.linspace(min_idx_eps, range_factor*max_idx_eps, num_samples)
        if idx_eps[0] > 2 and idx_eps[0] < 6:
            idx_eps_vec = idx_eps_vec/10                           # convert to 1/mm
        eps_vec[:,idx_eps[0]] = idx_eps_vec

        max_idx_eps1 = np.max(mat_data_np['X_train'][:,idx_eps[1]])
        min_idx_eps1 = np.min(mat_data_np['X_train'][:,idx_eps[1]])
        eps_vec_min, eps_vec_max = small_value*np.ones((num_samples,8)), small_value*np.ones((num_samples,8))
        eps_vec_min[:,idx_eps[0]] = idx_eps_vec                             # assign same varying values of eps_0
        eps_vec_max[:,idx_eps[0]] = idx_eps_vec                             # assign same varying values of eps_0
        eps_vec_min[:,idx_eps[1]] = np.tile(min_idx_eps1, (num_samples,))  # add constant min or max values of eps_1
        eps_vec_max[:,idx_eps[1]] = np.tile(max_idx_eps1, (num_samples,))

        t_vec = np.tile(np.array(geom), (num_samples, 1))

        inp_vec = {
             'min': np.concatenate((eps_vec_min, t_vec), axis = 1),
             '0': np.concatenate((eps_vec, t_vec), axis = 1),
             'max': np.concatenate((eps_vec_max, t_vec), axis = 1),
        }


    return inp_vec


def calculate_sig_D_NLFEA(inp_vec, rho_sublayer):
    sig_D_NLFEA = {}
    for key in inp_vec.keys():
        if inp_vec[key] is not None:
            dict_CC.update({'fsy': 435, 'fsu': 470, 'Es': 205e3, 'Esh': 8e3, 'D': 16, 'Dmax': 16, 's': 200})
            mat_dict = dict_CC
            analytical_sampler = Sampler_utils_vb(E1 = None, nu1=0, E2=None, nu2=None, mat_dict = mat_dict)
            t_extended = analytical_sampler.extend_material_parameters(inp_vec[key][:,8:])
            if t_extended.shape[1] == 10:
                # ensure that four input parameters are used for the geometry (note: for NLFEA calculation this isn't strictly required.)
                t_extended = np.concatenate((t_extended[:,:2],t_extended[:,1].reshape(-1,1), t_extended[:,2:]), axis = 1 )
            with HiddenPrints():
                dict_sampler = analytical_sampler.D_an(inp_vec[key][:,0:8], t_extended, num_layers=20, mat = 3, calc_meth='single', discrete='andreas', 
                                                       rho_sublayer = rho_sublayer)

            sig_D_NLFEA[key] = {
                    'sh_NLFEA': dict_sampler['sig_a'],
                    'D_NLFEA': dict_sampler['D_a'],
            }
        else:
            pass

    return sig_D_NLFEA

def calculate_sig_D_linel(inp_vec):
    '''
    calculates linel sig and D
    assumes nu = 0, choses E according to given concrete class
    '''

    sig_D_LFEA = {}
    index = int(np.where(dict_CC['CC'] == inp_vec['0'][0,-1])[0])
    E_linel = dict_CC['Ec'][index]
    analytical_sampler = Sampler_utils_vb(E1 = E_linel, nu1 = 0, E2 = None, nu2 = None, mat_dict = None)
    with HiddenPrints():
        dict_sampler = analytical_sampler.D_an(inp_vec['0'][:,0:8], inp_vec['0'][:,8], num_layers = 20, mat = 1, calc_meth='single', discrete = 'andreas')
    
    sig_D_LFEA['0'] = {
        'sh_linel': dict_sampler['sig_a'],
        'D_linel': dict_sampler['D_a']
    }

    return sig_D_LFEA

def predict_sig_D_NN(inp_vec, model_path, epnum, NN_comp, model_dim):
    # transform units of inp_vec
    sig_D_NN = {}
    sig_D_NN_1 = {}
    potential_dim = ['ONEDIM_x', 'ONEDIM_y', 'TWODIM', 'ALLDIM']
    if model_dim not in potential_dim:
        raise UserWarning(f'Please use one of the following model dimensions: {potential_dim}')
    if  NN_comp is not None and NN_comp[2] not in potential_dim:
        raise UserWarning(f'Please use one of the following model dimensions: {potential_dim}')
    for key in inp_vec.keys():
        if inp_vec[key] is not None:
            with open(os.path.join(model_path, 'inp.pkl'),'rb') as handle:
                inp = pickle.load(handle)
            if inp['input_size'] == 7 and inp_vec[key].shape[1] != 12: 
                inp_vec[key] = np.concatenate((inp_vec[key][:,:10], inp_vec[key][:,9].reshape(-1,1), inp_vec[key][:,10:11]), axis = 1)
            elif inp['input_size'] == 7: 
                pass #inp_vec[key] = inp_vec[key]
            elif inp['input_size'] == 6: 
                inp_vec[key] = inp_vec[key][:,:11]
            input_j = transf_units(inp_vec[key], 'eps-t', forward = True)
            path_model_name = os.path.join(model_path+'\\best_trained_model_'+epnum+'.pt')
            mat_NN_sig = predict_sig_D(input_j, model_path, path_model_name, 'train', transf_type = 'st-stitched', predict = 'sig', sc=False, model_dim = model_dim)
            mat_NN_D = predict_sig_D(input_j, model_path, path_model_name, 'train', transf_type = 'st-stitched', predict = 'D', sc=False, model_dim = model_dim)

            sig_NN = transf_units(mat_NN_sig['sig_h'], 'sig', forward = False)
            De_NN = transf_units(mat_NN_D['D_pred'], 'D', forward = False, linel=False)

            sig_D_NN[key] = {
                'sh_NN': sig_NN,
                'D_NN': De_NN,
            }

            if NN_comp is not None:
                with open(os.path.join(NN_comp[0], 'inp.pkl'),'rb') as handle:
                    inp = pickle.load(handle)
                if inp['input_size'] == 7 and inp_vec[key].shape[1] != 12: 
                    inp_vec[key] = np.concatenate((inp_vec[key][:,:10], inp_vec[key][:,9].reshape(-1,1), inp_vec[key][:,10:]), axis = 1)
                elif inp['input_size'] == 7: 
                    pass #inp_vec[key] = inp_vec[key]
                elif inp['input_size'] == 6:
                    inp_vec[key] = inp_vec[key][:,:11]
                input_j = transf_units(inp_vec[key], 'eps-t', forward = True)
                path_model_name = os.path.join(NN_comp[0]+'\\best_trained_model_'+NN_comp[1]+'.pt')
                mat_NN_sig_1 = predict_sig_D(input_j, NN_comp[0], path_model_name, 'train', transf_type = 'st-stitched', predict = 'sig', sc=False, model_dim = NN_comp[2])
                mat_NN_D_1 = predict_sig_D(input_j, NN_comp[0], path_model_name, 'train', transf_type = 'st-stitched', predict = 'D', sc=False, model_dim = NN_comp[2])

                sig_NN_1 = transf_units(mat_NN_sig_1['sig_h'], 'sig', forward = False)
                De_NN_1 = transf_units(mat_NN_D_1['D_pred'], 'D', forward = False, linel=False)

                sig_D_NN_1[key] = {
                    'sh_NN_1': sig_NN_1,
                    'D_NN_1': De_NN_1,
                }
                sig_D_NN[key].update(sig_D_NN_1[key])
        else:
            pass

    return sig_D_NN

def get_sig_D_train(inp_vec, geom, model_path, NN_comp, idx_eps):
    sig_D_train = {}
    sig_D_train1 = {}
    for key in inp_vec.keys():
        if inp_vec[key] is not None: 
            with open(os.path.join(model_path, 'mat_data_np_TrainEvalTest.pkl'),'rb') as handle:
                mat_data_np = pickle.load(handle)
            with open(os.path.join(model_path, 'inp.pkl'),'rb') as handle:
                inp = pickle.load(handle)

            dim = inp['out_size']
            eps_t_train_d = mat_data_np['X_train']
            sig_train_d = transf_units(mat_data_np['y_train'][:,:dim], 'sig', forward = False, linel = False)
            D_train_d = transf_units(mat_data_np['y_train'][:,dim:].reshape((-1, dim, dim)), 'D', forward = False, linel = False)
            # mask 1: geometry filtering
            if (eps_t_train_d[:,dim:].shape[1] - np.array(geom).shape[0]) != 0: 
                geom_ = np.concatenate((geom[0:2], [geom[1]], [geom[2]]), axis = 0)
            else: 
                geom_ = np.array(geom)
            mask1 = np.all(abs(eps_t_train_d[:,dim:] - geom_) < 1e-5, axis = 1)    

            # mask 2: filter out small values of non- "idx_eps[0]" values
            eps_t_train_d_copy = np.zeros((eps_t_train_d.shape[0], dim))
            eps_t_train_d_copy[:,idx_eps[0]] = eps_t_train_d[:,idx_eps[0]]
            mask2 = np.all(abs(eps_t_train_d[:,:dim] - eps_t_train_d_copy) < 2e-6, axis = 1)
            
            mask = mask1 & mask2

            if mask.sum() < 1: 
                raise UserWarning('The filtering yielded zero points. Please reconsider your choice of "geom".')
            else: 
                print(f'Remanining rows after filtering mask 1 for NN {model_path[-5:]}: {mask1.sum()}')
                print(f'Remaining rows after filtering mask 2 for NN {model_path[-5:]}: {(mask1 & mask2).sum()}')

            sig_D_train[key] = {
                'eh_train': eps_t_train_d[:,:dim][mask],
                'sh_train': sig_train_d[mask],
                'D_train': D_train_d[mask],
            }
            
            if NN_comp is not None:
                with open(os.path.join(NN_comp[0], 'mat_data_np_TrainEvalTest.pkl'),'rb') as handle:
                    mat_data_np1 = pickle.load(handle)
                with open(os.path.join(NN_comp[0], 'inp.pkl'),'rb') as handle:
                    inp1 = pickle.load(handle)

                dim1 = inp1['out_size']
                eps_t_train_d1 = mat_data_np1['X_train']
                sig_train_d1 = transf_units(mat_data_np1['y_train'][:,:dim1], 'sig', forward = False, linel = False)
                D_train_d1 = transf_units(mat_data_np1['y_train'][:,dim1:].reshape((-1, dim1, dim1)), 'D', forward = False, linel = False)
                # mask 1: geometry filtering
                if (eps_t_train_d1[:,dim1:].shape[1] - np.array(geom).shape[0]) != 0: 
                    geom_ = np.concatenate((geom[0:2], [geom[1]], [geom[2]]), axis = 0)
                else: 
                    geom_ = np.array(geom)
                mask1_ = np.all(abs(eps_t_train_d1[:,dim1:] - geom_) < 1e-5, axis = 1)

                # mask 2: filter out small values of non- "idx_eps[0]" values
                eps_t_train_d_copy_ = np.zeros((eps_t_train_d1.shape[0], dim1))
                eps_t_train_d_copy_[:,idx_eps[0]] = eps_t_train_d1[:,idx_eps[0]]
                mask2_ = np.all(abs(eps_t_train_d1[:,:dim1] - eps_t_train_d_copy_) < 2e-6, axis = 1)
                
                mask_ = mask1_ & mask2_

                if mask_.sum() < 1: 
                    raise UserWarning('The filtering yielded zero points. Please reconsider your choice of "geom".')
                else: 
                    print(f'Remanining rows after filtering mask 1 for NN {NN_comp[0][-5:]}: {mask1_.sum()}')
                    print(f'Remanining rows after filtering mask 2 for NN {NN_comp[0][-5:]}: {(mask1_ & mask2_).sum()}')

                sig_D_train1[key] = {
                    'eh_train1': eps_t_train_d1[:,:dim1][mask_],
                    'sh_train1': sig_train_d1[mask_],
                    'D_train1': D_train_d1[mask_],
                }
                sig_D_train[key].update(sig_D_train1[key])

        else: 
            pass

    return sig_D_train


def get_min_max_lims(idx_eps, idx, model_path, model_dim, ax_quantiles=[0,1], small_value = 1e-20, tag = 'sig'):
    '''
    gets min and max lims according to dataset. 
    idx_eps             (list)      same as in upper function
    idx                 (list)      index of given plot for sigma
    model_path          (str)       path to data
    model_dim           (str)       dimensionality of NN model
    ax_quantiles        (list)      quantiles --> plot these instead of min-max
    tag                 (str)       'sig' or 'D'

    '''
    with open(os.path.join(model_path, 'mat_data_np_TrainEvalTest.pkl'),'rb') as handle:
                mat_data_np = pickle.load(handle)

    if tag == 'sig': 
        sig_transf = transf_units(mat_data_np['y_train'], 'sig', forward = False, linel = False)
        sig_min = np.quantile(sig_transf[:,idx], ax_quantiles[0])
        sig_max = np.quantile(sig_transf[:,idx], ax_quantiles[1])
        D_min, D_max = None, None

    elif tag == 'D':
        if model_dim == 'ONEDIM_x': 
            y_train_data = mat_data_np['y_train'][:,1:2].reshape((-1,1))
            y_train_data_ = np.concatenate((y_train_data, np.ones((y_train_data.shape[0], 63))*small_value), axis = 1).reshape((-1,8,8))
            D_transf = transf_units(y_train_data_, 'D', forward = False, linel = False)
        elif model_dim == 'ONEDIM_y':
            y_train_data = mat_data_np['y_train'][:,1:2].reshape((-1,1))
            y_train_data_ = np.concatenate((np.ones((y_train_data.shape[0], 9))*small_value, y_train_data, np.ones((y_train_data.shape[0], 54))*small_value), axis = 1).reshape((-1,8,8))
            D_transf = transf_units(y_train_data_, 'D', forward = False, linel = False)
        elif model_dim == 'TWODIM':
            y_train_data = mat_data_np['y_train']
            y_train_data_ = np.concatenate((y_train_data[:,3:6], np.ones((y_train_data.shape[0], 5))*small_value,
                                            y_train_data[:,6:9], np.ones((y_train_data.shape[0], 5))*small_value,
                                            y_train_data[:,9:12], np.ones((y_train_data.shape[0], 5))*small_value,
                                            np.ones((y_train_data.shape[0], 40))*small_value), axis = 1).reshape((-1,8,8))
            D_transf = transf_units(y_train_data_, 'D', forward = False, linel = False)
        else:
            D_transf = transf_units(mat_data_np['y_train'][:,8:72].reshape((-1,8,8)), 'D', forward = False, linel = False)
        D_min = np.quantile(D_transf[:,idx_eps,idx], ax_quantiles[0])
        D_max = np.quantile(D_transf[:,idx_eps,idx], ax_quantiles[1])
        sig_min, sig_max = None, None

    lims = {
         'y_lim_0': [sig_min, sig_max],
         'y_lim_1': [D_min, D_max],
    }

    return lims


def get_colors_from_map(inp_vector):
    cmap1, cmap2, cmap3 = plt.cm.gist_yarg, plt.cm.Blues, plt.cm.RdPu
    values = np.linspace(0.2,0.8,3)
    colors1, colors2, colors3 = {}, {}, {}
    for v, key in zip(values, inp_vector.keys()):
        colors1[key], colors2[key], colors3[key] = cmap1(v), cmap2(v), cmap3(v)
    return colors1, colors2, colors3



################ preprocessing experimental data #####################


def preprocess_exp(path, b, test_no):
    '''
    path        (str)       path to experimental data (matlab file)
    b           (int)       width of specimen - required for stiffness calculation, in [mm]
    test_no     (int)       =1 or =2 depending on which experimental data shall be plotted.
    '''
    mat = scipy.io.loadmat(path)

    # unpack dict
    EA1 = mat['vb'][0][0][0]        # [MN]
    EA2 = mat['vb'][0][0][1]        # [MN]
    f1 = mat['vb'][0][0][2]         # [kN]
    f2 = mat['vb'][0][0][3]         # [kN]
    eps1 = mat['vb'][0][0][4]       # [mm/m]
    eps2 = mat['vb'][0][0][5]       # [mm/m]

    # calculate D from EA by dividing through width b and transforming to [N/mm]
    D_1 = EA1*1e6/b
    D_2 = EA2*1e6/b

    # calculate sig from force by dividing thorugh width b and transforming to [N/mm]
    sig_1 = f1*1e3/b
    sig_2 = f2*1e3/b
    
    # get eps by transforming to [-]
    eps_1 = eps1*1e-3
    eps_2 = eps2*1e-3

    exp_dict_1 = {
        'eps_exp': eps_1, 
        'sig_exp': sig_1,
        'De_exp': D_1,
    }

    exp_dict_2 = {
        'eps_exp': eps_2, 
        'sig_exp': sig_2,
        'De_exp': D_2,
    }

    if test_no == 1: 
        return exp_dict_1
    elif test_no == 2: 
        return exp_dict_2
    



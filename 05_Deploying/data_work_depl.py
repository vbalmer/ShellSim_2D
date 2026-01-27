'''
NOTE: This is a copy of the file data_work (only difference: wandb commented out). Do not change it's contents unless you also want to 
change it in the original class.
Copy date: 21.10.2024
Updated transform_data function on 31.10.2024
Updated all changes made on 20.11.2024
Updated all changes made again on 06.03.2025: 
    "plots_mike_dataset" can remain different compared to "data_work.py" from training folder
    "plots_paper_comp" can remain in the "data_work.py" from the training folder, does not need a local copy in data_work_depl.py
    "numit" in multiple_diagonal_plots is added as a variable here (to be able to plot it per iteration in deplyoment)
Update on 17.03.2025: 
    changed the statistics to "stitched" for sobolev-trained model
    added scaling in transformation function
Update on 19.03.2025: Added the limit 'fixation' to multiple_diagonal_plots, added video creating
'''


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset
from matplotlib.offsetbox import AnchoredText
import os
# import wandb
import pickle
import torch
import glob
import math
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.transforms
from matplotlib.font_manager import FontProperties
import shutil
import re
import cv2      # for video

'''--------------------------------------LOADING DATA--------------------------------------------------'''

# Define torch dataset
class MyDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
    
'''--------------------------------------LOADING DATA--------------------------------------------------'''


def read_data(path: str, id: str):
    '''
    Reads in the data from given path
    id      (str)           Identifyer of data to be read: either 'sig', 'eps' or 't'
    '''
    if 'add' in id:
        with open(os.path.join(path, 'newadd_data_'+id.replace("_add", "") +'.pkl'),'rb') as handle:
            new_data = pickle.load(handle)
    else: 
        with open(os.path.join(path, 'new_data_'+id +'.pkl'),'rb') as handle:
            new_data = pickle.load(handle)
    
    # transform data to numpy
    new_data = pd.DataFrame.from_dict(new_data)
    new_data_np = new_data.to_numpy()

    return new_data_np

def save_data(X_train:np.array, X_eval:np.array, X_test:np.array, y_train:np.array, y_eval:np.array, y_test:np.array, path:str):
    mat = {}
    mat['X_train'] = X_train
    mat['y_train'] = y_train
    mat['X_eval'] = X_eval
    mat['y_eval'] = y_eval
    mat['X_test'] = X_test
    mat['y_test'] = y_test

    with open(os.path.join(path, 'mat_data_np_TrainEvalTest.pkl'), 'wb') as fp:
        pickle.dump(mat, fp)
    return


def data_to_torch(X_train_t:np.array, y_train_t:np.array, X_eval_t:np.array, y_eval_t:np.array, 
                  X_test_t:np.array, y_test_t:np.array, 
                  path: str, sobolev: bool, batch_size: int, tag = False):
    '''
    Takes numpy arrays and creates torch dataloaders
    Saves relevant information to mat_data_TrainEvalTest

    '''

    # transform data to torch
    X_train_tt = torch.from_numpy(X_train_t)
    X_train_tt = X_train_tt.type(torch.float32)
    X_eval_tt = torch.from_numpy(X_eval_t)
    X_eval_tt = X_eval_tt.type(torch.float32)
    X_test_tt = torch.from_numpy(X_test_t)
    X_test_tt = X_test_tt.type(torch.float32)

    if sobolev:
        y_train_tt = torch.from_numpy(y_train_t)
        y_train_tt = y_train_tt.type(torch.float32)
        y_eval_tt = torch.from_numpy(y_eval_t)
        y_eval_tt = y_eval_tt.type(torch.float32)
        y_test_tt = torch.from_numpy(y_test_t)
        y_test_tt = y_test_tt.type(torch.float32)
    elif not sobolev:
        if not tag:
            y_train_tt = torch.from_numpy(y_train_t[:,0:8])
            y_train_tt = y_train_tt.type(torch.float32)
            y_eval_tt = torch.from_numpy(y_eval_t[:,0:8])
            y_eval_tt = y_eval_tt.type(torch.float32)
            y_test_tt = torch.from_numpy(y_test_t[:,0:8])
            y_test_tt = y_test_tt.type(torch.float32)
        else: 
            y_train_tt = torch.from_numpy(y_train_t)
            y_train_tt = y_train_tt.type(torch.float32)
            y_eval_tt = torch.from_numpy(y_eval_t)
            y_eval_tt = y_eval_tt.type(torch.float32)
            y_test_tt = torch.from_numpy(y_test_t)
            y_test_tt = y_test_tt.type(torch.float32)


    # Creating Datasets
    data_train_t = MyDataset(X_train_tt, y_train_tt)
    data_eval_t = MyDataset(X_eval_tt, y_eval_tt)
    data_test_t = MyDataset(X_test_tt, y_test_tt)

    # Transform to DataLoaders
    train_loader = DataLoader(data_train_t, batch_size, shuffle=True) #, num_workers=16) 
    val_loader = DataLoader(data_eval_t, batch_size, shuffle=False) #, num_workers=16)
    test_loader = DataLoader(data_test_t, batch_size, shuffle=False) #, num_workers=16)

    print('Training set size:', len(list(train_loader)))
    print('Validation set size:', len(list(val_loader)))
    print('Test set size:', len(list(test_loader)))

    # Save data
    mat = {}
    mat['X_train_tt'] = X_train_tt
    mat['y_train_tt'] = y_train_tt
    mat['X_eval_tt'] = X_eval_tt
    mat['y_eval_tt'] = y_eval_tt
    mat['X_test_tt'] = X_test_tt
    mat['y_test_tt'] = y_test_tt
    mat['test_loader'] = test_loader

    with open(os.path.join(path, 'mat_data_TrainEvalTest.pkl'), 'wb') as fp:
        pickle.dump(mat, fp)


    loaders = {
        "train": train_loader, 
        "val": val_loader, 
        "test": test_loader
    }

    return loaders, mat

'''--------------------------------------HISTOGRAMS,STATISTICS--------------------------------------------------'''

def histogram(X:np.array, y:np.array, amt_data_points:int, nbins: int, name: str, path: str):
    '''
    Plots histograms of training and evaluation data set to check for consistency with previous steps of process.
    '''
    
    keys1 = ['eps_x', 'eps_y', 'eps_xy', 'chi_x', 'chi_y', 'chi_xy', 'gamma_xz', 'gamma_yz']
    keys2 = ['n_x', 'n_y', 'n_xy', 'm_x', 'm_y', 'm_xy', 'v_xz', 'v_yz']
    keys3 = np.array([['D11', 'D12', 'D13', 'D14', 'D15', 'D16', 'D17', 'D18'],
            ['D21', 'D22', 'D23', 'D24', 'D25', 'D26', 'D27', 'D28'],
            ['D31', 'D32', 'D33', 'D34', 'D35', 'D36', 'D37', 'D38'],
            ['D41', 'D42', 'D43', 'D44', 'D45', 'D46', 'D47', 'D48'],
            ['D51', 'D52', 'D53', 'D54', 'D55', 'D56', 'D57', 'D58'],
            ['D61', 'D62', 'D63', 'D64', 'D65', 'D66', 'D67', 'D68'],
            ['D71', 'D72', 'D73', 'D74', 'D75', 'D76', 'D77', 'D78'],
            ['D81', 'D82', 'D83', 'D84', 'D85', 'D86', 'D87', 'D88']])

    if name == 'sig': 
        keys = keys2
        plot_data = y
    elif name == 'eps':
        keys = keys1
        plot_data = X
    elif name == 't':
        keys = 't'
        plot_data = X
    elif name == 't2':
        keys = ['t1', 't2', 'nl']
        plot_data = X
    elif name == 'De':
        keys = keys3
        plot_data = y
    else: 
        raise "Error: No real name defined"

    if name == 'sig' or name == 'eps':
        fig2, axs2 = plt.subplots(2, 4, sharey=True, tight_layout=True)
        for i in range(3):
            axs2[0, i].set_xlabel(keys[i])
            axs2[1, i].set_xlabel(keys[i+3])
            axs2[0, i].hist(plot_data[:,i], bins=nbins)
            axs2[1, i].hist(plot_data[:,i+3], bins=nbins)        
        axs2[0, 3].set_xlabel(keys[6])
        axs2[1, 3].set_xlabel(keys[7])    
        axs2[0, 3].hist(plot_data[:,6], bins=nbins)
        axs2[1, 3].hist(plot_data[:,7], bins=nbins)
    elif name == 't' or name == 't2': 
        fig2, axs2 = plt.subplots(1, X.shape[1], sharey = True, tight_layout = True)
        for i in range(X.shape[1]):
            if X.shape[1] == 1:
                axs2.hist(plot_data[:,i], bins = nbins)
                axs2.set_xlabel(keys[i])
            else: 
                axs2[i].hist(plot_data[:,i], bins = nbins)
                axs2[i].set_xlabel(keys[i])
    elif name == 'De':
        fig2, axs2 = plt.subplots(8, 8, figsize = (16, 16), sharey = True)
        for i in range(8):
            for j in range(8):
                axs2[i, j].hist(plot_data[:, i, j], bins = nbins)
                axs2[i, j].set_xlabel(keys3[i,j])


    if len(plot_data) == amt_data_points:
        fig2.suptitle('All Data, n = ' + str(len(plot_data)))
    elif len(plot_data) > 0.5*amt_data_points:
        fig2.suptitle('Training Data, n = ' + str(len(plot_data)))
    elif len(plot_data) < 0.2*amt_data_points:
        fig2.suptitle('Test Data, n = ' + str(len(plot_data)))
    else:
        fig2.suptitle('Validation Data, n = ' + str(len(plot_data)))


    if path is not None: 
        plt.savefig(os.path.join(path, 'hist_'+name))
        print('saved histogram ', name)

    return



def histogram_torch(data: DataLoader, amt_data_points:int, nbins: int, name: str):       
    '''
    Plots histograms of training and evaluation data set to check for consistency with previous steps of process.
    '''
    all_features = np.zeros((len(list(data)), 8))
    all_labels = np.zeros((len(list(data)), 8))

    for i, (features, labels) in enumerate(data):
        feat_eps = features[:,0:8]
        all_features[i,:] = feat_eps.numpy()
        all_labels[i,:] = labels.numpy()
    # print(all_features.shape)
    
    fig2, axs2 = plt.subplots(2, 4, sharey=True, tight_layout=True)
    keys1 = ['eps_x', 'eps_y', 'eps_xy', 'chi_x', 'chi_y', 'chi_xy', 'gamma_xz', 'gamma_yz']
    keys2 = ['n_x', 'n_y', 'n_xy', 'm_x', 'm_y', 'm_xy', 'v_xz', 'v_yz']

    if name == 'sig': 
        keys = keys2
        plot_data = all_labels
    elif name == 'eps':
        keys = keys1
        plot_data = all_features
    else: 
        raise "Error: No real name defined"

    for i in range(3):
        axs2[0, i].set_xlabel(keys[i])
        axs2[1, i].set_xlabel(keys[i+3])
        axs2[0, i].hist(plot_data[:,i], bins=nbins)
        axs2[1, i].hist(plot_data[:,i+3], bins=nbins)        
    axs2[0, 3].set_xlabel(keys[6])
    axs2[1, 3].set_xlabel(keys[7])    
    axs2[0, 3].hist(plot_data[:,6], bins=nbins)
    axs2[1, 3].hist(plot_data[:,7], bins=nbins)
    
    if len(data) == amt_data_points:
        fig2.suptitle('All Data, n = ' + str(len(data)))
    elif len(data) > 0.5*amt_data_points:
        fig2.suptitle('Training Data, n = ' + str(len(data)))
    elif len(data) < 0.2*amt_data_points:
        fig2.suptitle('Test Data, n = ' + str(len(data)))
    else:
        fig2.suptitle('Validation Data, n = ' + str(len(data)))

    plt.show()

    return plt


def statistics_pd(data:pd.DataFrame):
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    max = data.max(axis=0)
    min = data.min(axis=0)
    q_5 = data.quantile(0.05, axis=0)
    q_95 = data.quantile(0.95, axis=0)
    stats = {
        'mean': mean,
        'std': std,
        'max': max,
        'min': min,
        'q_5': q_5,
        'q_95': q_95,
    }
    return stats

def statistics(data:np.array):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    max = np.max(data, axis=0)
    min = np.min(data, axis=0)
    q_95 = np.percentile(data, 95, axis=0)
    q_5 = np.percentile(data, 5, axis=0)
    stats = {
        'mean': mean,
        'std': std,
        'max': max,
        'min': min,
        'q_5': q_5,
        'q_95': q_95
    }
    return stats



def transform_data(data:np.array, stats_:dict, forward: bool, type: list, sc: False):
    """
    Returns standardised data set with transformed data for the forward case.
    Or returns original-scale data set transformed back from given transformed data.

    Input: 
    data            (pd.DataFrame)      still the individual data sets for sig, eps
    stats           (dict)              contains statistical values that are constant for the transformation
    forward         (bool)              True: forward transform (to standardnormaldistr.), 
                                        False: backward transform to original values
    type            (str-list)          if 'std': standard normal distr.
                                        if 'range': max-min
                                        if 'st-stitched': standard normal distr. 
                                        but for D entries: transformed according to sigma and eps - std
    sc              (bool)              True: scaling of data (+10) for potentially better training
                                        
    Output: 
    new_data       (np.array)           Uniformly distributed "new" data set
    """

    # Calculate new data

    data_stdnorm = np.zeros((data.shape))
    data_nonorm = np.zeros((data.shape))
    data_range = np.zeros((data.shape))
    data_norange = np.zeros((data.shape))
    new_data = np.zeros((data.shape))
    D_coeff = np.zeros((8,8))
    np_data = data
    
    # for the stitched transformation of the D-matrix, need transformation of D based on D_coeff:
    for j in range(8):
        for k in range(8):
            D_coeff[j,k] = stats_['stats_y_train']['std'][j]/stats_['stats_X_train']['std'][k]
    D_coeff_ = D_coeff.reshape((1, 64))

    # Carry out the transformation
    if forward:
        for i in range(data.shape[1]):
            # Defining the correct statistical values (x or y depending on input data)
            if 'x' in type[i]: 
                stats = stats_['stats_X_train']
            elif 'y' in type[i]: 
                stats = stats_['stats_y_train']
            else: 
                raise RuntimeError('Please define an appropriate transformation type including the variable x or y to be transformed.')
            
            # Transforming the data
            if 'std' in type[i]:
                # 1 - Transformation to Standard Normal distribution
                if stats['std'][i] == 0:
                    stats['std'][i] = 1.0
                data_stdnorm[:,i] = (np_data[:,i]-stats['mean'][i]*np.ones(np_data.shape[0]))/(stats['std'][i]*np.ones(np_data.shape[0]))
                new_data[:,i] = data_stdnorm[:,i]
                if sc: 
                    new_data[:,i] = data_stdnorm[:,i]+10
            elif 'range' in type[i]:
                # 2 - Transformation maximin / range
                if stats['max'][i]-stats['min'][i] == 0:
                    stats['max'][i] = 1
                data_range[:,i] = (np_data[:,i]-stats['min'][i]*np.ones(np_data.shape[0]))/((stats['max'][i]-stats['min'][i])*np.ones(np_data.shape[0]))
                new_data[:,i] = data_range[:,i]*1
            elif 'st-stitched' in type[i]: 
                # 3 - Transformation of D, based on physically meaningful transformation of statistical variables of sigma and eps
                # assumes shape of y: sig+D (nx8)+(nx64)
                data_stdnorm[:,i] = np.divide(1,D_coeff_[0,i-8])*np_data[:,i]
                new_data[:,i] = data_stdnorm[:,i]


    elif not forward:
        for i in range(data.shape[1]):
            # Defining the correct statistical values (x or y depending on input data)
            if 'x' in type[i]: 
                stats = stats_['stats_X_train']
            elif 'y' in type[i]: 
                stats = stats_['stats_y_train']
            else: 
                raise RuntimeError('Please define an appropriate transformation type including the variable x or y to be transformed.')
            
            # Transforming the data
            if 'std' in type[i]:
                # 1 - Redo Standard normal distribution
                if sc: 
                    np_data[:,i] = np_data[:,i]-10
                if stats['std'][i] == 0:
                    stats['std'][i] = 1.0
                data_nonorm[:,i] = np_data[:,i]*stats['std'][i]*np.ones(np_data.shape[0])+stats['mean'][i]*np.ones(np_data.shape[0])
                new_data[:,i] = data_nonorm[:,i]
                
            elif 'range' in type[i]:
                # 2 - Redo maximin / range
                if stats['max'][i]-stats['min'][i] == 0:
                    stats['max'][i] = 1
                data_norange[:,i] = (np_data[:,i]/1)*(stats['max'][i]-stats['min'][i]) + stats['min'][i]*np.ones(np_data.shape[0])
                new_data[:,i] = data_norange[:,i]
            elif 'st-stitched' in type[i]: 
                # 3 - Transformation of D, based on physically meaningful transformation of statistical variables of sigma and eps
                # assumes shape of y: sig+D (nx8)+(nx64)
                data_nonorm[:,i] = D_coeff_[0,i-8]*np_data[:,i]
                new_data[:,i] = data_nonorm[:,i]
    return new_data


'''-------------------------------------- INITIALISE WANDB --------------------------------------------------'''

# def init_wandb(inp, inp1, inp2, name, project_name):
#     run = wandb.init(
#     project = project_name,
#     config = {
#         "inp": inp,
#         "inp1": inp1,
#         "inp2": inp2,
#         "data_name": name
#         },
#     )
#     return run



def transf_units(vec:np.array, id:str, forward:bool, linel = True):
    '''
    Transforms the units for input from simulation to training and back
    vec:        (np.array)          Vector to be transformed
    id:         (str)               Identifier 'sig', 'D' or 'eps-t' depending on the desired transformation of vec
                                    Expected shapes: sig: (n,8), eps-t: (n,9), D: (n, 8,8)
    forward:    (bool)              If true: Transformation is forward (i.e. from N, mm to MN, cm)
                                    If false: Transformation is backward (i.e. from MN, cm to N, mm)
    linel       (bool)              If true: Sets values of D_mb = 0 in stiffness matrix
                                    If false: Also transforms the units of D_mb according to the correct transformation
    '''

    if id == 'sig':
        sig_t = np.zeros_like(vec)
        if forward:
            sig_t[:, 0:3] = vec[:, 0:3]*(10**(-6))*(10**(1))
            sig_t[:, 3:6] = vec[:, 3:6]*(10**(-6))*1
            sig_t[:, 6:8] = vec[:, 6:8]*(10**(-6))*(10**(1))
            out_vec = sig_t
        else: 
            sig_t[:, 0:3] = vec[:, 0:3]*(10**(6))*(10**(-1))
            sig_t[:, 3:6] = vec[:, 3:6]*(10**(6))*1
            sig_t[:, 6:8] = vec[:, 6:8]*(10**(6))*(10**(-1))
            out_vec = sig_t
    elif id == 'sig-t':
        sig_t = np.zeros_like(vec)
        if forward:
            sig_t[:, 0:3] = vec[:, 0:3]*(10**(-6))*(10**(1))
            sig_t[:, 3:6] = vec[:, 3:6]*(10**(-6))*1
            sig_t[:, 6:8] = vec[:, 6:8]*(10**(-6))*(10**(1))
            sig_t[:,8:] = vec[:,8:]
            out_vec = sig_t
        else: 
            sig_t[:, 0:3] = vec[:, 0:3]*(10**(6))*(10**(-1))
            sig_t[:, 3:6] = vec[:, 3:6]*(10**(6))*1
            sig_t[:, 6:8] = vec[:, 6:8]*(10**(6))*(10**(-1))
            sig_t[:,8:] = vec[:,8:]
            out_vec = sig_t

    elif id == 'eps-t':
        eps_t = np.zeros_like(vec)
        if forward:
            eps_t[:, 0:3] = vec[:, 0:3]*1
            eps_t[:, 3:6] = vec[:, 3:6]*(10**1)
            eps_t[:, 6:8] = vec[:, 6:8]*1
            eps_t[:,8:] = vec[:,8:]
            out_vec = eps_t
        else:
            eps_t[:, 0:3] = vec[:, 0:3]*1
            eps_t[:, 3:6] = vec[:, 3:6]*(10**(-1))
            eps_t[:, 6:8] = vec[:, 6:8]*1
            eps_t[:,8:] = vec[:,8:]
            out_vec = eps_t
    elif id == 'eps': 
        eps_t = np.zeros_like(vec)
        if forward: 
            eps_t[:, 0:3] = vec[:, 0:3]*1
            eps_t[:, 3:6] = vec[:, 3:6]*(10**1)
            eps_t[:, 6:8] = vec[:, 6:8]*1
            out_vec = eps_t
        else: 
            eps_t[:, 0:3] = vec[:, 0:3]*1
            eps_t[:, 3:6] = vec[:, 3:6]*(10**(-1))
            eps_t[:, 6:8] = vec[:, 6:8]*1
            out_vec = eps_t
    
    elif id == 'D':

        D_t = np.zeros_like(vec)
        if linel:
            if forward:
                D_t[:, 0:3, 0:3] = vec[:, 0:3, 0:3]*(10**(-6))*(10**(1))
                D_t[:,3:6, 3:6] = vec[:, 3:6, 3:6]*(10**(-6))*(10**(-1))
                D_t[:, 6:8, 6:8] = vec[:, 6:8, 6:8]*(10**(-6))*(10**(1))
                out_vec = D_t
            else:
                D_t[:, 0:3, 0:3] = vec[:, 0:3, 0:3]*(10**(6))*(10**(-1))
                D_t[:,3:6, 3:6] = vec[:, 3:6, 3:6]*(10**(6))*(10**(1))
                D_t[:, 6:8, 6:8] = vec[:, 6:8, 6:8]*(10**(6))*(10**(-1))
                out_vec = D_t
        else:
            if forward:
                D_t[:, 0:3, 0:3] = vec[:, 0:3, 0:3]*(10**(-6))*(10**(1))
                D_t[:,3:6, 3:6] = vec[:, 3:6, 3:6]*(10**(-6))*(10**(-1))
                D_t[:, 6:8, 6:8] = vec[:, 6:8, 6:8]*(10**(-6))*(10**(1))
                D_t[:,0:3, 3:6] = vec[:, 0:3, 3:6]*(10**(-6))
                D_t[:,3:6, 0:3] = vec[:, 3:6, 0:3]*(10**(-6))
                out_vec = D_t
            else: 
                D_t[:, 0:3, 0:3] = vec[:, 0:3, 0:3]*(10**(6))*(10**(-1))
                D_t[:,3:6, 3:6] = vec[:, 3:6, 3:6]*(10**(6))*(10**(1))
                D_t[:, 6:8, 6:8] = vec[:, 6:8, 6:8]*(10**(6))*(10**(-1))
                D_t[:,0:3, 3:6] = vec[:, 0:3, 3:6]*(10**(6))
                D_t[:,3:6, 0:3] = vec[:, 3:6, 0:3]*(10**(6))
                out_vec = D_t

    return out_vec






'''--------------------------------------POSTPROCESSING--------------------------------------------------'''

def calculate_errors(Y, predictions, stats, transf, id = 'sig'):

    if id == 'sig':
        num_cols = 9
        num_cols_plt = 8
    elif id == 'De': 
        num_cols = 72
        num_cols_plt = 17
    elif id == 'De-NLRC': 
        num_cols = 72
        num_cols_plt = 38
        if stats['stats_y_train']['mean'].shape[0] == 38:
            num_cols = 38
            num_cols_plt = 38
    elif id == 'eps':
        num_cols = 8
        num_cols_plt = 8

    ### Transform the units of the statistics which relate back to the train set to the units desired in the diagonal plot
    if transf == 't' or transf == 't-inv':
        mean_train_ = 0*np.ones((1,num_cols))
        q_5_train_ = -1.645*np.ones((1,num_cols))
        q_95_train_ = 1.645*np.ones((1,num_cols))
    elif transf == 'o' or transf == 'o-inv':
        # the statistics are already in the units MN, cm
        mean_train_ = stats['stats_y_train']['mean'][0:num_cols].reshape((1,num_cols))
        q_5_train_ = stats['stats_y_train']['q_5'][0:num_cols].reshape((1,num_cols))
        q_95_train_ = stats['stats_y_train']['q_95'][0:num_cols].reshape((1,num_cols))
    elif transf == 'u' or transf == 'u-inv':
        stats_new = {}
        if id == 'sig' or id == 'eps':
            for key, value in stats['stats_y_train'].items():
                stats_new[key] = transf_units(value[0:num_cols].reshape(1,num_cols), id, forward=False).reshape(num_cols,)
            mean_train_ = stats_new['mean'][0:num_cols].reshape((1,num_cols))
            q_5_train_ = stats_new['q_5'][0:num_cols].reshape((1,num_cols))
            q_95_train_ = stats_new['q_95'][0:num_cols].reshape((1,num_cols))         
        elif id == 'De' or id == 'De-NLRC':
            if id == 'De':
                LINEL = True
            elif id == 'De-NLRC':
                LINEL = False
            if num_cols == 38:
                for key, value in stats['stats_y_train'].items():
                    value_ = np.zeros((1,8,8))
                    value_[:,:6,:6] = value[0:num_cols-2].reshape((1,6,6))
                    value_[:,6,6] = value[36].reshape((1,))
                    value_[:,7,7] = value[37].reshape((1,))
                    stats_new[key] = transf_units(value_, 'D', forward=False, linel = LINEL).reshape(64,)
                mean_train_ = stats_new['mean'].reshape((1,64))
                q_5_train_ = stats_new['q_5'].reshape((1,64))
                q_95_train_ = stats_new['q_95'].reshape((1,64))
            else: 
                for key, value in stats['stats_y_train'].items():
                    stats['stats_y_train'][key] = transf_units(value[0:8].reshape(1,8), 'sig', forward=False, linel = LINEL).reshape(8,)
                    stats_new[key] = transf_units(value[8:num_cols].reshape((1,8,8)), 'D', forward=False, linel = LINEL).reshape(64,)
                mean_train_ = np.hstack((stats['stats_y_train']['mean'][0:8], stats_new['mean'])).reshape((1,num_cols))
                q_5_train_ = np.hstack((stats['stats_y_train']['q_5'][0:8], stats_new['q_5'])).reshape((1,num_cols))
                q_95_train_ = np.hstack((stats['stats_y_train']['q_95'][0:8], stats_new['q_95'])).reshape((1,num_cols))


    if id == 'De':
        # Kick out irrelevant data (that should be zero) and reshape matrix to (num_rows x 12) format, for lin.el. calculation
        mean_train_De = mean_train_[:,8:72].reshape((-1, 8, 8))
        q_5_train_De = q_5_train_[:,8:72].reshape((-1, 8, 8))
        q_95_train_De = q_95_train_[:,8:72].reshape((-1, 8, 8))

        mean_train = np.hstack((mean_train_De[:,0:2, 0:2].reshape((-1,4)), mean_train_De[:,2, 2].reshape((-1,1)),
                                mean_train_De[:,3:5, 3:5].reshape((-1,4)), mean_train_De[:,5, 5].reshape((-1,1)),
                                mean_train_De[:,0:2, 3:5].reshape((-1,4)), mean_train_De[:,2, 5].reshape((-1,1)),
                                mean_train_De[:,6, 6].reshape((-1,1)), mean_train_De[:,7, 7].reshape((-1,1))))
        q_5_train = np.hstack((q_5_train_De[:,0:2, 0:2].reshape((-1,4)), q_5_train_De[:,2, 2].reshape((-1,1)),
                                q_5_train_De[:,3:5, 3:5].reshape((-1,4)), q_5_train_De[:,5, 5].reshape((-1,1)),
                                q_5_train_De[:,0:2, 3:5].reshape((-1,4)), q_5_train_De[:,2, 5].reshape((-1,1)),
                                q_5_train_De[:,6, 6].reshape((-1,1)), q_5_train_De[:,7, 7].reshape((-1,1))))
        q_95_train = np.hstack((q_95_train_De[:,0:2, 0:2].reshape((-1,4)), q_95_train_De[:,2, 2].reshape((-1,1)),
                                q_95_train_De[:,3:5, 3:5].reshape((-1,4)), q_95_train_De[:,5, 5].reshape((-1,1)),
                                q_95_train_De[:,0:2, 3:5].reshape((-1,4)), q_95_train_De[:,2, 5].reshape((-1,1)),
                                q_95_train_De[:,6, 6].reshape((-1,1)), q_95_train_De[:,7, 7].reshape((-1,1))))
    elif id == 'De-NLRC':
        # for nonlinear version of De (i.e. Dmb is not zero)
        if num_cols == 38:
            # if directly predicting D with the network
            if transf == 't' or transf == 'o': 
                mean_train = mean_train_
                q_5_train = q_5_train_
                q_95_train = q_95_train_
            else: 
                mean_train_De = mean_train_[:,0:64].reshape((-1, 8, 8))
                q_5_train_De = q_5_train_[:,0:64].reshape((-1, 8, 8))
                q_95_train_De = q_95_train_[:,0:64].reshape((-1, 8, 8))
                mean_train = np.concatenate((mean_train_De[:,:6,:6].reshape((-1, 36)), mean_train_De[:,6,6].reshape((-1, 1)), mean_train_De[:,7,7].reshape((-1, 1))), axis = 1)
                q_5_train = np.concatenate((q_5_train_De[:,:6,:6].reshape((-1,36)), q_5_train_De[:,6,6].reshape((-1,1)), q_5_train_De[:,7,7].reshape((-1,1))), axis = 1)
                q_95_train = np.concatenate((q_95_train_De[:,:6,:6].reshape((-1,36)), q_95_train_De[:,6,6].reshape((-1,1)), q_95_train_De[:,7,7].reshape((-1,1))), axis = 1)
        else:
            # if predicting sigma with network and D with derivatives
            mean_train_De = mean_train_[:,8:72].reshape((-1, 8, 8))
            q_5_train_De = q_5_train_[:,8:72].reshape((-1, 8, 8))
            q_95_train_De = q_95_train_[:,8:72].reshape((-1, 8, 8))
            mean_train = np.concatenate((mean_train_De[:,:6,:6].reshape((-1, 36)), mean_train_De[:,6,6].reshape((-1, 1)), mean_train_De[:,7,7].reshape((-1, 1))), axis = 1)
            q_5_train = np.concatenate((q_5_train_De[:,:6,:6].reshape((-1,36)), q_5_train_De[:,6,6].reshape((-1,1)), q_5_train_De[:,7,7].reshape((-1,1))), axis = 1)
            q_95_train = np.concatenate((q_95_train_De[:,:6,:6].reshape((-1,36)), q_95_train_De[:,6,6].reshape((-1,1)), q_95_train_De[:,7,7].reshape((-1,1))), axis = 1)
    
    elif id == 'sig' or id == 'eps':
        mean_train = mean_train_
        q_5_train = q_5_train_
        q_95_train = q_95_train_



    ### Calculate errors
    r_squared2 = np.zeros((1,num_cols_plt))
    rse_max=np.zeros((1,num_cols_plt))
    n_5p, n_10p, rmse, aux_, aux__, rrmse = np.zeros((1,num_cols_plt)), np.zeros((1,num_cols_plt)), np.zeros((1,num_cols_plt)), np.zeros((1,num_cols_plt)), np.zeros((1,num_cols_plt)), np.zeros((1,num_cols_plt))
    rrse_max, nrse_max, nrmse, log_max, mean_log_err = np.zeros((1,num_cols_plt)), np.zeros((1,num_cols_plt)), np.zeros((1,num_cols_plt)), np.zeros((1,num_cols_plt)), np.zeros((1,num_cols_plt))
    rse, nrse, log_err = np.zeros((Y.shape[0], num_cols_plt)), np.zeros((Y.shape[0], num_cols_plt)), np.zeros((Y.shape[0], num_cols_plt))
    # Delta_max_i, Delta_max_max, Delta_max_mean = np.zeros((Y.shape[0], num_cols_plt)), np.zeros((1,num_cols_plt)), np.zeros((1,num_cols_plt))
    for i in range(num_cols_plt):
        Y_col = Y[:, i].flatten()
        pred_col = predictions[:,i].flatten()
        r_squared2[:,i] = np.corrcoef(Y_col, pred_col)[0, 1]**2
        rse[:,i] = np.sqrt((pred_col-Y_col)**2)
        rse_max[:,i] = np.max(rse[:,i])
        rmse[:,i] = np.sqrt(np.mean((pred_col - Y_col) ** 2))
        aux_[:,i] = np.sqrt(np.mean((mean_train[:,i]*np.ones(Y_col.shape) - Y_col) ** 2))
        if aux_[:,i].any() == 0:
            aux_[aux_[:,i] == 0] = 1
        rrmse[:,i] = np.divide(rmse[:,i],aux_[:,i])
        rrse_max[0,i] = np.max(np.divide(np.sqrt((pred_col-Y_col)**2), aux_[:,i]))

        # Delta_max_i[:,i] = np.abs(Y_col - pred_col)/np.maximum(np.abs(Y_col), np.abs(pred_col))
        # Delta_max_max[0,i] = np.max(Delta_max_i[:,i])
        # Delta_max_mean[0,i] = np.mean(Delta_max_i[:,i])

        # Calculate normalised RMSE
        aux__[:,i] = q_95_train[:,i]-q_5_train[:,i]
        if aux__[:,i].any() == 0:
            aux__[aux__[:,i] == 0] = 1
        nrse[:,i] = np.divide(np.sqrt((pred_col-Y_col)**2), aux__[:,i])*100
        nrmse[:,i] = np.divide(rmse[:,i], aux__[:,i])
        nrse_max[:,i] = np.max(nrse[:,i]/100)

        # Calculate log error (not used at the moment)
        log_err[:,i] = np.log(pred_col+1)-np.log(Y_col+1)
        mean_log_err[:,i] = np.mean(log_err[:,i])
        log_max[:,i] = np.max(log_err[:,i])


    errors = {
        'rse': rse,
        'rse_max': rse_max,
        'rmse': rmse,
        'nrmse': nrmse,
        'nrse': nrse,
        'nrse_max': nrse_max,
        'rrmse': rrmse,
        'rrse_max': rrse_max,
        'r_squared2': r_squared2,
    }

    return errors


def multiple_diagonal_plots_wrapper(save_path: str, plot_data:dict, stats:dict, color='nrse'):
    for model in ['exp1', 'exp2', 'exp3', 'MoE']:
        # normalised plots
        multiple_diagonal_plots(save_path+'\MoE\\'+model, plot_data[model]['all_test_labels_t'], plot_data[model]['all_predictions_t'], 't', stats, color)
        # original scale plots
        multiple_diagonal_plots(save_path+'\MoE\\'+model, plot_data[model]['all_test_labels'], plot_data[model]['all_predictions'], 'o', stats, color)
        # simulation scale plots
        plot_data_label_u = transf_units(plot_data[model]['all_test_labels'], 'sig', forward = False)
        plot_data_pred_u = transf_units(plot_data[model]['all_predictions'], 'sig', forward = False)
        multiple_diagonal_plots(save_path+'\MoE\\'+model, plot_data_label_u, plot_data_pred_u, 'u', stats, 'rse')
    return



def multiple_diagonal_plots(save_path: str, Y: np.array, predictions: np.array, transf:str, stats:dict, color='nrse', numit = 0,
                            Y_train = None, pred_train = None, xlim = None, ylim = None, norms_ = None):
    ''''
    save_path       (str)           path where images are saved
    Y               (np.array)      Ground truth
    predictions     (np.array)      Predictions
    transf          (str)           't': transformed(normalised), 'o': original scale [MN, cm], 'u': units for simulation [N,mm]
    stats           (dict)          data statistics for normalising / relativising the RMSE
    color           (str)           scatter color: if 'nrse': only one colour bar across all plots. If 'rse': 3 separate colorbars for n, m, v
    y_train         (np.array)      Values of training data (if want to test on training data)
    pred_train      (np.array)      Values of training data (if want to test on training data)
    xlim, ylim      (np.array)      Defines the xlim and ylim for all 8 stress values. For checking the smaller-range predictions. 
                                    If these values are chosen, then the error calculation will also be carried out just for the data in this range. 
                                    nRMSE will still be calcualated w.r.t. entire training data range. 
    norms_           (list)          fix max, min of colorbar range, such that over different iterations the same colors are used.                                 
    '''

    
    # print(shutil.which("latex"))
    # mpl.rcParams.update(mpl.rcParamsDefault)
    # plt.rcParams["text.usetex"] = True

    # plt.rcParams.update({
    #     "text.usetex": True,
    #     "font.family": "Helvetica",
    #     "font.size": 12,
    #     })
    

    if "inv" in transf:
        errors = calculate_errors(Y, predictions, stats, transf, id = 'eps')
    elif (xlim and ylim) is not None:
        mask_x = (Y[:,:8]>=xlim[0]) & (Y[:,:8]<=xlim[1])
        mask_y = (predictions[:,:8]>=ylim[0]) & (predictions[:,:8]<=ylim[1])
        mask = mask_x & mask_y
        valid_rows = mask.all(axis=1)
        if np.sum(valid_rows) == 0:
            print('No points found for this region. Please increase range.')
            # set only one point to true to be able to continue the program
            mask_f = np.zeros(400, dtype = bool)
            mask_f[0] = True
            valid_rows = mask_f
        errors = calculate_errors(Y[valid_rows,:8], predictions[valid_rows,:8], stats, transf, id = 'sig')
    else: 
        errors = calculate_errors(Y, predictions, stats, transf, id = 'sig')



    # Plot figure
    fig, axa = plt.subplots(3, 3, figsize=[15.5, 12], dpi=100)
    fig.subplots_adjust(wspace=0.5)
    index_mask = np.array([[0,1,2],
                           [3,4,5],
                           [6,7,7]])
    num_rows = Y.shape[0]
    mask_labels = np.zeros((Y.shape[0], 8))

    if transf == 'o':
        plotname = np.array([['$n_x$', '$n_y$', '$n_{xy}$'],
                            ['$m_x$', '$m_y$', '$m_{xy}$'],
                            ['$v_x$', '$v_y$', '$v_y$']])
        plotname_p = np.array([[r'$\tilde{n}_{x}$', r'$\tilde{n}_{y}$', r'$\tilde{n}_{xy}$'],
                            [r'$\tilde{m}_x$', r'$\tilde{m}_y$', r'$\tilde{m}_{xy}$'],
                            [r'$\tilde{v}_x$', r'$\tilde{v}_y$', r'$\tilde{v}_y$']])
        units = np.array([[r'$\rm [MN/cm]$', r'$\rm [MN/cm]$', r'$\rm [MN/cm]$'],
                          [r'$\rm [MNcm/cm]$', r'$\rm [MNcm/cm]$', r'$\rm [MNcm/cm]$'],  
                          [r'$\rm [MN/cm]$', r'$\rm [MN/cm]$', r'$\rm [MN/cm]$']])
        
    elif transf == 'u':
        plotname = np.array([['$n_x$', '$n_y$', '$n_{xy}$'],
                            ['$m_x$', '$m_y$', '$m_{xy}$'],
                            ['$v_x$', '$v_y$', '$v_y$']])
        plotname_p = np.array([[r'$\tilde{n}_{x}$', r'$\tilde{n}_{y}$', r'$\tilde{n}_{xy}$'],
                            [r'$\tilde{m}_x$', r'$\tilde{m}_y$', r'$\tilde{m}_{xy}$'],
                            [r'$\tilde{v}_x$', r'$\tilde{v}_y$', r'$\tilde{v}_y$']])
        units = np.array([[r'$\rm [N/mm]$', r'$\rm [N/mm]$', r'$\rm [N/mm]$'],
                          [r'$\rm [Nmm/mm]$', r'$\rm [Nmm/mm]$', r'$\rm [Nmm/mm]$'],  
                          [r'$\rm [N/mm]$', r'$\rm [N/mm]$', r'$\rm [N/mm]$']])

    elif transf =='t':
        plotname = np.array([[r'$n_{x,norm}$', r'$n_{y,norm}$', r'$n_{xy,norm}$'],
                        [r'$m_{x,norm}$', r'$m_{y,norm}$', r'$m_{xy,norm}$'],
                        [r'$v_{x,norm}$', r'$v_{y,norm}$', r'$v_{y,norm}$']])
        plotname_p = np.array([[r'$\tilde{n}_{x,norm}$', r'$\tilde{n}_{y,norm}$', r'$\tilde{n}_{xy,norm}$'],
                        [r'$\tilde{m}_{x,norm}$', r'$\tilde{m}_{y,norm}$', r'$\tilde{m}_{xy,norm}$'],
                        [r'$\tilde{v}_{x,norm}$', r'$\tilde{v}_{y,norm}$', r'$\tilde{v}_{y,norm}$']])
        units = np.array([['$[-]$', '$[-]$', '$[-]$'],
                          ['$[-]$', '$[-]$', '$[-]$'],  
                          ['$[-]$', '$[-]$', '$[-]$']])
        
    elif transf == 't-inv':
        plotname = np.array([[r'$\varepsilon_{x,norm}$', r'$\varepsilon_{y,norm}$', r'$\varepsilon_{xy,norm}$'],
                        [r'$\chi_{x,norm}$', r'$\chi_{y,norm}$', r'$\chi_{xy,norm}$'],
                        [r'$\gamma_{x,norm}$', r'$\gamma_{y,norm}$', r'$t$']])
        plotname_p = np.array([[r'$\tilde{\varepsilon}_{x,norm}$', r'$\tilde{\varepsilon}_{y,norm}$', r'$\tilde{\varepsilon}_{xy,norm}$'],
                        [r'$\tilde{\chi}_{x,norm}$', r'$\tilde{\chi}_{y,norm}$', r'$\tilde{\chi}_{xy,norm}$'],
                        [r'$\tilde{\gamma}_{x,norm}$', r'$\tilde{\gamma}_{y,norm}$', r'$\tilde{t}$']])
        units = np.array([['$[-]$', '$[-]$', '$[-]$'],
                          ['$[-]$', '$[-]$', '$[-]$'],  
                          ['$[-]$', '$[-]$', '$[-]$']])
    
    elif transf == 'u-inv':
        plotname = np.array([[r'$\varepsilon_{x}$', r'$\varepsilon_{y}$', r'$\varepsilon_{xy}$'],
                        [r'$\chi_{x}$', r'$\chi_{y}$', r'$\chi_{xy}$'],
                        [r'$\gamma_{x}$', r'$\gamma_{y}$', r'$t$']])
        plotname_p = np.array([[r'$\tilde{\varepsilon}_{x}$', r'$\tilde{\varepsilon}_{y}$', r'$\tilde{\varepsilon}_{xy}$'],
                        [r'$\tilde{\chi}_{x}$', r'$\tilde{\chi}_{y}$', r'$\tilde{\chi}_{xy}$'],
                        [r'$\tilde{\gamma}_{x}$', r'$\tilde{\gamma}_{y}$', r'$\tilde{t}$']])
        units = np.array([[r'$\rm [-]$', r'$\rm [-]$', r'$\rm [-]$'],
                          [r'$\rm [1/mm]$', r'$\rm [1/mm]$', r'$\rm [1/mm]$'],  
                          [r'$\rm [-]$', r'$\rm [-]$', r'$\rm [-]$']])
        
    elif transf == 'o-inv': 
        plotname = np.array([[r'$\varepsilon_{x}$', r'$\varepsilon_{y}$', r'$\varepsilon_{xy}$'],
                        [r'$\chi_{x}$', r'$\chi_{y}$', r'$\chi_{xy}$'],
                        [r'$\gamma_{x}$', r'$\gamma_{y}$', r'$t$']])
        plotname_p = np.array([[r'$\tilde{\varepsilon}_{x}$', r'$\tilde{\varepsilon}_{y}$', r'$\tilde{\varepsilon}_{xy}$'],
                        [r'$\tilde{\chi}_{x}$', r'$\tilde{\chi}_{y}$', r'$\tilde{\chi}_{xy}$'],
                        [r'$\tilde{\gamma}_{x}$', r'$\tilde{\gamma}_{y}$', r'$\tilde{t}$']])
        units = np.array([[r'$\rm [-]$', r'$\rm [-]$', r'$\rm [-]$'],
                          [r'$\rm [1/cm]$', r'$\rm [1/cm]$', r'$\rm [1/cm]$'],  
                          [r'$\rm [-]$', r'$\rm [-]$', r'$\rm [-]$']])


    # find max, min for colorbars
    if norms_ is not None: 
        norms = norms_
    else: 
        norms = [mcolors.Normalize(vmin=np.min(errors[color][:, index_mask[i, :]]),
                           vmax=np.max(errors[color][:, index_mask[i, :]]))
                            for i in range(3)]
    scatters = []


    for i in range(3):
        for j in range(3):
            if i ==2 and j==2:
                    axa[i,j].set_title(' ')
            else: 
                if (xlim and ylim) is not None:
                    scatter = axa[i,j].scatter(Y[valid_rows,index_mask[i,j]], predictions[valid_rows,index_mask[i,j]], marker = 'o', s = 20, 
                                c = errors[color][:, index_mask[i,j]], cmap = 'plasma', linestyle='None', alpha = 0.4, 
                                norm = norms[i])
                    scatters.append(scatter)
                else:
                    scatter = axa[i,j].scatter(Y[:,index_mask[i,j]], predictions[:,index_mask[i,j]], marker = 'o', s = 20, 
                                c = errors[color][:, index_mask[i,j]], cmap = 'plasma', linestyle='None', alpha = 0.4, 
                                norm = norms[i])
                    scatters.append(scatter)
                if Y_train is not None:
                    scatter2 = axa[i,j].scatter(Y_train[:,index_mask[i,j]], pred_train[:,index_mask[i,j]], marker = 'o', 
                                  c = errors[color][:, index_mask[i,j]], cmap = 'viridis', fillstyle = 'none', 
                                  s = 8, linestyle='None', alpha = 0.4, norms = norms[i])
                axa[i,j].set_ylabel(plotname_p[i,j]+' '+ units[i,j])
                axa[i,j].set_xlabel(plotname[i,j]+' '+ units[i,j])
                
                if (xlim and ylim) is not None:
                    axa[i,j].set_xlim(xlim[0][index_mask[i,j]], xlim[1][index_mask[i,j]])
                    axa[i,j].set_ylim(ylim[0][index_mask[i,j]], ylim[1][index_mask[i,j]])
                else:
                    if Y_train is not None:
                        axa[i,j].set_xlim([np.min([np.min(Y_train[:,index_mask[i,j]]), np.min(pred_train[:,index_mask[i,j]])]), np.max([np.max(Y_train[:,index_mask[i,j]]), np.max(pred_train[:,index_mask[i,j]])])])
                        axa[i,j].set_ylim([np.min([np.min(Y_train[:,index_mask[i,j]]), np.min(pred_train[:,index_mask[i,j]])]), np.max([np.max(Y_train[:,index_mask[i,j]]), np.max(pred_train[:,index_mask[i,j]])])])
                    else:
                        axa[i,j].set_xlim([np.min([np.min(Y[:,index_mask[i,j]]), np.min(predictions[:,index_mask[i,j]])]), np.max([np.max(Y[:,index_mask[i,j]]), np.max(predictions[:,index_mask[i,j]])])])
                        axa[i,j].set_ylim([np.min([np.min(Y[:,index_mask[i,j]]), np.min(predictions[:,index_mask[i,j]])]), np.max([np.max(Y[:,index_mask[i,j]]), np.max(predictions[:,index_mask[i,j]])])])
                
                axa[i,j].grid(True, which='major', color='#666666', linestyle='-')
                at = AnchoredText('$R^2 = ' + np.array2string(errors['r_squared2'][:,index_mask[i,j]][0], precision=2) + '$ \n' +
                                  '$RMSE = ' + np.array2string(errors['rmse'][:,index_mask[i,j]][0], precision=2) + '$ \n' +
                                  '$rRMSE = ' + np.array2string(errors['rrmse'][:,index_mask[i,j]][0]*100, precision=0) + '\% $ \n' +
                                  '$||rRSE||_{\infty} = ' + np.array2string(errors['rrse_max'][:,index_mask[i,j]][0]*100, precision=0) + '\% $ \n'+
                                  '$nRMSE = ' + np.array2string(errors['nrmse'][:,index_mask[i,j]][0]*100, precision=0) + '\% $  \n'+
                                  '$||nRSE||_{\infty} = ' + np.array2string(errors['nrse_max'][:,index_mask[i,j]][0]*100, precision=0) + '\% $',
                                #    '$MALE = ' + np.array2string(mean_log_err[:,index_mask[i,j]][0], precision=2) +' $ \n'+
                                #   '$||ALE||_{\infty} = ' + np.array2string(log_max[:,index_mask[i,j]][0], precision=2)+ ' $',
                                #  '$||\Delta_{max}||_{\infty} = ' + np.array2string(Delta_max_max[0,index_mask[i,j]]*100, precision=0) + '\% $ \n' + 
                                #  '$\Delta_{max, avg} = ' + np.array2string(Delta_max_mean[0,index_mask[i,j]]*100, precision=0) + '\% $',
                                prop=dict(size=10), frameon=True,loc='upper left')
                at.patch.set_boxstyle('round,pad=0.,rounding_size=0.2')
                axa[i,j].add_artist(at)
                if Y_train is not None:
                    axa[i,j].plot([np.min([np.min(Y_train[:,index_mask[i,j]]), np.min(pred_train[:,index_mask[i,j]])]), np.max([np.max(Y_train[:,index_mask[i,j]]), np.max(pred_train[:,index_mask[i,j]])])], [np.min([np.min(Y_train[:,index_mask[i,j]]), np.min(pred_train[:,index_mask[i,j]])]), np.max([np.max(Y_train[:,index_mask[i,j]]), np.max(pred_train[:,index_mask[i,j]])])],
                                  color='white', linestyle='--', linewidth = 1)
                else:
                    axa[i,j].plot([np.min([np.min(Y[:,index_mask[i,j]]), np.min(predictions[:,index_mask[i,j]])]), np.max([np.max(Y[:,index_mask[i,j]]), np.max(predictions[:,index_mask[i,j]])])], [np.min([np.min(Y[:,index_mask[i,j]]), np.min(predictions[:,index_mask[i,j]])]), np.max([np.max(Y[:,index_mask[i,j]]), np.max(predictions[:,index_mask[i,j]])])],
                                  color='white', linestyle='--', linewidth = 1)
                for l in range(Y.shape[0]):
                    if np.all(mask_labels[l]):
                        axa[i,j].text(Y[l,index_mask[i,j]], predictions[l, index_mask[i,j]], str(l), fontsize=12, color='red', ha='center', va='bottom')
            
    axa[-1, -1].axis('off')
    
    at_ = AnchoredText('avg $rRMSE = ' + np.array2string(np.mean(errors['rrmse'][0,:])*100, precision=0) + '\% $ \n'+
                       'avg $||rRSE||_{\infty}= '+ np.array2string(np.mean(errors['rrse_max'][0,:])*100, precision=0) + '\% $\n'+
                       'avg $nRMSE = ' + np.array2string(np.mean(errors['nrmse'][0,:])*100, precision=0) + '\% $ \n'+
                       'avg $||nRSE||_{\infty}= '+ np.array2string(np.mean(errors['nrse_max'][0,:])*100, precision=0) + '\% $\n' +
                       ('N = '+ str(np.sum(valid_rows)) if (xlim and ylim) is not None else 'N = ' +str(Y.shape[0])), 
                       prop=dict(size=10), frameon=True, loc='center')
    at_.patch.set_boxstyle('round,pad=0.,rounding_size=0.2')
    axa[-1, -1].add_artist(at_)

    for i in range(3):
        if color == 'rse': 
            if i == 0 or i == 2:
                if 'inv' in transf:
                    name = 'RSE \: [-]'
                else: 
                    name = 'RSE \: [N/mm]'
            if i == 1:
                if 'inv' in transf: 
                    name = 'RSE \: [1/mm]'
                else: 
                    name = 'RSE \: [Nmm/mm]'
        elif color == 'nrse':
           name = 'nRSE \: [\%]'
        cbar = fig.colorbar(scatters[i*3], ax=axa[i,:], orientation='vertical', label=f'Row {i+1} $'+name+'$')
        cbar.set_label('$'+name+'$')


    axa = plt.gca()
    axa.set_aspect('equal', 'box')
    axa.axis('square')


    # Save figure
    # plt.tight_layout()
    if save_path is not None:
        if transf == 't' or transf == 't-inv':
            filename = os.path.join(save_path, 'diagonal_match_'+str(numit)+'transformed.png')
            plt.savefig(filename, dpi=100, bbox_inches='tight')
            # wandb.log({"45°-plot, t": wandb.Image(filename)})
            if "inv" not in transf: 
                print('saved sig-t-plots')
            else: 
                print('saved eps-t-plots')
        elif transf == 'o' or transf == 'o-inv':
            filename = os.path.join(save_path, 'diagonal_match_'+str(numit)+'original.png')
            plt.savefig(filename, dpi=100, bbox_inches='tight')
            # wandb.log({"45°-plot, og": wandb.Image(filename)})
            if "inv" not in transf:
                print('saved sig-o-plots')
            else: 
                print('saved eps-o-plots')
        elif transf == 'u' or transf == 'u-inv':
            if (xlim and ylim) is not None:
                filename = os.path.join(save_path, 'diagonal_match_'+str(numit)+'original_units_newlim.png')
            else: 
                filename = os.path.join(save_path, 'diagonal_match_'+str(numit)+'original_units.png')
            plt.savefig(filename, dpi=100, bbox_inches='tight')
            # wandb.log({"45°-plot, og_u": wandb.Image(filename)})
            if "inv" not in transf and (xlim and ylim) is None:
                print('saved sig-u-plots')
            elif (xlim and ylim) is not None:
                print('saved sig-u-plots with adjusted limits at ', save_path)
            else: 
                print('saved eps-u-plots')
    # plt.show()
    plt.close()

    return norms



def multiple_diagonal_plots_D(save_path: str, Y_inp: np.array, predictions_inp: np.array, transf:str, stats: dict, color:str, numit = 0, 
                              xlim = None, ylim = None, norms_ = None):
    '''
    displays all variables of D_m, D_mb, D_bm and D_b (6x6) + Ds (2x1) --> for nonlinear RC
    '''

    # plt.rcParams.update({
    #    "text.usetex": True,
    #    "font.family": "Helvetica",
    #    "font.size": 12,
    #    })
    

    plotname = np.array([['$D_{m,11}$', '$D_{m,12}$', '$D_{m,13}$', '$D_{mb,11}$', '$D_{mb,12}$', '$D_{mb,13}$'],
                        ['$D_{m,21}$', '$D_{m,22}$', '$D_{m,23}$', '$D_{mb,21}$', '$D_{mb,22}$', '$D_{mb,23}$'],
                        ['$D_{m,31}$', '$D_{m,32}$', '$D_{m,33}$', '$D_{mb,31}$', '$D_{mb,32}$', '$D_{mb,33}$'],
                        ['$D_{bm,11}$', '$D_{bm,12}$', '$D_{bm,13}$', '$D_{b,11}$', '$D_{b,12}$', '$D_{b,13}$'],
                        ['$D_{bm,21}$', '$D_{bm,22}$', '$D_{bm,23}$', '$D_{b,21}$', '$D_{b,22}$', '$D_{b,23}$'],
                        ['$D_{bm,31}$', '$D_{bm,32}$', '$D_{bm,33}$', '$D_{b,31}$', '$D_{b,32}$', '$D_{b,33}$'],
                        ['$D_{s,11}$', '$D_{s,22}$', '$D_{s,22}$', '$D_{s,22}$', '$D_{s,22}$', '$D_{s,22}$']
                        ])
    plotname_p = plotname

    if transf == 'o':
        units = np.array([[r'$\rm [MN/cm]$', r'$\rm [MN/cm]$', r'$\rm [MN/cm]$', r'$\rm [MN]$', r'$\rm [MN]$', r'$\rm [MN]$'],
                          [r'$\rm [MN/cm]$', r'$\rm [MN/cm]$', r'$\rm [MN/cm]$', r'$\rm [MN]$', r'$\rm [MN]$', r'$\rm [MN]$'],
                          [r'$\rm [MN/cm]$', r'$\rm [MN/cm]$', r'$\rm [MN/cm]$', r'$\rm [MN]$', r'$\rm [MN]$', r'$\rm [MN]$'],
                          [r'$\rm [MN]$', r'$\rm [MN]$', r'$\rm [MN]$', r'$\rm [MNcm]$', r'$\rm [MNcm]$', r'$\rm [MNcm]$'],
                          [r'$\rm [MN]$', r'$\rm [MN]$', r'$\rm [MN]$', r'$\rm [MNcm]$', r'$\rm [MNcm]$', r'$\rm [MNcm]$'],
                          [r'$\rm [MN]$', r'$\rm [MN]$', r'$\rm [MN]$', r'$\rm [MNcm]$', r'$\rm [MNcm]$', r'$\rm [MNcm]$'],
                          [r'$\rm [MN/cm]$', r'$\rm [MN/cm]$', r'$\rm [MN/cm]$', r'$\rm [MNcm]$', r'$\rm [MNcm]$', r'$\rm [MNcm]$']
                          ])
    elif transf =='u':
        units = np.array([[r'$\rm [N/mm]$', r'$\rm [N/mm]$', r'$\rm [N/mm]$', r'$\rm [N]$', r'$\rm [N]$', r'$\rm [N]$'],
                          [r'$\rm [N/mm]$', r'$\rm [N/mm]$', r'$\rm [N/mm]$', r'$\rm [N]$', r'$\rm [N]$', r'$\rm [N]$'],
                          [r'$\rm [N/mm]$', r'$\rm [N/mm]$', r'$\rm [N/mm]$', r'$\rm [N]$', r'$\rm [N]$', r'$\rm [N]$'],
                          [r'$\rm [N]$', r'$\rm [N]$', r'$\rm [N]$', r'$\rm [Nmm]$', r'$\rm [Nmm]$', r'$\rm [Nmm]$'],
                          [r'$\rm [N]$', r'$\rm [N]$', r'$\rm [N]$', r'$\rm [Nmm]$', r'$\rm [Nmm]$', r'$\rm [Nmm]$'],
                          [r'$\rm [N]$', r'$\rm [N]$', r'$\rm [N]$', r'$\rm [Nmm]$', r'$\rm [Nmm]$', r'$\rm [Nmm]$'],
                          [r'$\rm [N/mm]$', r'$\rm [N/mm]$', r'$\rm [N/mm]$', r'$\rm [Nmm]$', r'$\rm [Nmm]$', r'$\rm [Nmm]$']
                          ])

    index_mask = np.array([[ 0,  1,  2,  3,  4,  5],
                           [ 6,  7,  8,  9, 10, 11],
                           [12, 13, 14, 15, 16, 17],
                           [18, 19, 20, 21, 22, 23],
                           [24, 25, 26, 27, 28, 29],
                           [30, 31, 32, 33, 34, 35],
                           [36, 37, 37, 37, 37, 37]
                           ])
    
    
    Y = np.concatenate((Y_inp[:,:6, :6].reshape((-1,36)), Y_inp[:,6,6].reshape((-1,1)), Y_inp[:,7,7].reshape((-1,1))), axis = 1)
    predictions = np.concatenate((predictions_inp[:,:6,:6].reshape((-1,36)), predictions_inp[:,6,6].reshape((-1,1)), predictions_inp[:,7,7].reshape((-1,1))), axis = 1)

    if (xlim and ylim) is not None:
        mask_x = (Y[:,:]>=xlim[0]) & (Y[:,:]<=xlim[1])
        mask_y = (predictions[:,:]>=ylim[0]) & (predictions[:,:]<=ylim[1])
        mask = mask_x & mask_y
        valid_rows = mask.all(axis=1)
        if np.sum(valid_rows) == 0:
            raise Warning('No points found for this region. Please increase range.')
        errors = calculate_errors(Y[valid_rows,:], predictions[valid_rows,:], stats, transf, id = 'De-NLRC')
    else: 
        errors = calculate_errors(Y, predictions, stats, transf, id = 'De-NLRC')

    #norms = [mcolors.Normalize(vmin=np.min(errors[color][:, index_mask[i, :]]),
    #                       vmax=np.max(errors[color][:, index_mask[i, :]]))
    #                         for i in range(7)]
    
    block_positions = [(0, 0), (0, 3), (3, 0), (3, 3)]
    norms = []

    for bx, by in block_positions:
        block_errors = np.concatenate([
            errors[color][:, index_mask[b0, b1]].flatten()
            for b0 in range(bx, bx + 3) for b1 in range(by, by + 3)
            ])
        norms.append(mcolors.Normalize(vmin=np.min(block_errors), vmax=np.max(block_errors)))

    # Last row special case
    last_row_errors = np.concatenate([errors[color][:, index_mask[6, j]].flatten() for j in range(3)])
    vmin, vmax = np.min(last_row_errors), np.max(last_row_errors)
    norm_last_row = mcolors.Normalize(vmin=vmin, vmax=vmax)
    norms.extend([norm_last_row] * 3)


    exp_rmse = np.floor(np.log10(errors['rmse']+1)).astype(int)
    base_rmse = errors['rmse']/(10**exp_rmse)
    scatters = []
    

    fig, axa = plt.subplots(7, 6, figsize=[35, 30], dpi=100)
    fig.subplots_adjust(wspace=0.5)


    for i in range(7):
        for j in range(6):
            if i == 6 and j > 1:
                axa[i,j].set_title('  ')
            else: 
                if (xlim and ylim) is not None:
                    scatter = axa[i,j].scatter(Y[valid_rows,index_mask[i,j]], predictions[valid_rows,index_mask[i,j]], marker = 'o', s = 20, 
                                c = errors[color][:, index_mask[i,j]], cmap = 'plasma', linestyle='None', alpha = 1, 
                                norm = norms[i // 3 * 2 + j // 3])
                    scatters.append(scatter)
                else:
                    scatter = axa[i,j].scatter(Y[:,index_mask[i,j]], predictions[:,index_mask[i,j]], marker = 'o', s = 20, 
                                            c = errors[color][:, index_mask[i,j]], cmap = 'plasma', linestyle='None', alpha = 1, 
                                            norm = norms[i // 3 * 2 + j // 3])
                    scatters.append(scatter)
                # axa[i,j].plot(Y[:,index_mask[i,j]], predictions[:,index_mask[i,j]], marker = 'o', ms = 5, linestyle='None')
                axa[i,j].set_ylabel(plotname_p[i,j]+' '+ units[i,j])
                axa[i,j].set_xlabel(plotname[i,j]+' '+ units[i,j])
                if (xlim and ylim) is not None:
                    axa[i,j].set_xlim(xlim[0][index_mask[i,j]], ylim[1][index_mask[i,j]])
                    axa[i,j].set_ylim(xlim[0][index_mask[i,j]], ylim[1][index_mask[i,j]])
                else:
                    pass
                    # axa[i,j].set_xlim([np.min([np.min(Y[:,index_mask[i,j]]), np.min(predictions[:,index_mask[i,j]])]), np.max([np.max(Y[:,index_mask[i,j]]), np.max(predictions[:,index_mask[i,j]])])])
                    # axa[i,j].set_ylim([np.min([np.min(Y[:,index_mask[i,j]]), np.min(predictions[:,index_mask[i,j]])]), np.max([np.max(Y[:,index_mask[i,j]]), np.max(predictions[:,index_mask[i,j]])])])
                axa[i,j].grid(True, which='major', color='#666666', linestyle='-')
                at = AnchoredText('$R^2 = ' + np.array2string(errors['r_squared2'][:,index_mask[i,j]][0], precision=3) + '$ \n' +
                                    '$RMSE = ' + np.array2string(base_rmse[:,index_mask[i,j]][0], precision=2)+ '\\times 10^{'+ np.array2string(exp_rmse[:,index_mask[i,j]][0], precision=0) +'}$ \n' +
                                    '$rRMSE = ' + np.array2string(errors['rrmse'][:,index_mask[i,j]][0]*100, precision=0) + '\% $ \n'+
                                    # '$||rRSE||_{\infty} = ' + np.array2string(errors['rrse_max'][:,index_mask[i,j]][0]*100, precision=0) + '\% $ \n'+
                                    '$nRMSE = ' + np.array2string(errors['nrmse'][:,index_mask[i,j]][0]*100, precision=0) + '\% $',
                                    # '$||nRSE||_{\infty} = ' + np.array2string(errors['nrse_max'][:,index_mask[i,j]][0]*100, precision=0) + '\% $',
                                prop=dict(size=10), frameon=True, loc='upper right') #upper left
                at.patch.set_boxstyle('round,pad=0.,rounding_size=0.2')
                axa[i,j].add_artist(at)
                axa[i,j].plot([np.min([np.min(Y[:,index_mask[i,j]]), np.min(predictions[:,index_mask[i,j]])]), np.max([np.max(Y[:,index_mask[i,j]]), np.max(predictions[:,index_mask[i,j]])])], [np.min([np.min(Y[:,index_mask[i,j]]), np.min(predictions[:,index_mask[i,j]])]), np.max([np.max(Y[:,index_mask[i,j]]), np.max(predictions[:,index_mask[i,j]])])], 
                                color='grey', linestyle='--', linewidth = 1)
                axa[i,j].set_aspect('equal', 'box')
    
    axa[-1, -1].axis('off')
    axa[-1, -2].axis('off')
    axa[-1, -3].axis('off')
    axa[-1, -4].axis('off')


    at_ = AnchoredText('avg $rRMSE = ' + np.array2string(np.mean(errors['rrmse'][0,:])*100, precision=0) + '\% $ \n'+
                       'avg $||rRSE||_{\infty}= '+ np.array2string(np.mean(errors['rrse_max'][0,:])*100, precision=0) + '\% $\n'+
                       'avg $nRMSE = ' + np.array2string(np.mean(errors['nrmse'][0,:])*100, precision=0) + '\% $ \n'+
                       'avg $||nRSE||_{\infty}= '+ np.array2string(np.mean(errors['nrse_max'][0,:])*100, precision=0) + '\% $\n' +
                       ('N = '+ str(np.sum(valid_rows)) if (xlim and ylim) is not None else 'N = ' +str(Y.shape[0])),
                        prop=dict(size=10), frameon=True, loc='center')
    at_.patch.set_boxstyle('round,pad=0.,rounding_size=0.2')
    axa[-1, -1].add_artist(at_)
    
    # Define 3x3 blocks for colorbars
    colorbar_labels = {"rse": ["RSE [N/mm]", "RSE [N]", "RSE [N]", "RSE [Nmm]"], "nrse":  ["nRSE [%]", "nRSE [%]", "nRSE [%]", "nRSE [%]"]}
    colorbar_label = colorbar_labels.get(color, "Error Scale")

    for norm, (bx, by), label in zip(norms, block_positions, colorbar_label):
        sm = plt.cm.ScalarMappable(cmap='plasma', norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axa[bx:bx+3, by:by+3], orientation='vertical', fraction=0.02, pad=0.04)
        cbar.set_label(label, fontsize=12)

    # Last row special case
    sm = plt.cm.ScalarMappable(cmap='plasma', norm=norms[-1])
    sm.set_array([])
    cbar_last_row = fig.colorbar(sm, ax=axa[6, :3], orientation='vertical', fraction=0.02, pad=0.04)
    cbar_last_row.set_label("RSE [N/mm]", fontsize=12)

    axa = plt.gca()
    axa.axis('square')


    # Save figure
    if save_path is not None and transf == 'o':
        filename = os.path.join(save_path, 'diagonal_match_'+str(numit)+'D_nonzero_o.png')
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        # wandb.log({"45°-plot, D_o": wandb.Image(filename)})
        print(f'saved D-o-plot at {save_path}')
    if save_path is not None and transf == 'u':
        if (xlim and ylim) is not None:
            filename = os.path.join(save_path, 'diagonal_match_'+str(numit)+'D_nonzero_u_newlim.png')
        else:
            filename = os.path.join(save_path, 'diagonal_match_'+str(numit)+'D_nonzero_u.png')
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        # wandb.log({"45°-plot, D_u": wandb.Image(filename)})
        if (xlim and ylim) is not None: 
            print(f'saved D-u-plot with adjusted limits at {save_path}')
        else:
            print(f'saved D-u-plot at {save_path}')
    # plt.show()
    plt.close()


def multiple_diagonal_plots_Dnz(save_path: str, Y_inp: np.array, predictions_inp: np.array, transf:str, stats: dict, color:str):
    '''
    displays all variables which are not zero in a linear elastic stiffness matrix 
    --> use in case of glass or lin.el. material training / testing
    '''

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Helvetica",
        "font.size": 12,
        })


    # Kick out irrelevant data (that should be zero) and reshape matrix to (num.rows x 12) format
    mat_comp_m = {
        "Simulation": np.hstack(((Y_inp[:, 0:2, 0:2]).reshape((Y_inp.shape[0],4)), Y_inp[:,2,2].reshape((Y_inp.shape[0],1)))),
        "Prediction": np.hstack(((predictions_inp[:,0:2, 0:2]).reshape((Y_inp.shape[0],4)), predictions_inp[:,2,2].reshape((Y_inp.shape[0],1))))
        }
    mat_comp_b = {
        "Simulation": np.hstack(((Y_inp[:,3:5, 3:5]).reshape((Y_inp.shape[0],4)), Y_inp[:,5,5].reshape((Y_inp.shape[0],1)))),
        "Prediction": np.hstack(((predictions_inp[:,3:5, 3:5]).reshape((Y_inp.shape[0],4)), predictions_inp[:,5,5].reshape((Y_inp.shape[0],1))))
        }
    
    mat_comp_mb = {
        "Simulation": np.hstack(((Y_inp[:,0:2, 3:5]).reshape((Y_inp.shape[0],4)), Y_inp[:,2,5].reshape((Y_inp.shape[0],1)))),
        "Prediction": np.hstack(((predictions_inp[:,0:2, 3:5]).reshape((Y_inp.shape[0],4)), predictions_inp[:,2,5].reshape((Y_inp.shape[0],1))))
    }

    mat_comp_s = {
        "Simulation": np.hstack((Y_inp[:,6,6], Y_inp[:,7,7])).reshape((Y_inp.shape[0],2)),
        "Prediction": np.hstack((predictions_inp[:,6,6], predictions_inp[:,7,7])).reshape((Y_inp.shape[0],2))
        }
    
    Y = np.hstack((mat_comp_m['Simulation'], mat_comp_b['Simulation'], mat_comp_mb['Simulation'], mat_comp_s['Simulation']))
    predictions = np.hstack((mat_comp_m['Prediction'], mat_comp_b['Prediction'], mat_comp_mb['Prediction'], mat_comp_s['Prediction']))

    errors = calculate_errors(Y, predictions, stats, transf, id = 'De')
    exp_rmse = np.floor(np.log10(errors['rmse']+1)).astype(int)
    base_rmse = errors['rmse']/(10**exp_rmse)
    
    # Plot figure
    fig, axa = plt.subplots(4, 5, figsize=[20, 12], dpi=100)
    fig.subplots_adjust(wspace=0.5)
    
    num_rows = Y.shape[0]

    plotname = np.array([['$D_{m,11}$', '$D_{m,12}$', '$D_{m,21}$', '$D_{m,22}$', '$D_{m,33}$'],
                        ['$D_{b,11}$', '$D_{b,12}$', '$D_{b,21}$', '$D_{b,22}$', '$D_{b,33}$'],
                        ['$D_{mb,11}$', '$D_{mb,12}$', '$D_{mb,21}$', '$D_{mb,22}$', '$D_{mb,33}$'],
                        ['$D_{s,11}$', '$D_{s,22}$', '$D_{s,22}$', '$D_{s,22}$', '$D_{s,22}$']])
    plotname_p = np.array([[r'$\tilde{D}_{m,11}$', r'$\tilde{D}_{m,12}$', r'$\tilde{D}_{m,21}$', r'$\tilde{D}_{m,22}$', r'$\tilde{D}_{m,33}$'],
                        [r'$\tilde{D}_{b,11}$', r'$\tilde{D}_{b,12}$', r'$\tilde{D}_{b,21}$', r'$\tilde{D}_{b,22}$', r'$\tilde{D}_{b,33}$'],
                        [r'$\tilde{D}_{mb,11}$', r'$\tilde{D}_{mb,12}$', r'$\tilde{D}_{mb,21}$', r'$\tilde{D}_{mb,22}$', r'$\tilde{D}_{mb,33}$'],
                        [r'$\tilde{D}_{s,11}$', r'$\tilde{D}_{s,22}$', r'$\tilde{D}_{s,22}$', r'$\tilde{D}_{s,22}$', r'$\tilde{D}_{s,22}$']])
    
    if transf == 'o':
        units = np.array([[r'$\rm [MN/cm]$', r'$\rm [MN/cm]$', r'$\rm [MN/cm]$', r'$\rm [MN/cm]$', r'$\rm [MN/cm]$'],
                            [r'$\rm [MNcm]$', r'$\rm [MNcm]$', r'$\rm [MNcm]$', r'$\rm [MNcm]$', r'$\rm [MNcm]$'],
                            [r'$\rm [MN]$', r'$\rm [MN]$', r'$\rm [MN]$', r'$\rm [MN]$', r'$\rm [MN]$'],
                            [r'$\rm [MN/cm]$', r'$\rm [MN/cm]$', r'$\rm [MN/cm]$', r'$\rm [MN/cm]$', r'$\rm [MN/cm]$']])
    elif transf =='u':
        units = np.array([[r'$\rm [N/mm]$', r'$\rm [N/mm]$', r'$\rm [N/mm]$', r'$\rm [N/mm]$', r'$\rm [N/mm]$'],
                            [r'$\rm [Nmm]$', r'$\rm [Nmm]$', r'$\rm [Nmm]$', r'$\rm [Nmm]$', r'$\rm [Nmm]$'],
                            [r'$\rm [N]$', r'$\rm [N]$', r'$\rm [N]$', r'$\rm [N]$', r'$\rm [N]$'],
                            [r'$\rm [N/mm]$', r'$\rm [N/mm]$', r'$\rm [N/mm]$', r'$\rm [N/mm]$', r'$\rm [N/mm]$']])
        
    index_mask = np.array([[0,  1,  2,  3,  4],
                           [5,  6,  7,  8,  9],
                           [10, 11, 12, 13, 14],
                           [15, 16, 16, 16, 16]])


    # find max, min for colorbars
    norms = [mcolors.Normalize(vmin=np.min(errors[color][:, index_mask[i, :]]),
                           vmax=np.max(errors[color][:, index_mask[i, :]]))
                            for i in range(4)]
    scatters = []


    for i in range(4):
        for j in range(5):
            if i == 3 and j > 1:
                axa[i,j].set_title('  ')
            else:
                scatter = axa[i,j].scatter(Y[:,index_mask[i,j]], predictions[:,index_mask[i,j]], marker = 'o', s = 20, 
                                            c = errors[color][:, index_mask[i,j]], cmap = 'plasma', linestyle='None', alpha = 1, 
                                            norm = norms[i])
                scatters.append(scatter)
                # axa[i,j].plot(Y[:,index_mask[i,j]], predictions[:,index_mask[i,j]], marker = 'o', ms = 5, linestyle='None')
                axa[i,j].set_ylabel(plotname_p[i,j]+' '+ units[i,j])
                axa[i,j].set_xlabel(plotname[i,j]+' '+ units[i,j])
                axa[i,j].set_xlim([np.min([np.min(Y[:,index_mask[i,j]]), np.min(predictions[:,index_mask[i,j]])]), np.max([np.max(Y[:,index_mask[i,j]]), np.max(predictions[:,index_mask[i,j]])])])
                axa[i,j].set_ylim([np.min([np.min(Y[:,index_mask[i,j]]), np.min(predictions[:,index_mask[i,j]])]), np.max([np.max(Y[:,index_mask[i,j]]), np.max(predictions[:,index_mask[i,j]])])])
                axa[i,j].grid(True, which='major', color='#666666', linestyle='-')
                at = AnchoredText('$R^2 = ' + np.array2string(errors['r_squared2'][:,index_mask[i,j]][0], precision=3) + '$ \n' +
                                  '$RMSE = ' + np.array2string(base_rmse[:,index_mask[i,j]][0], precision=2)+ '\\times 10^{'+ np.array2string(exp_rmse[:,index_mask[i,j]][0], precision=0) +'}$ \n' +
                                  '$rRMSE = ' + np.array2string(errors['rrmse'][:,index_mask[i,j]][0]*100, precision=0) + '\% $ \n'+
                                  # '$||rRSE||_{\infty} = ' + np.array2string(errors['rrse_max'][:,index_mask[i,j]][0]*100, precision=0) + '\% $ \n'+
                                  '$nRMSE = ' + np.array2string(errors['nrmse'][:,index_mask[i,j]][0]*100, precision=0) + '\% $',
                                  # '$||nRSE||_{\infty} = ' + np.array2string(errors['nrse_max'][:,index_mask[i,j]][0]*100, precision=0) + '\% $',
                                prop=dict(size=10), frameon=True, loc='lower right')
                at.patch.set_boxstyle('round,pad=0.,rounding_size=0.2')
                axa[i,j].add_artist(at)
                axa[i,j].plot([np.min([np.min(Y[:,index_mask[i,j]]), np.min(predictions[:,index_mask[i,j]])]), np.max([np.max(Y[:,index_mask[i,j]]), np.max(predictions[:,index_mask[i,j]])])], [np.min([np.min(Y[:,index_mask[i,j]]), np.min(predictions[:,index_mask[i,j]])]), np.max([np.max(Y[:,index_mask[i,j]]), np.max(predictions[:,index_mask[i,j]])])], 
                              color='grey', linestyle='--', linewidth = 1)
                axa[i,j].set_aspect('equal', 'box')
    
    axa[-1, -1].axis('off')
    axa[-1, -2].axis('off')
    axa[-1, -3].axis('off')
    at_ = AnchoredText('avg $rRMSE = ' + np.array2string(np.mean(errors['rrmse'][0,:])*100, precision=0) + '\% $ \n'+
                       'avg $||rRSE||_{\infty}= '+ np.array2string(np.mean(errors['rrse_max'][0,:])*100, precision=0) + '\% $\n'+
                       'avg $nRMSE = ' + np.array2string(np.mean(errors['nrmse'][0,:])*100, precision=0) + '\% $ \n'+
                       'avg $||nRSE||_{\infty}= '+ np.array2string(np.mean(errors['nrse_max'][0,:])*100, precision=0) + '\% $',
                        prop=dict(size=10), frameon=True, loc='center')
    at_.patch.set_boxstyle('round,pad=0.,rounding_size=0.2')
    axa[-1, -1].add_artist(at_)

    for i in range(4):
        if color == 'rse': 
            if i == 0 or i == 3:
                name = 'RSE \: [N/mm]'
            if i == 1:
                name = 'RSE \: [Nmm]'
            if i ==2: 
                name = 'RSE \: [N]'
        elif color == 'nrse':
           name = 'nRSE \: [\%]'
        cbar = fig.colorbar(scatters[i*3], ax=axa[i,:], orientation='vertical', label=f'Row {i+1} $'+name+'$')
        cbar.set_label('$'+name+'$')

    axa = plt.gca()
    axa.axis('square')


    # Save figure
    if save_path is not None and transf == 'o':
        filename = os.path.join(save_path, 'diagonal_match_'+'D_nonzero_o.png')
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        # wandb.log({"45°-plot, D_o": wandb.Image(filename)})
        print('saved D-o-plot')
    if save_path is not None and transf == 'u':
        filename = os.path.join(save_path, 'diagonal_match_'+'D_nonzero_u.png')
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        # wandb.log({"45°-plot, D_u": wandb.Image(filename)})
        print('saved D-u-plot')
    # plt.show()
    plt.close()
    return



def multiple_diagonal_plots_paper(save_path: str, Y: np.array, predictions: np.array, transf:str, stats:dict, color='nrse', Y_train = None, pred_train = None):
    ''''
    save_path       (str)           path where images are saved
    Y               (np.array)      Ground truth
    predictions     (np.array)      Predictions
    transf          (str)           't': transformed(normalised), 'o': original scale [MN, cm], 'u': units for simulation [N,mm]
    stats           (dict)          data statistics for normalising / relativising the RMSE
    color           (str)           scatter color: if 'nrse': only one colour bar across all plots. If 'rse': 3 separate colorbars for n, m, v
    '''

    # for the paper convert the units to kN/m and kNm/m
    Y[:,3:6] = Y[:,3:6]*10**(-3)
    predictions[:,3:6] = predictions[:,3:6]*10**(-3)

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Times New Roman",
        "font.size": 9,
        # "legend.fontsize": 5,
        })

    plt.rc('text.latex', preamble=
        r'\usepackage{amsmath}' + 
        r'\usepackage{times}')

    errors = calculate_errors(Y, predictions, stats, transf, id = 'sig')

    # Plot figure
    fig = plt.figure(figsize=(16/2.54, 12/2.54), dpi=300)
    axa = fig.subplots(3, 3)
    fig.subplots_adjust(wspace=0.6)
    fig.subplots_adjust(hspace=0.4)

    index_mask = np.array([[0,1,2],
                           [3,4,5],
                           [6,7,7]])


    if transf == 'u':
        plotname = np.array([[r'\textit{n}$_\textit{x}$', r'\textit{n}$_\textit{y}$', r'\textit{n}$_{\textit{xy}}$'],
                            [r'\textit{m}$_\textit{x}$', r'\textit{m}$_\textit{y}$', r'\textit{m}$_{\textit{xy}}$'],
                            [r'\textit{v}$_{\textit{xz}}$', r'\textit{v}$_{\textit{yz}}$', '$v_\textit{y}$']])
        # plotname_p = np.array([[r'$\tilde{n}_{x}$', r'$\tilde{n}_{y}$', r'$\tilde{n}_{xy}$'],
         #                    [r'$\tilde{m}_x$', r'$\tilde{m}_y$', r'$\tilde{m}_{xy}$'],
         #                   [r'$\tilde{v}_x$', r'$\tilde{v}_y$', r'$\tilde{v}_y$']])
        plotname_p = np.array([[r'\textit{n}$_{\textit{x,pred}}$', r'\textit{n}$_{\textit{y,pred}}$', r'\textit{n}$_{\textit{xy,pred}}$'],
                            [r'\textit{m}$_{\textit{x,pred}}$', r'\textit{m}$_{\textit{y,pred}}$', r'\textit{m}$_{\textit{xy,pred}}$'],
                            [r'\textit{v}$_{\textit{xz,pred}}$', r'\textit{v}$_{\textit{yz,pred}}$', r'\textit{v}$_{\textit{yz,pred}}$']])
        units = np.array([[r'[kN/m]', r'[kN/m]', r'[kN/m]'],
                          [r'[kNm/m]', r'[kNm/m]', r'[kNm/m]'],  
                          [r'[kN/m]', r'[kN/m]', r'[kN/m]']])


    # find max, min for colorbars
    norms = [mcolors.Normalize(vmin=np.min(errors[color][:, index_mask[i, :]]),
                           vmax=np.max(errors[color][:, index_mask[i, :]]))
                            for i in range(3)]
    scatters = []


    for i in range(3):
        for j in range(3):
            # formatting axes
            for spine in axa[i,j].spines.values():
                spine.set_linewidth(0.5)  # Set the width of the outline
                spine.set_color('black')   # Set the color of the outline
            axa[i,j].tick_params(axis='both', labelsize=4, length=2, width=0.25, color = 'black', labelcolor = 'black')

            if i ==2 and j==2:
                    axa[i,j].set_title(' ')
            else: 
                scatter = axa[i,j].scatter(Y[:,index_mask[i,j]], predictions[:,index_mask[i,j]], marker = 'o', s = 5, 
                              c = errors[color][:, index_mask[i,j]], cmap = 'plasma', linestyle='None', alpha = 1, 
                              norm = norms[i])
                scatters.append(scatter)
                # Only plot three ticks per axis (min, max, mean)
                def round_to(x, round_by):
                    if x >= 0: 
                        return round(x-(x % round_by))
                    else:
                        return round(x-(x % round_by) + round_by)
                
                y_val = [round_to(num, 500) for num in axa[i,j].get_ylim()]
                x_val = [round_to(num, 500) for num in axa[i,j].get_xlim()]     #[min, max]
                y_val.append(round((y_val[0]+y_val[1])/2))
                x_val.append(round((x_val[0]+x_val[1])/2))

                axa[i,j].set_yticks([y_val[0], y_val[2], y_val[1]])
                axa[i,j].set_yticklabels([y_val[0], y_val[2], y_val[1]])
                axa[i,j].set_xticks([x_val[0], x_val[2], x_val[1]])
                axa[i,j].set_xticklabels([x_val[0], x_val[2], x_val[1]])

                if Y_train is not None:
                    print('The paper version of the test plot is not thought for plotting training predictions')
                
                # Labelling of axes
                axa[i,j].set_ylabel(plotname_p[i,j]+' '+ units[i,j])
                axa[i,j].set_xlabel(plotname[i,j]+' '+ units[i,j])
                # axa[i,j].set_xlim([np.min([np.min(Y[:,index_mask[i,j]]), np.min(predictions[:,index_mask[i,j]])]), np.max([np.max(Y[:,index_mask[i,j]]), np.max(predictions[:,index_mask[i,j]])])])
                # axa[i,j].set_ylim([np.min([np.min(Y[:,index_mask[i,j]]), np.min(predictions[:,index_mask[i,j]])]), np.max([np.max(Y[:,index_mask[i,j]]), np.max(predictions[:,index_mask[i,j]])])])
                axa[i,j].grid(True, which='major', color='lightgrey', linestyle='-', linewidth=0.25)
                at = AnchoredText(r'$||$\textit{RSE}$||_{\infty}$ = ' + np.array2string(errors['rse_max'][:,index_mask[i,j]][0], precision=1) + ' \n' +
                                  r'\textit{RMSE} = ' + np.array2string(errors['rmse'][:,index_mask[i,j]][0], precision=2) + ' \n' +
                                  r'\textit{rRMSE} = ' + np.array2string(errors['rrmse'][:,index_mask[i,j]][0]*100, precision=1) + '\% \n' +
                                  r'\textit{nRMSE} = ' + np.array2string(errors['nrmse'][:,index_mask[i,j]][0]*100, precision=1) + '\%',
                                prop=dict(size=5), 
                                frameon=True, loc='upper left')
                at.patch.set_edgecolor('lightgrey')
                at.patch.set_boxstyle('round,pad=0.,rounding_size=0.1')
                at.patch.set_linewidth(0.5)
                axa[i,j].add_artist(at)
                # Plot the 45°-dashed line
                axa[i,j].plot([np.min([np.min(Y[:,index_mask[i,j]]), np.min(predictions[:,index_mask[i,j]])]), 
                               np.max([np.max(Y[:,index_mask[i,j]]), np.max(predictions[:,index_mask[i,j]])])], 
                               [np.min([np.min(Y[:,index_mask[i,j]]), np.min(predictions[:,index_mask[i,j]])]), 
                                np.max([np.max(Y[:,index_mask[i,j]]), np.max(predictions[:,index_mask[i,j]])])],
                                color='lightgrey', linestyle='--', linewidth = 1)
            
    axa[-1, -1].axis('off')
    at_ = AnchoredText(r'avg \textit{rRMSE} = ' + np.array2string(np.mean(errors['rrmse'][0,:])*100, precision=1) + '\% \n'+
                       r'avg $||$\textit{rRSE}$||_{\infty}$= '+ np.array2string(np.mean(errors['rrse_max'][0,:])*100, precision=1) + '\% \n'+
                       r'avg \textit{nRMSE} = ' + np.array2string(np.mean(errors['nrmse'][0,:])*100, precision=1) + '\% \n'+
                       r'avg $||$\textit{nRSE}$||_{\infty}$= '+ np.array2string(np.mean(errors['nrse_max'][0,:])*100, precision=1) + '\%',
                        prop=dict(size=7), 
                        frameon=True,loc='center')
    at_.patch.set_edgecolor('lightgrey')
    at_.patch.set_boxstyle('round,pad=0.,rounding_size=0.1')
    at_.patch.set_linewidth(0.5)
    axa[-1, -1].add_artist(at_)

    for i in range(3):
        if color == 'rse': 
            if i == 0 or i == 2:
                name = r'\textit{RSE}'
                unit = '[kN/m]'
            if i == 1:
                name = r'\textit{RSE}'
                unit = '[kNm/m]'
        elif color == 'nrse':
           name = 'nRSE \: [\%]'
        cbar = fig.colorbar(scatters[i*3], ax=axa[i,:], orientation='vertical', label=f'Row {i+1}'+name+' '+unit)
        cbar.set_label('$'+name+'$'+' '+unit)
        cbar.ax.tick_params(width=0.5, labelsize=5)
        cbar.outline.set_linewidth(0.5)

    # axa = plt.gca()
    # axa.set_aspect('equal', 'box')
    # axa.axis('square')


    # Save figure
    if save_path is not None:
        if transf == 'u':
            # filename = os.path.join(save_path, 'diagonal_match_'+'original_units_paper.png')
            filename = os.path.join(save_path, 'diagonal_match_'+'original_units_paper.tif')
            plt.savefig(filename, dpi = 600)
            wandb.log({"45°-plot, og_u": wandb.Image(filename)})
            print('saved sig-u-plots-paper')


    return



'''----------------------------------VERSION SAVING / COPYING------------------------------------------------------------'''



def get_latest_version_folder(base_folder):
        """
        Finds the latest version folder in the base folder with the format 'v_num'.
        """
        version_folders = [f for f in os.listdir(base_folder) if re.match(r'v_\d+', f)]
        version_numbers = [int(re.search(r'v_(\d+)', folder).group(1)) for folder in version_folders]
        return max(version_numbers) if version_numbers else 0


def copy_files_with_incremented_version(src_folder, base_dest_folder, files_to_copy):
    """
    Copies files from src_folder to a new folder in base_dest_folder with an incremented version number.
    """
    # Get the latest version number in the destination folder and increment it
    latest_version = get_latest_version_folder(base_dest_folder)
    new_version = latest_version + 1
    new_folder_name = f"v_{new_version}"
    new_folder_path = os.path.join(base_dest_folder, new_folder_name)
    
    # Create the new versioned folder
    os.makedirs(new_folder_path, exist_ok=True)
    
    # Copy files from the source folder to the new versioned folder
    for file_name in files_to_copy:
        src_path = os.path.join(src_folder, file_name)
        dest_path = os.path.join(new_folder_path, file_name)   

        if file_name == 'best_trained_model.pt':
            model_file = glob.glob(os.path.join(src_folder, 'best_trained_model_*.pt'))
            for i in range(len(model_file)):
                file_name_new = os.path.basename(model_file[i])
                src_path_new = os.path.join(src_folder, file_name_new)
                dest_path_new = os.path.join(new_folder_path, file_name_new)
                if os.path.exists(src_path_new):
                    shutil.copy2(src_path_new, dest_path_new)
                    os.remove(src_path_new)
                else: 
                    print(f"{file_name} does not exist in the source folder.")

        else: 
            if os.path.exists(src_path):
                shutil.copy2(src_path, dest_path)
            else:
                print(f"{file_name} does not exist in the source folder.")

    
    print(f"Files copied to {new_folder_path}")


def copy_files_to_plots_folder(src_folder, dest_folder, files_to_copy):
    """
    Copies specified files from src_folder to a new 'plots' folder within dest_folder.
    """
    # Create the 'plots' folder within the destination folder
    plots_folder = os.path.join(dest_folder, 'plots')
    os.makedirs(plots_folder, exist_ok=True)
    
    # Copy only the specified files to the 'plots' folder
    for file_name in files_to_copy:
        src_path = os.path.join(src_folder, file_name)
        dest_path = os.path.join(plots_folder, file_name)
        
        if os.path.exists(src_path):  # Ensure the file exists in the source folder
            shutil.copy2(src_path, dest_path)
        else:
            print(f"{file_name} does not exist in the source folder.")
    
    print(f"Specified files copied to {plots_folder}")


def create_vid(src_folder, dest_file, fps=30):
    # Get sorted image files
    images = sorted([img for img in os.listdir(src_folder) 
                     if "_newlim" in img and img.endswith((".png", ".jpg", ".jpeg"))])

    if not images:
        print("No images found in the source folder.")
        return

    # Read first image to get frame size
    first_image_path = os.path.join(src_folder, images[0])
    first_image = cv2.imread(first_image_path)
    if first_image is None:
        print("Error reading the first image.")
        return
    
    height, width, _ = first_image.shape

    hold_time_seconds = 2  # Show each image for 2 seconds
    frame_repeat = int(fps * hold_time_seconds)

    # Define video writer
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    video = cv2.VideoWriter(dest_file, fourcc, fps, (width, height))

    

    # Add images to video
    total_frames_written = 0
    for image in images:
        img_path = os.path.join(src_folder, image)
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"Skipping {img_path}, could not read the file.")
            continue
        _ = 0
        for _ in range(frame_repeat):
           if _ == 0: 
               print(img_path)
               plt.imshow(frame)
               plt.show()
           video.write(frame)
           total_frames_written += 1



    # Release the video writer
    video.release()
    cv2.destroyAllWindows()
    print(f"Video saved as {dest_file}")
    print('total frames written', total_frames_written)


    cap = cv2.VideoCapture(dest_file)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_reported = cap.get(cv2.CAP_PROP_FPS)
    duration = frame_count / fps_reported if fps_reported else 0
    cap.release()

    print(f"Reported FPS: {fps_reported}")
    print(f"Total frames: {frame_count}")
    print(f"Calculated duration: {duration:.2f} seconds")
    

    return


def images_to_video(input_folder, output_video, fps=12):
    # Get the list of image files in the folder
    image_files = [f for f in os.listdir(input_folder) if f.endswith('.png')]
    
    # Sort the image files to maintain order
    image_files.sort(key=lambda x: tuple(map(int, x.split('-')[1:-1])))
    print(image_files.sort(key=lambda x: tuple(map(int, x.split('-')[1:-1]))))
    #image_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

    # Read the first image to get dimensions
    img = cv2.imread(os.path.join(input_folder, image_files[0]))
    height, width, _ = img.shape

    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # You can also use 'XVID'
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # Write each image to the video
    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)
        img = cv2.imread(image_path)
        video.write(img)

    # Release the VideoWriter object
    video.release()








'''---------------------------------- PLOTTING RAW SAMPLED DATA ------------------------------------------------------------'''


def plots_mike(X, predictions, true, save_path, tag = None):
    '''
    X:              (np.array)      input test vector (eps and t)
    predictions:    (np.array)      predictions (sig)
    sig:            (np.array)      ground truth (sig)
    '''
    plotname_sig = np.array(['$n_x$', '$n_y$', '$n_{xy}$',
                            '$m_x$', '$m_y$', '$m_{xy}$',
                            '$v_{xz}$', '$v_{yz}$', '$v_y$'])
    
    plotname_eps = np.array([r'$\varepsilon_x$', r'$\varepsilon_y$', r'$\varepsilon_{xy}$',
                             r'$\chi_x$', r'$\chi_y$', r'$\chi_{xy}$',
                             r'$\gamma_x$', r'$\gamma_y$', r'$\gamma_{xy}$',
                             ])
    
    if tag == 'D':
        plotname_sig = np.array(['$D_{m,11}$', '$D_{m,12}$', '$D_{m,13}$', '$D_{mb,11}$', '$D_{mb,12}$', '$D_{mb,13}$',
                        '$D_{m,21}$', '$D_{m,22}$', '$D_{m,23}$', '$D_{mb,21}$', '$D_{mb,22}$', '$D_{mb,23}$',
                        '$D_{m,31}$', '$D_{m,32}$', '$D_{m,33}$', '$D_{mb,31}$', '$D_{mb,32}$', '$D_{mb,33}$',
                        '$D_{bm,11}$', '$D_{bm,12}$', '$D_{bm,13}$', '$D_{b,11}$', '$D_{b,12}$', '$D_{b,13}$',
                        '$D_{bm,21}$', '$D_{bm,22}$', '$D_{bm,23}$', '$D_{b,21}$', '$D_{b,22}$', '$D_{b,23}$',
                        '$D_{bm,31}$', '$D_{bm,32}$', '$D_{bm,33}$', '$D_{b,31}$', '$D_{b,32}$', '$D_{b,33}$',
                        '$D_{s,11}$', '$D_{s,22}$'
                        ])

    nRows = 8
    nCols = predictions.shape[1]
    fig, axs = plt.subplots(nCols, nRows, figsize=(2*nRows, 2*nCols))
    for i in range(nCols):
        for j in range(nRows):
            axs[i, j].plot(X[:,j], true[:,i], 'o', label = 'ground truth')
            axs[i, j].plot(X[:,j], predictions[:,i], 'ro', markerfacecolor = 'none', label='predictions')
            if i == nRows-1:
                axs[i, j].set_xlabel(plotname_eps[j])
            if j == 0:
                axs[i, j].set_ylabel(plotname_sig[i], rotation = 90)
    plt.legend()

    if save_path is not None:
        if tag == None:
            plt.savefig(os.path.join(save_path, "testo.png"))
            print('saved mike plot')
        elif tag == 'D':
            plt.savefig(os.path.join(save_path, "testo-D.png"))
            print('saved mike plot for D')
    plt.close()


def plots_mike_dataset(x_train, x_test, x_add, y_train, y_test, y_add, save_path, tag, tag2 = 'sig', numit = 0):
    '''
    train       (array)     data for training (either x or y)
    test        (test)      data for testing or evaluation (either x or y, corresp. to train)
    save_path   (str)       save_path
    '''

    # plt.rcParams.update({
    #     "text.usetex": True,
    #     "font.family": "Helvetica",
    #     "font.size": 12,
    #     })
    
    units = np.array([r' $[MN/cm]$', r' $[MN/cm]$', r' $[MN/cm]$',
                          r' $[MNcm/cm]$', r' $[MNcm/cm]$', r' $[MNcm/cm]$',  
                          r' $[MN/cm]$', r' $[MN/cm]$', r' $[MN/cm]$'])
    
    plotname_sig = np.array(['$n_x$', '$n_y$', '$n_{xy}$',
                            '$m_x$', '$m_y$', '$m_{xy}$',
                            '$v_{xz}$', '$v_{yz}$', '$t_1$', '$t_2$', '$n_{lay}$'])
    
    if tag2 == 'D':
        plotname_sig = np.array(['$D_{m,11}$', '$D_{m,12}$', '$D_{m,13}$', '$D_{mb,11}$', '$D_{mb,12}$', '$D_{mb,13}$',
                        '$D_{m,21}$', '$D_{m,22}$', '$D_{m,23}$', '$D_{mb,21}$', '$D_{mb,22}$', '$D_{mb,23}$',
                        '$D_{m,31}$', '$D_{m,32}$', '$D_{m,33}$', '$D_{mb,31}$', '$D_{mb,32}$', '$D_{mb,33}$',
                        '$D_{bm,11}$', '$D_{bm,12}$', '$D_{bm,13}$', '$D_{b,11}$', '$D_{b,12}$', '$D_{b,13}$',
                        '$D_{bm,21}$', '$D_{bm,22}$', '$D_{bm,23}$', '$D_{b,21}$', '$D_{b,22}$', '$D_{b,23}$',
                        '$D_{bm,31}$', '$D_{bm,32}$', '$D_{bm,33}$', '$D_{b,31}$', '$D_{b,32}$', '$D_{b,33}$',
                        '$D_{s,11}$', '$D_{s,22}$'
                        ])
    
    plotname_eps = np.array([r'$\varepsilon_x$', r'$\varepsilon_y$', r'$\varepsilon_{xy}$',
                             r'$\chi_x$', r'$\chi_y$', r'$\chi_{xy}$',
                             r'$\gamma_{xz}$', r'$\gamma_{yz}$', r'$t_1$', r'$t_2$', r'$n_{lay}$'
                             ])

    index_mask = np.array([0,1,2,3,4,5, 6, 7, 7])

    nRows = 8
    nCols = y_add.shape[1]
    fig, axs = plt.subplots(nCols, nRows, figsize=(3*nRows, 3*nCols))
    for i in range(nCols):
        for j in range(nRows):
            axs[i, j].plot(x_train[:,j], y_train[:,i], 'o', 
                           label = 'train', color = 'blue', alpha = 0.7, markersize = 3)
            axs[i, j].plot(x_test[:,j], y_test[:,i], 'o', 
                           markerfacecolor = 'lightcoral', markeredgecolor = 'lightcoral', 
                           label = tag, alpha = 0.1, markersize = 3)
            if x_add is not None:
                axs[i, j].plot(x_add[:,j], y_add[:,i], 'o', 
                           markerfacecolor = 'lightgreen', markeredgecolor = 'lightgreen', 
                           label = 'NN', alpha = 0.1, markersize = 3)
            if i == nRows-1:
                axs[i, j].set_xlabel(plotname_eps[j])
            # if j == 0:
            #     axs[i, j].set_ylabel(plotname_sig[i])
    
    plt.title('training vs '+tag+' data')
    plt.tight_layout
    plt.legend()

    if save_path is not None:
        plt.savefig(os.path.join(save_path, "data_scatter_"+tag+' '+str(numit)+".png"))
        print('saved data scatter plot ', tag)
    plt.close()


def plot_nathalie(data_in, data_in_test = None, save_path=None, tag=None):
    if tag == 'eps+t':
        if data_in.shape[1] == 9:
            col = ['epsx', 'epsy', 'epsxy', 'chix', 'chiy', 'chixy', 'gamx', 'gamy', 't']
            label = np.array([r'$\varepsilon_x$', r'$\varepsilon_y$', r'$\varepsilon_{xy}$',
                                r'$\chi_x$', r'$\chi_y$', r'$\chi_{xy}$',
                                r'$\gamma_x$', r'$\gamma_y$', r'$t$'
                                ])
        elif data_in.shape[1] == 10:
            col = ['epsx', 'epsy', 'epsxy', 'chix', 'chiy', 'chixy', 'gamx', 'gamy', 't_1', 't_2']
            label = np.array([r'$\varepsilon_x$', r'$\varepsilon_y$', r'$\varepsilon_{xy}$',
                                r'$\chi_x$', r'$\chi_y$', r'$\chi_{xy}$',
                                r'$\gamma_x$', r'$\gamma_y$', r'$t_1$', r'$t_2$'
                                ])
        elif data_in.shape[1] == 11:
            col = ['epsx', 'epsy', 'epsxy', 'chix', 'chiy', 'chixy', 'gamx', 'gamy', 't_1', 't_2', 'nl']
            label = np.array([r'$\varepsilon_x$', r'$\varepsilon_y$', r'$\varepsilon_{xy}$',
                                r'$\chi_x$', r'$\chi_y$', r'$\chi_{xy}$',
                                r'$\gamma_x$', r'$\gamma_y$', r'$t_1$', r'$t_2$', r'$n_{lay}$'
                                ])
    elif tag == 'eps+t_RC':
        col = ['epsx', 'epsy', 'epsxy', 'chix', 'chiy', 'chixy', 'gamx', 'gamy', 't', 'rho', 'CC']
        label = np.array([r'$\varepsilon_x$', r'$\varepsilon_y$', r'$\varepsilon_{xy}$',
                            r'$\chi_x$', r'$\chi_y$', r'$\chi_{xy}$',
                            r'$\gamma_x$', r'$\gamma_y$', r'$t$', r'$\rho$', r'$CC$'
                            ])
    if tag == 'sig':
        col = ['nx', 'ny', 'nxy', 'mx', 'my', 'mxy', 'vx', 'vy']
        label = np.array(['$n_x$', '$n_y$', '$n_{xy}$',
                            '$m_x$', '$m_y$', '$m_{xy}$',
                            '$v_{xz}$', '$v_{yz}$', '$v_y$'])
    elif tag == 'eps':
        col = ['epsx', 'epsy', 'epsxy', 'chix', 'chiy', 'chixy', 'gamx', 'gamy']
        label = np.array([r'$\varepsilon_x$', r'$\varepsilon_y$', r'$\varepsilon_{xy}$',
                             r'$\chi_x$', r'$\chi_y$', r'$\chi_{xy}$',
                             r'$\gamma_x$', r'$\gamma_y$', r'$\gamma_{xy}$',
                             ])
    elif tag == 't':
        col = ['t_1', 'rho', 'CC', 'Ec', 'tb0', 'tb1', 'ect', 'ec0', 'fcp', 'fct']
        label = np.array([r'$t_1$', r'$\rho$', r'$CC$', r'$Ec$', r'$tb0$', r'$tb1$',
                           r'$\varepsilon_{ct}$', r'$\varepsilon_{c0}$', r'$f_{cp}$', 
                           r'$f_{ct}$'])
    df_in = pd.DataFrame(data_in, columns = col)
    df_in = df_in.convert_dtypes()
    mat_fig = pd.plotting.scatter_matrix(df_in, alpha = 0.2, figsize= (8,8), diagonal='kde')
    if data_in_test is not None:
        df_in_eval = pd.DataFrame(data_in_test, columns=col)
        df_in_eval = df_in_eval.convert_dtypes()

        for i in range(len(df_in_eval.columns)):
            for j in range(len(df_in_eval.columns)):
                ax = mat_fig[i, j]
                ax.scatter(df_in_eval[df_in_eval.columns[j]], df_in_eval[df_in_eval.columns[i]], color='lightcoral', alpha=0.5, s=3)

    for i, ax in enumerate(mat_fig[:,0]):
        ax.set_ylabel(label[i], labelpad = 10, rotation=90)
    for j, ax in enumerate(mat_fig[-1,:]):
        ax.set_xlabel(label[j], labelpad = 10)
    
    if save_path is not None:
        plt.savefig(os.path.join(save_path, 'scatter_matrix_'+tag+'.png'), dpi = 300)
        print('Saved scatter matrix plot for '+tag)
    plt.tight_layout
    plt.close()
    return
'''
Copy from 04_Training, do not alter

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
import wandb
import pickle
import torch


# Define torch dataset
class MyDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]



def read_data(path: str, id: str):
    '''
    Reads in the data from given path
    id      (str)           Identifyer of data to be read: either 'sig', 'eps' or 't'
    '''
    with open(os.path.join(path, 'new_data_'+id +'.pkl'),'rb') as handle:
        new_data = pickle.load(handle)
    
    # transform data to numpy
    new_data = pd.DataFrame.from_dict(new_data)
    new_data_np = new_data.to_numpy()

    return new_data_np


def data_to_torch(X_train_t:np.array, y_train_t:np.array, X_eval_t:np.array, y_eval_t:np.array, 
                  X_test_t:np.array, y_test_t:np.array, path: str, batch_size: int):
    '''
    Takes numpy arrays and creates torch dataloaders
    Saves relevant information to mat_data_TrainEvalTest

    '''

    # batch_size = 1          # leave at 1 for the moment...

    # transform data to torch
    X_train_tt = torch.from_numpy(X_train_t)
    X_train_tt = X_train_tt.type(torch.float32)
    y_train_tt = torch.from_numpy(y_train_t)
    y_train_tt = y_train_tt.type(torch.float32)

    X_eval_tt = torch.from_numpy(X_eval_t)
    X_eval_tt = X_eval_tt.type(torch.float32)
    y_eval_tt = torch.from_numpy(y_eval_t)
    y_eval_tt = y_eval_tt.type(torch.float32)

    X_test_tt = torch.from_numpy(X_test_t)
    X_test_tt = X_test_tt.type(torch.float32)
    y_test_tt = torch.from_numpy(y_test_t)
    y_test_tt = y_test_tt.type(torch.float32)


    # Creating Datasets
    data_train_t = MyDataset(X_train_tt, y_train_tt)
    data_eval_t = MyDataset(X_eval_tt, y_eval_tt)
    data_test_t = MyDataset(X_test_tt, y_test_tt)

    # Transform to DataLoaders
    train_loader = DataLoader(data_train_t, batch_size, shuffle=True)
    val_loader = DataLoader(data_eval_t, batch_size, shuffle=False)
    test_loader = DataLoader(data_test_t, batch_size, shuffle=False)

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
    # mat['inp'] = inp
    # mat['stats_X_train'] = stats_X_train
    # mat['stats_y_train'] = stats_y_train
    # mat['stats_X_test'] = stats_X_test
    # mat['stats_y_test'] = stats_y_test
    mat['test_loader'] = test_loader

    with open(os.path.join(path, 'mat_data_TrainEvalTest.pkl'), 'wb') as fp:
        pickle.dump(mat, fp)


    return train_loader, val_loader, test_loader



def histogram(X:np.array, y:np.array, amt_data_points:int, nbins: int, name: str):
    '''
    Plots histograms of training and evaluation data set to check for consistency with previous steps of process.
    '''
    fig2, axs2 = plt.subplots(2, 4, sharey=True, tight_layout=True)
    keys1 = ['eps_x', 'eps_y', 'eps_xy', 'chi_x', 'chi_y', 'chi_xy', 'gamma_xz', 'gamma_yz']
    keys2 = ['n_x', 'n_y', 'n_xy', 'm_x', 'm_y', 'm_xy', 'v_xz', 'v_yz']


    if name == 'sig': 
        keys = keys2
        plot_data = y
    elif name == 'eps':
        keys = keys1
        plot_data = X
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


    if len(plot_data) == amt_data_points:
        fig2.suptitle('All Data, n = ' + str(len(plot_data)))
    elif len(plot_data) > 0.5*amt_data_points:
        fig2.suptitle('Training Data, n = ' + str(len(plot_data)))
    elif len(plot_data) < 0.2*amt_data_points:
        fig2.suptitle('Test Data, n = ' + str(len(plot_data)))
    else:
        fig2.suptitle('Validation Data, n = ' + str(len(plot_data)))

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
    stats = {
        'mean': mean,
        'std': std,
        'max': max,
        'min': min
    }
    return stats

def statistics(data:np.array):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    max = np.max(data, axis=0)
    min = np.min(data, axis=0)
    stats = {
        'mean': mean,
        'std': std,
        'max': max,
        'min': min
    }
    return stats



def transform_data(data:np.array, stats:dict, forward: bool):
    """
    Returns standardised data set with transformed data for the forward case (note: no physical sense anymore!).
    Or returns original-scale data set transformed back from given transformed data.

    Input: 
    data            (pd.DataFrame)      still the individual data sets for sig, eps
    stats           (dict)              contains statistical values that are constant for the transformation
    forward         (bool)              True: forward transform (to standardnormaldistr.), 
                                        False: backward transform to original values
                                        
    Output: 
    new_data       (np.array)           Uniformly distributed "new" data set
    """

    # Calculate new data

    data_stdnorm = np.zeros((data.shape))
    # data_nomaximin = np.zeros((data.shape))
    # data_maximin = np.zeros((data.shape))
    data_nonorm = np.zeros((data.shape))
    np_data = data
    # print(mean[0])
    if forward:
        for i in range(stats['std'].shape[0]):
            # 1 - Transformation to Standard Normal distribution
            data_stdnorm[:,i] = (np_data[:,i]-stats['mean'][i]*np.ones(np_data.shape[0]))/(stats['std'][i]*np.ones(np_data.shape[0]))

            # 2 - maximin to get values between 0 and 1
            # data_maximin[:,i] = (data_stdnorm[:,i]-stats['min'][i])/(stats['max'][i]-stats['min'][i])

        new_data = data_stdnorm
    elif not forward:
        for i in range(stats['std'].shape[0]):
            # 1 - Redo maximin
            # data_nomaximin[:,i] = np_data[:,i]*(stats['max'][i]-stats['min'][i]) + stats['min'][i]

            # 2 - Redo Standard normal distribution
            data_nonorm[:,i] = np_data[:,i]*stats['std'][i]*np.ones(np_data.shape[0])+stats['mean'][i]*np.ones(np_data.shape[0])
            # data_nonorm[:,i] = data_nomaximin[:,i]*stats['std'][i]*np.ones(data_nomaximin.shape[0])-stats['mean'][i]*np.ones(data_nomaximin.shape[0])

        new_data = data_nonorm
    return new_data




def init_wandb(inp, project_name):
    run = wandb.init(
    project = project_name,
    config = {
        "input_size": inp['input_size'], 
        "out_size": inp['out_size'],
        "hidden_layers": inp['hidden_layers'],
        "epochs": inp['num_epochs'],
        "learning_rate": inp['learning_rate'],
        "activation": inp['activation'],
        "learning_rate": inp['learning_rate'],
        "dropout_rate": inp['dropout_rate'],
        "num_samples": inp['num_samples'],
        "fourier_mapping": inp['fourier_mapping'],
        },
    )
    return run




def multiple_diagonal_plots(save_path: str, Y: np.array, predictions: np.array, transf:str):
    
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Helvetica",
        "font.size": 12,
        # "axes.size": 12,
        # "xtick.size": 12,
        # "ytick.size": 12,
        })
    
    # plt.rc('font', size=12) #controls default text size
    # plt.rc('axes', labelsize=12)
    # plt.rc('xtick', labelsize=12) #fontsize of the x tick labels
    # plt.rc('ytick', labelsize=12) #fontsize of the y tick labels

    # Calculate statistical values
    r_squared2 = np.zeros((1,8))
    # diff = np.zeros((Y.shape[0], 1))
    n_5p = np.zeros((1,8))
    n_10p = np.zeros((1,8))
    for i in range(8):
        Y_col = Y[:, i].flatten()
        pred_col = predictions[:,i].flatten()
        r_squared2[:,i] = np.corrcoef(Y_col, pred_col)[0, 1]**2
        diff_i = np.divide(abs(Y_col-pred_col), abs(Y_col))
        n_5p[:,i] = np.sum(diff_i>0.05)
        n_10p[:,i] = np.sum(diff_i>0.1)


    # Plot figure
    fig, axa = plt.subplots(3, 3, figsize=[10, 10], dpi=100)
    index_mask = np.array([[0,1,2],
                           [3,4,5],
                           [6,7,7]])
    num_rows = Y.shape[0]

    if transf == 'o':
        plotname = np.array([['$n_x$', '$n_y$', '$n_{xy}$'],
                            ['$m_x$', '$m_y$', '$m_{xy}$'],
                            ['$v_x$', '$v_y$', '$v_y$']])
        plotname_p = np.array([[r'$\tilde{n}_{x}$', r'$\tilde{n}_{y}$', r'$\tilde{n}_{xy}$'],
                            [r'$\tilde{m}_x$', r'$\tilde{m}_y$', r'$\tilde{m}_{xy}$'],
                            [r'$\tilde{v}_x$', r'$\tilde{v}_y$', r'$\tilde{v}_y$']])
        units = np.array([[r'$\rm [kN/m]$', r'$\rm [kN/m]$', r'$\rm [kN/m]$'],
                          [r'$\rm [kNm/m]$', r'$\rm [kNm/m]$', r'$\rm [kNm/m]$'],  
                          [r'$\rm [kN/m]$', r'$\rm [kN/m]$', r'$\rm [kN/m]$']])
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


    for i in range(3):
        for j in range(3):
            if i ==2 and j==2:
                    axa[i,j].set_title(' ')
            else: 
                axa[i,j].plot(Y[:,index_mask[i,j]], predictions[:,index_mask[i,j]], marker = 'o', ms = 5, linestyle='None')
                axa[i,j].set_ylabel(plotname_p[i,j]+' '+ units[i,j])
                axa[i,j].set_xlabel(plotname[i,j]+' '+ units[i,j])
                axa[i,j].set_xlim([np.min([np.min(Y[:,index_mask[i,j]]), np.min(predictions[:,index_mask[i,j]])]), np.max([np.max(Y[:,index_mask[i,j]]), np.max(predictions[:,index_mask[i,j]])])])
                axa[i,j].set_ylim([np.min([np.min(Y[:,index_mask[i,j]]), np.min(predictions[:,index_mask[i,j]])]), np.max([np.max(Y[:,index_mask[i,j]]), np.max(predictions[:,index_mask[i,j]])])])
                axa[i,j].grid(True, which='major', color='#666666', linestyle='-')
                at = AnchoredText('$R^2 = ' + np.array2string(r_squared2[:,index_mask[i,j]][0], precision=3) + '$ \n' +
                                  '$n_{5\%} = ' + np.array2string(n_5p[:,index_mask[i,j]][0].astype(int)) + '/' + np.array2string(np.array([num_rows])[0]) + '$ \n' +
                                  '$n_{10\%} = ' + np.array2string(n_10p[:,index_mask[i,j]][0].astype(int)) + '/' + np.array2string(np.array([num_rows])[0]) + '$',
                                prop=dict(size=10), frameon=True,loc='upper left')
                at.patch.set_boxstyle('round,pad=0.,rounding_size=0.2')
                axa[i,j].add_artist(at)
                axa[i,j].plot([np.min([np.min(Y[:,index_mask[i,j]]), np.min(predictions[:,index_mask[i,j]])]), np.max([np.max(Y[:,index_mask[i,j]]), np.max(predictions[:,index_mask[i,j]])])], [np.min([np.min(Y[:,index_mask[i,j]]), np.min(predictions[:,index_mask[i,j]])]), np.max([np.max(Y[:,index_mask[i,j]]), np.max(predictions[:,index_mask[i,j]])])], color='darkorange', linestyle='--',
                    linewidth = 3)
            
    axa[-1, -1].axis('off')
        

    axa = plt.gca()
    axa.set_aspect('equal', 'box')
    axa.axis('square')


    # Save figure
    plt.tight_layout()
    if save_path is not None:
        if transf == 't':
            filename = os.path.join(save_path, 'diagonal_match_'+'transformed.png')
            plt.savefig(filename, dpi=100, bbox_inches='tight')
            wandb.log({"45°-plot, t": wandb.Image(filename)})
        elif transf == 'o':
            filename = os.path.join(save_path, 'diagonal_match_'+'original.png')
            plt.savefig(filename, dpi=100, bbox_inches='tight')
            wandb.log({"45°-plot, og": wandb.Image(filename)})
    plt.show()
    plt.close()

    return





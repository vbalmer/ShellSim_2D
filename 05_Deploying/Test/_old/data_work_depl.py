import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset
from matplotlib.offsetbox import AnchoredText
import os


# Define torch dataset
class MyDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]




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




def multiple_diagonal_plots(save_path: str, Y: np.array, predictions: np.array, transf:str):
    '''
    Do not change this function, rather the one found in data_work.py under 05_Training
    '''

    
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Helvetica",
        "font.size": 12,
        })


    # Calculate statistical values
    r_squared2 = np.zeros((1,8))
    # mae = np.zeros((1,8))
    mask_5p = np.zeros((Y.shape[0], 8))
    mask_10p = np.zeros((Y.shape[0], 8))
    mask_100p = np.zeros((Y.shape[0], 8))
    mask_labels = np.zeros((Y.shape[0], 8))
    n_5p = np.zeros((1,8))
    n_10p = np.zeros((1,8))
    n_100p = np.zeros((1,8))
    diff_avg = np.zeros((1,8))          # this is referred to in the plot as MAPE (mean absolute percentage error)
    diff_i = np.zeros((Y.shape[0],8))
    for i in range(8):
        Y_col = Y[:, i].flatten()
        pred_col = predictions[:,i].flatten()
        r_squared2[:,i] = np.corrcoef(Y_col, pred_col)[0, 1]**2
        # mae[:,i] = np.mean(abs(Y_col-pred_col))
        diff_i[:,i] = np.divide(np.absolute(Y_col-pred_col), np.absolute(Y_col))
        diff_avg[:,i] = np.mean(diff_i[:,i])
        mask_5p[:,i] = diff_i[:,i] > 0.05
        mask_10p[:,i] = diff_i[:,i] > 0.1
        mask_100p[:,i] = diff_i[:,i] > 1
        mask_labels[:,i] = diff_i[:,i] < 0.3
        n_5p[:,i] = mask_5p[:,i].sum()
        n_10p[:,i] = mask_10p[:,i].sum()
        n_100p[:,i] = mask_100p[:,i].sum()

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
        # units = np.array([[r'$\rm [MN/mm]$', r'$\rm [MN/mm]$', r'$\rm [MN/mm]$'],
        #                   [r'$\rm [MNmm/mm]$', r'$\rm [MNmm/mm]$', r'$\rm [MNmm/mm]$'],  
        #                   [r'$\rm [MN/mm]$', r'$\rm [MN/mm]$', r'$\rm [MN/mm]$']])
        units = np.array([[r'$\rm [MN/cm]$', r'$\rm [MN/cm]$', r'$\rm [MN/cm]$'],
                          [r'$\rm [MNcm/cm]$', r'$\rm [MNcm/cm]$', r'$\rm [MNcm/cm]$'],  
                          [r'$\rm [MN/cm]$', r'$\rm [MN/cm]$', r'$\rm [MN/cm]$']])
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
                                # '$MAE = ' + np.array2string(mae[:,index_mask[i,j]][0], precision=3) + '$ \n' +
                                  '$MAPE = ' + np.array2string(diff_avg[:,index_mask[i,j]][0]*100, precision=3) + r'\%' +'$ \n' +
                                  '$n_{5\%} = ' + np.array2string(n_5p[:,index_mask[i,j]][0].astype(int)) + '/' + np.array2string(np.array([num_rows])[0]) + '$ \n' +
                                  '$n_{10\%} = ' + np.array2string(n_10p[:,index_mask[i,j]][0].astype(int)) + '/' + np.array2string(np.array([num_rows])[0]) + '$ \n' +
                                  '$n_{100\%} = ' + np.array2string(n_100p[:,index_mask[i,j]][0].astype(int)) + '/' + np.array2string(np.array([num_rows])[0]) + '$',
                                prop=dict(size=10), frameon=True,loc='upper left')
                at.patch.set_boxstyle('round,pad=0.,rounding_size=0.2')
                axa[i,j].add_artist(at)
                axa[i,j].plot([np.min([np.min(Y[:,index_mask[i,j]]), np.min(predictions[:,index_mask[i,j]])]), np.max([np.max(Y[:,index_mask[i,j]]), np.max(predictions[:,index_mask[i,j]])])], [np.min([np.min(Y[:,index_mask[i,j]]), np.min(predictions[:,index_mask[i,j]])]), np.max([np.max(Y[:,index_mask[i,j]]), np.max(predictions[:,index_mask[i,j]])])], color='darkorange', linestyle='--',
                    linewidth = 3)
                for l in range(Y.shape[0]):
                    if np.all(mask_labels[l]):
                        axa[i,j].text(Y[l,index_mask[i,j]], predictions[l, index_mask[i,j]], str(l), fontsize=12, color='red', ha='center', va='bottom')
            
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
            # wandb.log({"45°-plot, t": wandb.Image(filename)})
        elif transf == 'o':
            filename = os.path.join(save_path, 'diagonal_match_'+'original.png')
            plt.savefig(filename, dpi=100, bbox_inches='tight')
            # wandb.log({"45°-plot, og": wandb.Image(filename)})
    plt.show()
    plt.close()

    return


def multiple_diagonal_plots_Dnz(save_path: str, Y: np.array, predictions: np.array):

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Helvetica",
        "font.size": 12,
        })
    
        
    
    # move to correct units
    Y = Y*10**(-6)
    Y[:,0:3,0:3] = Y[:,0:3,0:3]*10**(1)
    Y[:,6:8,6:8] = Y[:,6:8,6:8]*10**(1)
    Y[:,3:6,3:6] = Y[:,3:6,3:6]*10**(-1)

    # Kick out irrelevant data (that should be zero) and reshape matrix to (num.rows x 12) format
    mat_comp_m = {
        "Simulation": np.hstack(((Y[:, 0:2, 0:2]).reshape((Y.shape[0],4)), Y[:,2,2].reshape((Y.shape[0],1)))),
        "Prediction": np.hstack(((predictions[:,0:2, 0:2]).reshape((Y.shape[0],4)), predictions[:,3,3].reshape((Y.shape[0],1))))
        }
    mat_comp_b = {
        "Simulation": np.hstack(((Y[:,3:5, 3:5]).reshape((Y.shape[0],4)), Y[:,5,5].reshape((Y.shape[0],1)))),
        "Prediction": np.hstack(((predictions[:,3:5, 3:5]).reshape((Y.shape[0],4)), predictions[:,5,5].reshape((Y.shape[0],1))))
        }

    mat_comp_s = {
        "Simulation": np.hstack((Y[:,6,6], Y[:,7,7])).reshape((Y.shape[0],2)),
        "Prediction": np.hstack((predictions[:,6,6], predictions[:,7,7])).reshape((Y.shape[0],2))
        # "Simulation": (Y[6:8, 6:8]).reshape((4)),
        # "Prediction": (predictions[6:8, 6:8]).reshape((4))
        }
    
    Y = np.hstack((mat_comp_m['Simulation'], mat_comp_b['Simulation'], mat_comp_s['Simulation']))
    predictions = np.hstack((mat_comp_m['Prediction'], mat_comp_b['Prediction'], mat_comp_s['Prediction']))
    print(Y.shape)
    print(predictions.shape)

    # Calculate statistical values
    r_squared2 = np.zeros((Y.shape[0],12))
    for i in range(12):
        Y_col = Y[:,i]
        pred_col = predictions[:,i]
        r_squared2[:,i] = np.corrcoef(Y_col, pred_col)[0, 1]**2


    # Plot figure
    fig, axa = plt.subplots(2, 6, figsize=[15, 7], dpi=100)
    
    num_rows = Y.shape[0]

    plotname = np.array([['$D_{m,11}$', '$D_{m,12}$', '$D_{m,21}$', '$D_{m,22}$', '$D_{m,33}$', '$D_{s,11}$'],
                        ['$D_{b,11}$', '$D_{b,12}$', '$D_{b,21}$', '$D_{b,22}$', '$D_{b,33}$', '$D_{s,22}$']])
    plotname_p = np.array([[r'$\tilde{D}_{m,11}$', r'$\tilde{D}_{m,12}$', r'$\tilde{D}_{m,21}$', r'$\tilde{D}_{m,22}$', r'$\tilde{D}_{m,33}$', r'$\tilde{D}_{s,11}$'],
                        [r'$\tilde{D}_{b,11}$', r'$\tilde{D}_{b,12}$', r'$\tilde{D}_{b,21}$', r'$\tilde{D}_{b,22}$', r'$\tilde{D}_{b,33}$', r'$\tilde{D}_{s,22}$']])
    units = np.array([[r'$\rm [MN/cm]$', r'$\rm [MN/cm]$', r'$\rm [MN/cm]$', r'$\rm [MN/cm]$', r'$\rm [MN/cm]$', r'$\rm [MN/cm]$'],
                        [r'$\rm [MNcm]$', r'$\rm [MNcm]$', r'$\rm [MNcm]$', r'$\rm [MNcm]$', r'$\rm [MNcm]$', r'$\rm [MN/cm]$']])
    
    index_mask = np.array([[0, 1, 2, 3, 4, 10],
                           [5, 6, 7, 8, 9, 11]])


    for i in range(2):
        for j in range(6):
            axa[i,j].plot(Y[:,index_mask[i,j]], predictions[:,index_mask[i,j]], marker = 'o', ms = 5, linestyle='None')
            axa[i,j].set_ylabel(plotname_p[i,j]+' '+ units[i,j])
            axa[i,j].set_xlabel(plotname[i,j]+' '+ units[i,j])
            axa[i,j].set_xlim([np.min([np.min(Y[:,index_mask[i,j]]), np.min(predictions[:,index_mask[i,j]])]), np.max([np.max(Y[:,index_mask[i,j]]), np.max(predictions[:,index_mask[i,j]])])])
            axa[i,j].set_ylim([np.min([np.min(Y[:,index_mask[i,j]]), np.min(predictions[:,index_mask[i,j]])]), np.max([np.max(Y[:,index_mask[i,j]]), np.max(predictions[:,index_mask[i,j]])])])
            axa[i,j].grid(True, which='major', color='#666666', linestyle='-')
            at = AnchoredText('$R^2 = ' + np.array2string(r_squared2[:,index_mask[i,j]][0], precision=3) + '$',
                            # '$MAE = ' + np.array2string(mae[:,index_mask[i,j]][0], precision=3) + '$ \n' +
                            #   '$MAPE = ' + np.array2string(diff_avg[:,index_mask[i,j]][0]*100, precision=3) + r'\%' +'$ \n' +
                            #   '$n_{5\%} = ' + np.array2string(n_5p[:,index_mask[i,j]][0].astype(int)) + '/' + np.array2string(np.array([num_rows])[0]) + '$ \n' +
                            #   '$n_{10\%} = ' + np.array2string(n_10p[:,index_mask[i,j]][0].astype(int)) + '/' + np.array2string(np.array([num_rows])[0]) + '$ \n' +
                            #   '$n_{100\%} = ' + np.array2string(n_100p[:,index_mask[i,j]][0].astype(int)) + '/' + np.array2string(np.array([num_rows])[0]) + '$',
                            prop=dict(size=10), frameon=True,loc='upper left')
            at.patch.set_boxstyle('round,pad=0.,rounding_size=0.2')
            axa[i,j].add_artist(at)
            axa[i,j].plot([np.min([np.min(Y[:,index_mask[i,j]]), np.min(predictions[:,index_mask[i,j]])]), np.max([np.max(Y[:,index_mask[i,j]]), np.max(predictions[:,index_mask[i,j]])])], [np.min([np.min(Y[:,index_mask[i,j]]), np.min(predictions[:,index_mask[i,j]])]), np.max([np.max(Y[:,index_mask[i,j]]), np.max(predictions[:,index_mask[i,j]])])], color='darkorange', linestyle='--',
                linewidth = 3)
            axa[i,j].set_aspect('equal', 'box')
            # for l in range(Y.shape[0]):
            #     if np.all(mask_labels[l]):
            #         axa[i,j].text(Y[l,index_mask[i,j]], predictions[l, index_mask[i,j]], str(l), fontsize=12, color='red', ha='center', va='bottom')
    
    axa = plt.gca()
    
    axa.axis('square')


    # Save figure
    plt.tight_layout()
    if save_path is not None:
        filename = os.path.join(save_path, 'diagonal_match_'+'D_nonzero.png')
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        # wandb.log({"45°-plot, t": wandb.Image(filename)})
    plt.show()
    plt.close()

    return




# def multiple_diagonal_plots_D(save_path: str, Y: np.array, predictions: np.array, D_analytical: np.array):
    
#     plt.rc('font', size=12) #controls default text size
#     plt.rc('axes', labelsize=12)
#     plt.rc('xtick', labelsize=12) #fontsize of the x tick labels
#     plt.rc('ytick', labelsize=12) #fontsize of the y tick labels


#     r_squared2 = np.zeros((8,8))
#     for i in range(8):
#         for j in range(8):
#             Y_col = Y[:,:,:,i,j]
#             pred_col = predictions[:,:,:,i,j]
#             r_squared2[i,j] = np.corrcoef(Y_col.flatten(), pred_col.flatten())[0, 1]**2



#     plotname = np.array(([['D_00', 'D_01', 'D_02', 'D_03', 'D_04', 'D_05','D_06', 'D_07'],
#                      ['D_10', 'D_11', 'D_12', 'D_13', 'D_14', 'D_15','D_16', 'D_17'],
#                      ['D_20', 'D_21', 'D_22', 'D_23', 'D_24', 'D_25','D_26', 'D_27'],
#                      ['D_30', 'D_31', 'D_32', 'D_33', 'D_34', 'D_35','D_36', 'D_37'],
#                      ['D_40', 'D_41', 'D_42', 'D_43', 'D_44', 'D_45','D_46', 'D_47'],
#                      ['D_50', 'D_51', 'D_52', 'D_53', 'D_54', 'D_55','D_56', 'D_57'],
#                      ['D_60', 'D_61', 'D_62', 'D_63', 'D_64', 'D_65','D_66', 'D_67'],
#                      ['D_70', 'D_71', 'D_72', 'D_73', 'D_74', 'D_75','D_76', 'D_77']]))

#     units = np.array(([['[N/mm]', '[N/mm]', '[N/mm]', '[N]', '[N]', '[N]', '[-]', '[-]'],
#                    ['[N/mm]', '[N/mm]', '[N/mm]', '[N]', '[N]', '[N]', '[-]', '[-]'],
#                    ['[N/mm]', '[N/mm]', '[N/mm]', '[N]', '[N]', '[N]', '[-]', '[-]'],
#                    ['[N]', '[N]', '[N]', '[Nmm]', '[Nmm]', '[Nmm]', '[-]', '[-]'],
#                    ['[N]', '[N]', '[N]', '[Nmm]', '[Nmm]', '[Nmm]', '[-]', '[-]'],
#                    ['[N]', '[N]', '[N]', '[Nmm]', '[Nmm]', '[Nmm]', '[-]', '[-]'],
#                    ['[-]', '[-]', '[-]', '[-]', '[-]', '[-]', '[N/mm]', '[N/mm]'],
#                    ['[-]', '[-]', '[-]', '[-]', '[-]', '[-]', '[N/mm]', '[N/mm]'],
#                    ]))



#     fig, axa = plt.subplots(8, 8, figsize=[40, 40], dpi=100)

#     for i in range(8):
#         for j in range(8):
#             axa[i,j].plot(Y[:,:,:,i,j].flatten(), predictions[:,:,:,i,j].flatten(), marker = 'o', ms = 5, linestyle='None')
#             axa[i,j].set_ylabel('Predicted '+ plotname[i,j] + units[i,j])
#             axa[i,j].set_xlabel('Reference '+ plotname[i,j] + units[i,j])
#             axa[i,j].set_xlim([np.min([np.min(Y[:,:,:,i,j].flatten()), np.min(predictions[:,:,:,i,j].flatten())]), np.max([np.max(Y[:,:,:,i,j].flatten()), np.max(predictions[:,:,:,i,j].flatten())])])
#             axa[i,j].set_ylim([np.min([np.min(Y[:,:,:,i,j].flatten()), np.min(predictions[:,:,:,i,j].flatten())]), np.max([np.max(Y[:,:,:,i,j].flatten()), np.max(predictions[:,:,:,i,j].flatten())])])
#             axa[i,j].grid(True, which='major', color='#666666', linestyle='-')
#             at = AnchoredText('$R^2$ = ' + np.array2string(r_squared2[i,j].flatten(), precision=3), #+
#             #             '\n$V_r$ = '+ np.array2string(Vr, precision=3),
#                 prop=dict(size=10), frameon=True,loc='upper left')
#             at.patch.set_boxstyle('round,pad=0.,rounding_size=0.2')
#             axa[i,j].add_artist(at)
#             axa[i,j].plot([np.min([np.min(Y[:,:,:,i,j].flatten()), np.min(predictions[:,:,:,i,j].flatten())]), np.max([np.max(Y[:,:,:,i,j].flatten()), np.max(predictions[:,:,:,i,j].flatten())])], [np.min([np.min(Y[:,:,:,i,j].flatten()), np.min(predictions[:,:,:,i,j].flatten())]), np.max([np.max(Y[:,:,:,i,j].flatten()), np.max(predictions[:,:,:,i,j].flatten())])], 
#                           color='darkorange', linestyle='--', linewidth = 3)
            
#             axa[i,j].plot(D_analytical[i,j]*np.ones(predictions[:,:,:,i,j].flatten().shape), np.linspace(-10e11, 10e11, Y.shape[0]), color='red', linestyle='--',linewidth = 2)
#             axa[i,j].plot(np.linspace(-10e11, 10e11, Y.shape[0]), D_analytical[i,j]*np.ones(Y[:,:,:,i,j].flatten().shape), color='red', linestyle='--',linewidth = 2)


#     axa = plt.gca()
#     axa.set_aspect('equal', 'box')
#     axa.axis('square')

    
#     plt.tight_layout()
#     if save_path is not None:
#         filename = os.path.join(save_path, 'diagonal_match_'+'stiffness.png')
#         plt.savefig(filename, dpi=100, bbox_inches='tight')
#         # wandb.log({"45°-plot": wandb.Image(filename)})
#     plt.show()
#     plt.close()

#     return




def bar_chart(D_analytical:np.array, D_pred_avg:np.array, D_sim_avg:np.array):
    
    titles = np.array([['Membrane stiffness $D_m$', 'Mixed stiffness $D_{mb}(1)$', 'Shear stiffness $D_s$'],
                       ['Mixed stiffness $D_{mb}(2)$', 'Bending stiffness $D_b$', 'Bending stiffness $D_s$']])
    
    id = np.array([['Dm', 'Dmb(1)', 'Ds'], 
                   ['Dmb(2)', 'Db', 'Ds']])
    
    # ylabel = np.array([['Stiffness [MN/mm]', 'Stiffness [MN]', 'Stiffness [MN/mm]'], 
    #                    ['Stiffness [MN]', 'Stiffness [MNmm]', 'Stiffness [MN/mm]']])
    ylabel = np.array([['Stiffness [MN/cm]', 'Stiffness [MN]', 'Stiffness [MN/cm]'], 
                       ['Stiffness [MN]', 'Stiffness [MNcm]', 'Stiffness [MN/cm]']])
    
    ymax = np.array([[200, 5e3, 200],
                    [5e3, 6.5e3, 200]])
    
    ymin = np.array([[0, -50, -20],
                    [0, 0, 0]])
    
    a = np.array([[0, 0],
                  [3, 3]])
    
    b = np.array([[0, 3],
                  [0, 3]])
    

    # to improve graphical display of numbers (convert to MN/cm, MNcm/cm or MNcm)
    D_analytical_plt = np.round(D_analytical*10**(-6), 4)
    D_analytical_plt[0:3, 0:3] = D_analytical_plt[0:3, 0:3]*10**(1)
    D_analytical_plt[6:8, 6:8] = D_analytical_plt[6:8, 6:8]*10**(1)
    D_analytical_plt[3:6, 3:6] = D_analytical_plt[3:6, 3:6]*10**(-1)

    D_sim_avg_plt = np.round(D_sim_avg*10**(-6),4)
    D_sim_avg_plt[0:3, 0:3] = D_sim_avg_plt[0:3, 0:3]*10**(1)
    D_sim_avg_plt[6:8, 6:8] = D_sim_avg_plt[6:8, 6:8]*10**(1)
    D_sim_avg_plt[3:6, 3:6] = D_sim_avg_plt[3:6, 3:6]*10**(-1)

    D_pred_avg_plt = np.round(D_pred_avg,4)

    fig, ax = plt.subplots(2, 3, layout="constrained", figsize = (10, 7))

    for i in range(2):
        for j in range(3):
            width = 0.25
            multiplier = 0  
            
            if id[i,j] == 'Ds':
                mat_comp = {
                    "Analytical": (D_analytical_plt[6:8, 6:8]).reshape((4)),
                    "Simulation": (D_sim_avg_plt[6:8, 6:8]).reshape((4)),
                    "Prediction": (D_pred_avg_plt[6:8, 6:8]).reshape((4))
                    }
                names = ("D_77", "D_78", "D_87", "D_88")
                x = np.arange(len(names))
                
            else: 
                mat_comp = {
                    "Analytical": np.hstack(((D_analytical_plt[a[i,j]:a[i,j]+2, b[i,j]:b[i,j]+2]).reshape((4)), D_analytical_plt[a[i,j]+2,b[i,j]+2])),
                    "Simulation": np.hstack(((D_sim_avg_plt[a[i,j]:a[i,j]+2, b[i,j]:b[i,j]+2]).reshape((4)), D_sim_avg_plt[a[i,j]+2,b[i,j]+2])),
                    "Prediction": np.hstack(((D_pred_avg_plt[a[i,j]:a[i,j]+2, b[i,j]:b[i,j]+2]).reshape((4)), D_pred_avg_plt[a[i,j]+2,b[i,j]+2]))
                    }

                if id[i,j] == "Dm": names = ("D_11", "D_12", "D_21", "D_22", "D_33")
                elif id[i,j] == "Db": names = ("D_44", "D_45", "D_54", "D_55", "D_66")
                elif id[i,j] == "Dmb(1)": names = ("D_14", "D_15", "D_24", "D_25", "D_36")
                elif id[i,j] == "Dmb(2)": names = ("D_41", "D_51", "D_42", "D_52", "D_63")
                x = np.arange(len(names))
            
            # Plotting
            for attribute, measurement in mat_comp.items():
                offset = width * multiplier
                if i == 1 and j == 0: 
                    rects = ax[i,j].bar(x + offset, measurement, width, label=attribute)
                    ax[i,j].bar_label(rects, padding=3, rotation = 90, fmt='%.2f')
                    ax[i,j].set_title(titles[i,j])
                    ax[i,j].set_ylabel(ylabel[i,j])
                    ax[i,j].set_ylim(ymin[i,j], ymax[i,j])
                elif i == 1 and j == 2: 
                    ax[i,j].set_title('  ')
                else:
                    rects = ax[i,j].bar(x + offset, measurement, width)
                    ax[i,j].bar_label(rects, padding=3, rotation = 90, fmt='%.2f')
                    ax[i,j].set_title(titles[i,j])
                    ax[i,j].set_ylabel(ylabel[i,j])
                    ax[i,j].set_ylim(ymin[i,j], ymax[i,j])
                multiplier += 1
            
            ax[i,j].set_xticks(x + width, names)


    fig.legend(loc='lower right', ncols=1)
    ax[-1, -1].axis('off')

    plt.show()

    return


def D_plt(D_coeff, id, original):
    if original:
        if id == 'D_m':
            fig, ax = plt.subplots(ncols = 1)
            pos = ax.matshow(D_coeff[0:3, 0:3])
            ax.set_title('D_m_coeff')
            fig.colorbar(pos)
        elif id =='D_b':
            fig, ax = plt.subplots(ncols = 1)
            pos = ax.matshow(D_coeff[3:6, 3:6])
            ax.set_title('D_b_coeff')
            fig.colorbar(pos)
        elif id == 'all':
            fig, ax = plt.subplots(ncols = 1)
            pos = ax.matshow(D_coeff)
            ax.set_title('D_all_coeff')
            fig.colorbar(pos)
    else:
        if id == 'D_m':
            fig, ax = plt.subplots(ncols = 1)
            pos = ax.matshow(D_coeff[0:3, 0:3])
            ax.set_title('D_m_t')
            fig.colorbar(pos)
        elif id =='D_b':
            fig, ax = plt.subplots(ncols = 1)
            pos = ax.matshow(D_coeff[3:6, 3:6])
            ax.set_title('D_b_t')
            fig.colorbar(pos)
        elif id == 'all':
            fig, ax = plt.subplots(ncols = 1)
            pos = ax.matshow(D_coeff)
            ax.set_title('D_all_t')
            fig.colorbar(pos)
    
    plt.show()
    return





def D_an(eps:np.array, t: float):
    '''
    returns analytically calculated sig for given eps and t
    E, nu are assumed constant (as we assume analytical formulation for steel)
    eps expected in [-] or [1/mm]; t in [mm]
    D_analytical in [N/mm], [Nmm], [N]; sig in [N/mm] or [N]
    '''
    nu = 0.3
    E = 210000

    D_p = (E/(1+nu**2))*np.array([[1, nu, 0], 
                           [nu, 1, 0], 
                           [0, 0, 0.5 * (1 - nu)]])

    Dse = (5/6)*t*(2*E)/(4*(1+nu))


    D_an_1 = np.hstack([t*D_p, 0*D_p, np.zeros((3,2))])
    D_an_2 = np.hstack([0*D_p, (1/12)*(t**3)*D_p, np.zeros((3,2))])
    D_an_3 = np.hstack([np.zeros((2,3)), np.zeros((2,3)), np.array([[Dse, 0], [0, Dse]])])
    D_analytical = np.vstack([D_an_1, D_an_2, D_an_3])

    sig = np.matmul(D_analytical,eps)

    return D_analytical, sig




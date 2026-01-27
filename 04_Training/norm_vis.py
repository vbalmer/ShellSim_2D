# normalisation visualisation
# bav, 15.07.2025

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

ID = 'bm'                # 'm', 'b' or 's' (or 'mb', 'bm' in nonlinear case) for D
ID_SIG = 'sig-m'         # 'sig-m', 'sig-b' or 'sig-s' for sig-boxplots
LINEL = False
LOG = False


def reshape_De_values(De_original_, De_single_norm_, De_double_norm_, id, linel):
    
    if linel:
        if id == 'm': 
            j = [0,1,2]
        elif id == 'b': 
            j = [3,4,5]
        elif id == 's':
            j = [6,7,7]

        D_m_original_ = np.concatenate((De_original_[:,j[0],j[0]].reshape((-1,1)), De_original_[:,j[0],j[1]].reshape((-1,1)), 
                                De_original_[:,j[1],j[1]].reshape((-1,1)), De_original_[:,j[2],j[2]].reshape((-1,1))), axis = 1)
        D_m_single_norm_ = np.concatenate((De_single_norm_[:,j[0],j[0]].reshape((-1,1)), De_single_norm_[:,j[0],j[1]].reshape((-1,1)), 
                                            De_single_norm_[:,j[1],j[1]].reshape((-1,1)), De_single_norm_[:,j[2],j[2]].reshape((-1,1))), axis = 1)
        D_m_double_norm_ = np.concatenate((De_double_norm_[:,j[0],j[0]].reshape((-1,1)), De_double_norm_[:,j[0],j[1]].reshape((-1,1)), 
                                        De_double_norm_[:,j[1],j[1]].reshape((-1,1)), De_double_norm_[:,j[2],j[2]].reshape((-1,1))), axis = 1)
    else: 
        if id == 'm': 
            D_i_original_ = De_original_[:,:3,:3]   
            D_i_single_norm_ = De_single_norm_[:,:3,:3]
            D_i_double_norm_ = De_double_norm_[:,:3,:3]
        elif id == 'b':
            D_i_original_ = De_original_[:,3:6,3:6]
            D_i_single_norm_ = De_single_norm_[:,3:6,3:6]
            D_i_double_norm_ = De_double_norm_[:,3:6,3:6]
        elif id == 'mb':
            D_i_original_ = De_original_[:,0:3,3:6]
            D_i_single_norm_ = De_single_norm_[:,0:3,3:6]
            D_i_double_norm_ = De_double_norm_[:,0:3,3:6]
        elif id == 'bm':
            D_i_original_ = De_original_[:,3:6,0:3]
            D_i_single_norm_ = De_single_norm_[:,3:6,0:3]
            D_i_double_norm_ = De_double_norm_[:,3:6,0:3]
        elif id == 's':
            D_i_original_ = De_original_[:,6:8,6:8]
            D_i_single_norm_ = De_single_norm_[:,6:8,6:8]
            D_i_double_norm_ = De_double_norm_[:,6:8,6:8]

        if id != 's':
            k,i = np.triu_indices(3,k=0)
        else: 
            k,i = np.triu_indices(2,k=0)
        D_m_original_ = D_i_original_[:,k,i]
        D_m_single_norm_ = D_i_single_norm_[:,k,i]
        D_m_double_norm_ = D_i_double_norm_[:,k,i]

    if id == 's' and linel:
        D_m = {'original': D_m_original_[:,:-1], 'single_norm': D_m_single_norm_[:,:-1], 'double_norm': D_m_double_norm_[:,:-1]}
    else:
        D_m = {'original': D_m_original_, 'single_norm': D_m_single_norm_, 'double_norm': D_m_double_norm_}

    print('Shapes after preparation for plotting:', D_m['original'].shape, D_m['single_norm'].shape, D_m['double_norm'].shape)
    return D_m

def reshape_sig_values(sig_original, sig_single_norm, sig_double_norm, id):
    if id == 'sig-m':
        sig_m = {'original': sig_original[:,:3], 'single_norm': sig_single_norm[:,:3], 'double_norm': sig_double_norm[:,:3]}
    elif id == 'sig-b':
        sig_m = {'original': sig_original[:,3:6], 'single_norm': sig_single_norm[:,3:6], 'double_norm': sig_double_norm[:,3:6]}
    elif id == 'sig-s':
        sig_m = {'original': sig_original[:,6:], 'single_norm': sig_single_norm[:,6:], 'double_norm': sig_double_norm[:,6:]}



    return sig_m

def create_boxplot(De, linel, id, MAGIC_FACTOR1 = 0.2, MAGIC_FACTOR2 = 1, save_path=None):
    '''
    Creates a boxplot for De values (original, normalised and double-normalised)
    De      (np.array)  was originally the dataset of stiffnesses, can also be stresses that should be plotted
    linel   (bool)      if linel: shape of De is expected to be (n, 4) per D_i otw: (n,6) per D_i
    id      (str)       can be 'm', 's', 'b', 'mb', 'bm'
    '''
    if LINEL: 
        labels = ['D_11', 'D_12', 'D_22', 'D_33']
        if id == 's':
            labels = ['D_11', 'D_12', 'D_22']
    else:
        labels = ['D_11', 'D_12', 'D_13', 'D_22', 'D_23', 'D_33']
        if id == 's':
            labels = ['D_11', 'D_12', 'D_22']
        if id == 'sig-m':
            labels = ['n_x', 'n_y', 'n_xy']
        if id == 'sig-b':
            labels = ['m_x', 'm_y', 'mxy']
        if id == 'sig-s':
            labels = ['v_x', 'v_y']
    positions = np.arange(len(labels))
    width = 0.1


    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    # Plot original values on ax1 (left y-axis)
    box1 = ax1.boxplot(De['original'], positions=positions - width, widths=width, patch_artist=True,
                    boxprops=dict(facecolor='skyblue'), medianprops=dict(color='black'))

    if id == 's':
        mask = np.array([1, 1, 1])
    elif 'sig' not in id: 
        mask = (MAGIC_FACTOR1-1)*np.array([0, 1, 1, 0, 1, 0])+1
    else: 
        mask = np.array([1,1,1])

    box2 = ax2.boxplot(np.divide(De['single_norm'], mask), positions=positions, widths=width, patch_artist=True,
                    boxprops=dict(facecolor='lightgreen'), medianprops=dict(color='black'))

    box3 = ax2.boxplot(np.divide(De['double_norm'], MAGIC_FACTOR2), positions=positions + width, widths=width, patch_artist=True,
                    boxprops=dict(facecolor='salmon'), medianprops=dict(color='black'))


    # Set x-ticks and labels
    ax1.set_xticks(positions)
    ax1.set_xticklabels(labels)

    # Set labels
    ax1.set_ylabel('Original Value [MN,cm]')
    ax2.set_ylabel('Normalised Value')
    if LINEL:
        ax2.set_ylim([0,6])

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(color='skyblue', label='Original'),
        Patch(color='lightgreen', label='std-stitched'),
        Patch(color='salmon', label='std')
    ]
    ax1.legend(handles=legend_handles, loc='upper left')
    if 'sig' in id:
        plt.title(id)
        plt.savefig(os.path.join(save_path, 'boxplot_'+id+'.jpg'))
        print('saved boxplot '+id)
    else:
        plt.title('D_'+id)
        plt.savefig(os.path.join(save_path, 'boxplot_D_'+id+'.jpg'))
        print('saved boxplot D_'+id)
    return



#### MAIN #####

# Read data
if LINEL: 
    path1 = os.path.join(os.getcwd(),'04_Training\\new_data\\data_normalisation_linel\\single')
    path2 = os.path.join(os.getcwd(),'04_Training\\new_data\\data_normalisation_linel\\double')
elif LOG:
    path1 = os.path.join(os.getcwd(),'04_Training\\new_data\\data_normalisation_nonlin\\single')
    path2 = os.path.join(os.getcwd(),'04_Training\\new_data\\data_normalisation_nonlin\\log')
else:
    path1 = os.path.join(os.getcwd(),'04_Training\\new_data\\data_normalisation_nonlin\\single\\data_20250608_1838_fake')
    path2 = os.path.join(os.getcwd(),'04_Training\\new_data\\data_normalisation_nonlin\\double\\data_20250608_1838_fake_pureD')

read_data = []

for name in ['mat_data_np_TrainEvalTest', 'mat_data_TrainEvalTest']:
    for path in [path1, path2]:
        with open(os.path.join(path, name+'.pkl'),'rb') as handle:
                    data = pickle.load(handle)
                    read_data.append(data)

if read_data[3]['y_train_tt'].shape[1] == 38:  # in case the double path leads to the pure-D data

    De_original = read_data[0]['y_train'][:, 8:72]
    De_single_norm = read_data[2]['y_train_tt'][:, 8:72].numpy()
    De_original_ = De_original.reshape((-1,8,8))
    De_single_norm_ =  De_single_norm.reshape((-1,8,8))

    # only the "double-norm", which in this case would be the pure-D "std"-standardisation needs to be read differently.
    De_double_norm_ = np.zeros((read_data[3]['y_train_tt'].shape[0],8,8))
    De_double_norm_[:,:6,:6] = read_data[3]['y_train_tt'][:,:36].reshape((-1,6,6))
    De_double_norm_[:,6,6] = read_data[3]['y_train_tt'][:,36]
    De_double_norm_[:,7,7] = read_data[3]['y_train_tt'][:,37]
    
    # in this case, sigma is not calculated.

else: 
    De_original = read_data[0]['y_train'][:, 8:72]
    De_single_norm = read_data[2]['y_train_tt'][:, 8:72].numpy()
    De_double_norm = read_data[3]['y_train_tt'][:, 8:72].numpy()
    sig_original = read_data[0]['y_train'][:,:8]
    sig_single_norm = read_data[2]['y_train_tt'][:,:8].numpy()
    sig_double_norm = read_data[3]['y_train_tt'][:,:8].numpy()

    De_original_ = De_original.reshape((-1,8,8))
    De_single_norm_ =  De_single_norm.reshape((-1,8,8))
    De_double_norm_ = De_double_norm.reshape((-1,8,8))


print('D-Shapes after reading in data:', De_original_.shape, De_single_norm_.shape, De_double_norm_.shape)
if read_data[3]['y_train_tt'].shape[1] != 38:
    print('sig-Shapes after reading in data:', sig_original.shape, sig_single_norm.shape, sig_double_norm.shape)

# Reshaping
De = reshape_De_values(De_original_, De_single_norm_, De_double_norm_, id = ID, linel = LINEL)
if read_data[3]['y_train_tt'].shape[1] != 38:
    sig = reshape_sig_values(sig_original, sig_single_norm, sig_double_norm, id = ID_SIG)


# Plotting

create_boxplot(De, linel = LINEL, id = ID, save_path = path2)

if read_data[3]['y_train_tt'].shape[1] != 38:
    create_boxplot(sig, linel = LINEL, id = ID_SIG, save_path=path2)
















### NOT IN USE ###

# D_m_original = np.concatenate((De_original_[:,0,0], De_original_[:,0,1], 
#                                De_original_[:,1,1], De_original_[:,2,2]), axis = 0)
# D_m_single_norm = np.concatenate((De_single_norm_[:,0,0], De_single_norm_[:,0,1], 
#                                     De_single_norm_[:,1,1], De_single_norm_[:,2,2]), axis = 0)
# D_m_double_norm = np.concatenate((De_double_norm_[:,0,0], De_double_norm_[:,0,1], 
#                                     De_double_norm_[:,1,1], De_double_norm_[:,2,2]), axis = 0)

# D_m_tot = np.concatenate((D_m_original, D_m_single_norm, D_m_double_norm))


# df = pd.DataFrame({
#        'Normalisation Type': ['Original'] * D_m_original.shape[0] + ['Single-Norm'] * D_m_original.shape[0] + ['Double-Norm'] * D_m_original.shape[0],
#        'D_id': ['D_m11']*De_original_.shape[0] + ['D_m12']*De_original_.shape[0]  + ['D_m22']*De_original_.shape[0] + ['D_m33']*De_original_.shape[0] +
#                ['D_m11']*De_single_norm_.shape[0] + ['D_m12']*De_single_norm_.shape[0]  + ['D_m22']*De_single_norm_.shape[0] + ['D_m33']*De_single_norm_.shape[0] + 
#                ['D_m11']*De_double_norm_.shape[0] + ['D_m12']*De_double_norm_.shape[0]  + ['D_m22']*De_double_norm_.shape[0] + ['D_m33']*De_double_norm_.shape[0],
#        'value': D_m_tot

# })
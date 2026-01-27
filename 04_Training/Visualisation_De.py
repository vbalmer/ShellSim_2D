# visualisations for De

import numpy as np
import os
import pickle
from datetime import datetime
import matplotlib.pyplot as plt

from data_work import read_data


def scatter_simple(D_xx: np.array, save_path: str, tag:str):
    fig, ax = plt.subplots(D_xx.shape[1],D_xx.shape[1])
    for i in range(D_xx.shape[1]): 
        for j in range(D_xx.shape[1]):
            ax[i,j].scatter(D_xx[:,i], D_xx[:,j],s=1, color = 'blue', alpha = 0.5)
    if save_path is not None:
        plt.savefig(os.path.join(save_path, 'scatter_De'+tag), dpi = 300)
        print('Saved scatter De plot for '+tag)
    plt.close()


data = '04_Training\data\data_20250210_1745_fake'

path = os.getcwd()
path_data = os.path.join(path, data)
path_plots = os.path.join(path, '04_Training\\plots')

new_data_De_np = read_data(path_data, 'De').reshape((-1,8,8))

print(new_data_De_np.shape)


# scatter plot for D_m
D_m = new_data_De_np[:,0:3,0:3].reshape((-1,9))
scatter_simple(D_m, path_plots,'Dm')

# scatter plot for other D matrices
D_mb = new_data_De_np[:,0:3,3:6].reshape((-1,9))
scatter_simple(D_mb, path_plots,'Dmb')
D_bm = new_data_De_np[:,3:6,0:3].reshape((-1,9))
scatter_simple(D_bm, path_plots,'Dbm')
D_b = new_data_De_np[:,3:6,3:6].reshape((-1,9))
scatter_simple(D_b, path_plots,'Db')
D_s = new_data_De_np[:,6:8,6:8].reshape((-1,4))
scatter_simple(D_s, path_plots,'Ds')













# simple histogram
# fig, ax = plt.subplots(8,8)
# for i in range(8):
#   for j in range(8):
#       ax[i,j].hist(new_data_De_np[:,i,j])
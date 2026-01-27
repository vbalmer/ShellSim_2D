import numpy as np
from data_work import *


import matplotlib.pyplot as plt
import os
from datetime import datetime
from sampler_utils import *
import time


# State max, min for eps and t

##### Glass: 
# Sapmling boundaries level I
min = [-6e-5]*3+[-0.115e-3]*3+[-6e-5]*2
max = [6e-5]*3+[0.115e-3]*3+[6e-5]*2


# Thickness, material
# 1 = Glass, 2 = PVB
t_1 = [5, 10, 15, 20, 25]
t_2 = [0.4, 0.8, 1.2, 1.6, 2.0, 2.4]
num_layer = [3, 5, 7]
nu_1 = 0.24
nu_2 = 0.499
E_1 = 70000
E_2 = 0.003



##### Other
save_folder = True
n_samples = 60000
material = 10                # 10 = glass




################################## NO CHANGES REQUIRED BELOW HERE ##################################

# Sample the data
analytical_sampler = Sampler_utils_vb(min, max, E_1, nu_1, E_2, nu_2)
eps_and_t = analytical_sampler.sample(t_1, t_2, num_layer, n_samples)

eps = eps_and_t[:, 0:8]
t = eps_and_t[:,8:10].reshape(-1,2)
num_layer_sampled = eps_and_t[:,10].reshape(-1,1)

calc_method = 'single'
t0 = time.time()
dict_sampler = analytical_sampler.D_an(eps, t, num_layers=num_layer_sampled, mat = material, calc_meth=calc_method, discrete='andreas')
t1 = time.time()
print('Time for sampling: ', t1-t0)
sig_a = dict_sampler['sig_a']
D_a = dict_sampler['D_a']

# Reshape data
sig_a = np.squeeze(sig_a, axis=2)
# eps = np.squeeze(eps, axis = 2)
if calc_method == 'all':
    eps_rep = np.tile(eps, (1000,1))
    t_rep = np.tile(t, (1000))
else: 
    eps_rep = eps
    t_rep = np.concatenate((t, num_layer_sampled), 1)


# Plotting
path_plots = os.path.join(os.getcwd(), '04_Training\\plots')
plot_nathalie(np.concatenate((eps_rep[:,0:8], t_rep), axis=1), data_in_test = None, save_path = path_plots, tag = 'eps+t')
plot_nathalie(sig_a, data_in_test = None, save_path = path_plots, tag = 'sig')
plots_mike_dataset(eps_rep, eps_rep, sig_a, sig_a, path_plots, tag='test')

histogram(eps_rep, D_a.reshape((-1, 8, 8)), int(D_a.shape[0]), nbins=50, name='De', path=path_plots)



# save data

if save_folder:
    data_path = '04_Training\data'
    folder_name = f"data_{datetime.now().strftime('%Y%m%d_%H%M')}_fake"
    save_data_path = os.path.join(data_path, folder_name)
    os.makedirs(save_data_path, exist_ok=True)

    # t_rep = t_rep.reshape(-1, 2)
    with open(os.path.join(save_data_path, 'new_data_t.pkl'), 'wb') as fp:
            pickle.dump(t_rep.astype(np.float32), fp)
    with open(os.path.join(save_data_path, 'new_data_eps.pkl'), 'wb') as fp:
            pickle.dump(eps_rep.astype(np.float32), fp)
    with open(os.path.join(save_data_path, 'new_data_sig.pkl'), 'wb') as fp:
            pickle.dump(sig_a.astype(np.float32), fp)
    D_a = D_a.reshape(t_rep.shape[0], 64)
    with open(os.path.join(save_data_path, 'new_data_De.pkl'), 'wb') as fp:
            pickle.dump(D_a.astype(np.float32), fp)  
    print('Data saved to ', save_data_path)

else: 
    print('Data not saved')	

print('data shapes:')
print('t: ', t_rep.shape)
print('eps: ', eps_rep.shape)
print('sig: ', sig_a.shape)
print('De: ', D_a.shape)

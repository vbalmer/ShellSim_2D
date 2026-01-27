import numpy as np
from data_work import *


import matplotlib.pyplot as plt
import os
from datetime import datetime
from sampler_utils import *
import time


# State max, min for eps and t

##### Steel: 
# Sapmling boundaries level I
# min = [-6e-6]*2+[-6e-6]+[-20e-5]+[-20e-5]+[-20e-5]+[-4e-5]+[-4e-5]
# max = [6e-6]*2 +[6e-6]+[20e-5] +[20e-5] +[20e-5] +[4e-5] +[4e-5]
# min = [-0.28e-3]*3+[-0.15e-3]*3+[-0.42e-3]*2        # additional sampling boundaries
# max = [0.28e-3]*3 +[0.15e-3]*3 +[0.42e-3]*2         # additional sampling boundaries

# Sampling boundaries level III
# min = [-1.12e-3]*3+[-0.15e-3]*3+[-1.67e-3]*2
# max = [1.12e-3]*3 +[0.15e-3]*3 +[1.67e-3]*2


# Sampling boudaries level II
# min = [-2e-5]*3+[-0.15e-3]*3+[-4e-4]*2
# max = [2e-5]*3 +[0.15e-3]*3 +[4e-4]*2

# Thickness, material
# t = [15, 20, 25, 30]
# nu = 0.3
# E = 210000


##### Reinforced Concrete
# Sampling boundaries level I
# min = [-10e-6]*3+[-5e-6]*3+[-30e-6]*2
# max = [5e-6]*3 +[5e-6]*3 +[30e-6]*2


# Sampling boundaries level II
# min = [-0.1e-3]*2  +[-0.1e-3] +[-0.005e-3]*2 +[-0.01e-3] +[-0.1e-3]*2
# max = [0.005e-3]*2 +[0.1e-3]  +[0.005e-3]*2  +[0.01e-3]  +[0.1e-3]*2


# Sampling boundaries level III
min = [-0.37e-3]*2+[-0.47e-3] +[-0.005e-3]*2+[-0.01e-3] +[-0.47e-3]*2
# max = [0.09e-3]*2 +[0.47e-3]  +[0.005e-3]*2 +[0.01e-3]  +[0.47e-3]*2
max = [0.37e-3]*2 +[0.47e-3]  +[0.005e-3]*2 +[0.01e-3]  +[0.47e-3]*2


# Sampling boundaries for 2D case:
min = [-0.37e-3]*2+[-0.47e-3] +[-1e-15]*2+[-1e-15] +[-1e-15]*2
max = [0.37e-3]*2 +[0.47e-3]  +[1e-15]*2 +[1e-15]  +[1e-15]*2

# Thickness, material
t = [20,20]
# t = [5, 10, 15, 20, 25, 30]
# t = [200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400]
nu = 0.2
E = 33600


##### Other
save_folder = True
n_samples = 60000
material = 1




################################## NO CHANGES REQUIRED BELOW HERE ##################################

# Sample the data
analytical_sampler = Sampler_utils_vb(E1 = E, nu1 = nu, E2 = None)
eps_and_t = analytical_sampler.sample(min, max, t1=t, t2 = None, num_layer=None, num_samples=n_samples)

eps = eps_and_t[:, 0:8]
t = eps_and_t[:,8].reshape(-1,1)

calc_method = 'single'
t0 = time.time()
dict_sampler = analytical_sampler.D_an(eps, t, num_layers=1, mat = material, calc_meth=calc_method, discrete='andreas')
t1 = time.time()
print('Time for sampling: ', t1-t0)
sig_a = dict_sampler['sig_a']
D_a = dict_sampler['D_a']

# Reshape data
sig_a = np.squeeze(sig_a, axis=2)
sig_a[sig_a == 0] = 1e-10
# eps = np.squeeze(eps, axis = 2)
if calc_method == 'all':
    eps_rep = np.tile(eps, (1000,1))
    t_rep = np.tile(t, (1000))
else: 
    eps_rep = eps
    t_rep = t


# Plotting
path_plots = os.path.join(os.getcwd(), '04_Training\\plots')
plot_nathalie(np.concatenate((eps_rep[:,0:8], t_rep.reshape(-1,1)), axis=1), data_in_test = None, save_path = path_plots, tag = 'eps+t')
plot_nathalie(sig_a, data_in_test = None, save_path = path_plots, tag = 'sig')
plots_mike_dataset(eps_rep, eps_rep, sig_a, sig_a, path_plots, tag='test')

histogram(eps_rep, D_a.reshape((-1, 8, 8)), int(D_a.shape[0]), nbins=50, name='De', path=path_plots)



# save data

if save_folder:
    data_path = '04_Training\data'
    folder_name = f"data_{datetime.now().strftime('%Y%m%d_%H%M')}_fake"
    save_data_path = os.path.join(data_path, folder_name)
    os.makedirs(save_data_path, exist_ok=True)

    t_rep = t_rep.reshape(-1, 1)
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

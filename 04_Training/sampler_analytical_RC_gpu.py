import cupy as np
import pickle
from data_work import plot_nathalie, plots_mike_dataset, histogram
from concrete_classes import dict_CC


import os
from datetime import datetime
from sampler_utils_gpu import *
import time


# State max, min for eps and t
# min = [-3e-3] + [0] + [0] + [0]*2 + [0] + [0]*2
# max = [5e-3]  + [1e-20] + [1e-20] + [1e-20]*2 + [1e-20] + [1e-20]*2
# min = [0]       + [-3e-3] + [0]     + [0]*2     + [0]     + [0]*2
# max = [1e-20]   + [5e-3]  + [1e-20] + [1e-20]*2 + [1e-20] + [1e-20]*2
# min = [0]*2       + [-4e-3] + [0]*2     + [0]     + [0]*2
# max = [1e-20]*2   + [4e-3]  + [1e-20]*2 + [1e-20] + [1e-20]*2
# min = [-3e-3]*2 + [-4e-3] + [0]*2     + [0]     + [0]*2
# max = [5e-3]*2  + [4e-3]  + [1e-20]*2 + [1e-20] + [1e-20]*2
# min = [-3e-3]*2 + [-4e-3] + [-30e-6]*2 + [-40e-6] + [-0.5e-3]*2
# max = [5e-3]*2  + [4e-3]  + [50e-6]*2  +  [40e-6] + [0.5e-3]*2
# min = [0]*2 + [0] + [0]*2     + [0]     + [0]*2
# max = [1e-4]*2  + [1e-4]  + [1e-20]*2 + [1e-20] + [1e-20]*2
min = [-3e-3]*2 + [-4e-3]
max = [5e-3]*2  + [4e-3]


LOG_S = False                                       # turns on log-sampling
# min_neg_eps_st = [-2.52]*2 + [-2.40] + [-4.52]*2 + [-4.40] + [-3.30]*2 
# max_neg_eps_st = [1]*8
# min_pos_eps_st = [-2.3]*2 + [-2.40] + [-4.30]*2 + [-4.40] + [-3.30]*2
# max_pos_eps_st = [1]*8




# Thickness, material
# t = [200, 250, 300, 350, 400, 450]                  # [mm] thicknesses
# rho = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03]       # [-] reinforcement ratio (to start, rho_x = rho_y)
# CC = [0, 1, 2, 3, 4, 5]                             # [-] six different concrete classes
TWODIM = True
t = [300]
rho = [0.025]
CC = [1]
num_layer = 20                                      # [-]                 
nu_1 = 0                                            # [-]
fsy  = 435                                          # [MPa]
fsu = 470                                           # [MPa]
Es = 205e3                                          # [MPa]
Esh = 8e3                                           # [MPa]
D = 16                                              # [mm]
Dmax = 16                                           # [mm]
s = 200                                             # [mm]
rho_sublayer = True                        


# Collect all parameters for material in one dict
dict_CC.update({'fsy': fsy, 'fsu': fsu, 'Es': Es, 'Esh': Esh, 'D': D, 'Dmax': Dmax, 's': s})
mat_dict = dict_CC

##### Other
save_folder = True
n_samples = int(1000)
material = 3                # 3 = Reinforced concrete




################################## SAMPLER DEFINITION ###########################################

# Sample the data
analytical_sampler = Sampler_utils_vb(E1 = None, nu1=nu_1, E2=None, nu2=None, mat_dict = mat_dict)
if not LOG_S:
    eps_and_t = analytical_sampler.sample(min, max, t1=t, t2=None, num_layer=None, 
                                          num_samples=n_samples, rho=rho, CC=CC, twodim = TWODIM)
    eps = eps_and_t[:, 0:8] 
    t = eps_and_t[:,8:11].reshape(-1,3)                                 # contains [t, rho, CC]
else: 
    # only using min_neg and max_neg as they are larger than min_pos and max_pos.
    eps_and_t = analytical_sampler.sample(min_neg_eps_st, max_neg_eps_st, t1=t, t2=None, num_layer=None, 
                                          num_samples=n_samples, rho=rho, CC=CC)
    sign_eps = np.random.randint(0,2, size = (n_samples,8))
    sign_eps[sign_eps==0] = -1
    eps = eps_and_t[:,0:8]*sign_eps
    t = eps_and_t[:,8:11].reshape((-1,3))

print('Sampled eps and t.')

t_extended = analytical_sampler.extend_material_parameters(t)       # contains [t, rho, CC, Ec, tb0, tb1, ect, ec0, fcp, fct] 

calc_method = 'single'
t0 = time.time()
dict_sampler = analytical_sampler.D_an(np.array(eps), t_extended, num_layers=num_layer, mat = material, calc_meth=calc_method, 
                                       discrete='andreas', rho_sublayer = rho_sublayer)
t1 = time.time()
print('Time for calculating sig, D: ', t1-t0)
sig_a = dict_sampler['sig_a']
D_a = dict_sampler['D_a']

# Reshape data and transfer to numpy
import numpy as np
sig_a = np.squeeze(sig_a.get(), axis=2)
sig_a[sig_a == 0] = 1e-10
if calc_method == 'all':
    eps_rep = np.tile(eps, (1000,1))
    t_rep = np.tile(t, (1000))
else: 
    eps_rep = eps
    t_rep = t_extended.get()
    D_a = D_a.get()

if len(CC) == 1 and not TWODIM: 
      noise_level = 1e-8
      noise = np.random.normal(loc=0.0, scale=noise_level, size=t_rep.shape)
      t_rep_ = t_rep+noise

# Plotting
path_plots = os.path.join(os.getcwd(), '04_Training\\plots')
if len(CC) != 1:
    plot_nathalie(np.concatenate((eps_rep[:,0:8], t_rep[:,0:3]), axis=1), data_in_test = None, save_path = path_plots, tag = 'eps+t_RC')
    plot_nathalie(t_rep, data_in_test = None, save_path = path_plots, tag = 't')
    plot_nathalie(sig_a, data_in_test = None, save_path = path_plots, tag = 'sig')
elif TWODIM: 
     pass
else: 
    plot_nathalie(np.concatenate((eps_rep[:,0:8], t_rep_[:,0:3]), axis=1), data_in_test = None, save_path = path_plots, tag = 'eps+t_RC')
    plot_nathalie(t_rep_, data_in_test = None, save_path = path_plots, tag = 't')
    plot_nathalie(sig_a, data_in_test = None, save_path = path_plots, tag = 'sig')


plots_mike_dataset(eps_rep, eps_rep, sig_a, sig_a, path_plots, tag='test')
histogram(eps_rep, D_a.reshape((-1, 8, 8)), int(D_a.shape[0]), nbins=50, name='De', path=path_plots)


################################ SAVE DATA ################################

if save_folder:
    data_path = os.path.join(os.getcwd(),'04_Training\\data')
    folder_name = f"data_{datetime.now().strftime('%Y%m%d_%H%M')}_fake"
    save_data_path = os.path.join(data_path, folder_name)
    os.makedirs(save_data_path, exist_ok=True)

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

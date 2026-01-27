from data_work import *
import matplotlib.pyplot as plt

# Units should be [N] and [mm] everywhere


# Globally sampled
path = os.getcwd()
# name1 = '04_Training\\data\\data_20241104_2218_case8'
# name1 = '04_Training\\data\\data_20241022_1853_case4'
name1 = '04_Training\\data\\data_20241220_1213_casexx'
path_data1 = os.path.join(path, name1)

new_data_eps_np1 = read_data(path_data1, 'eps')
new_data_sig_np1 = read_data(path_data1, 'sig')
new_data_t_np1 = read_data(path_data1, 't')
new_data_De_np1 = read_data(path_data1, 'De')

# only take first n_glob samples for more fair comparison
n_glob = new_data_t_np1.shape[0]
n_glob = 4000



# Locally sampled
name2 = '04_Training\data\data_20241212_1605_fake'
path_data2 = os.path.join(path, name2)

new_data_eps_np2 = read_data(path_data2, 'eps')
new_data_sig_np2 = read_data(path_data2, 'sig')
new_data_t_np2 = read_data(path_data2, 't')
new_data_De_np2 = read_data(path_data2, 'De')

# Plots for moment and curvature
factor_eps = 10**6
factor_sig = 10**(-3)
data = {
    '0-0': [new_data_eps_np1[:n_glob,3:6]*factor_eps],
    '0-1': [new_data_eps_np2[:n_glob,3:6]*factor_eps],
    '1-0': [new_data_sig_np1[:n_glob,3:6]*factor_sig],
    '1-1': [new_data_sig_np2[:n_glob,3:6]*factor_sig],
    '2-0': [new_data_eps_np1[:n_glob,3:6]*factor_eps,new_data_sig_np1[:n_glob,3:6]*factor_sig],
    '2-1': [new_data_eps_np2[:n_glob,3:6]*factor_eps,new_data_sig_np2[:n_glob,3:6]*factor_sig],
}

# Plots for normal strains and stresses
factor_eps = 10**3
factor_sig = 1
data_n = {
    '0-0': [new_data_eps_np1[:n_glob,0:3]*factor_eps],
    '0-1': [new_data_eps_np2[:n_glob,0:3]*factor_eps],
    '1-0': [new_data_sig_np1[:n_glob,0:3]*factor_sig],
    '1-1': [new_data_sig_np2[:n_glob,0:3]*factor_sig],
    '2-0': [new_data_eps_np1[:n_glob,0:3]*factor_eps,new_data_sig_np1[:n_glob,0:3]*factor_sig],
    '2-1': [new_data_eps_np2[:n_glob,0:3]*factor_eps,new_data_sig_np2[:n_glob,0:3]*factor_sig],
}

# Plots for normal strains and stresses
factor_eps = 10**3
factor_sig = 1
data_v = {
    '0-0': [new_data_eps_np1[:n_glob,6:8]*factor_eps],
    '0-1': [new_data_eps_np2[:n_glob,6:8]*factor_eps],
    '1-0': [new_data_sig_np1[:n_glob,6:8]*factor_sig],
    '1-1': [new_data_sig_np2[:n_glob,6:8]*factor_sig],
    '2-0': [new_data_eps_np1[:n_glob,6:8]*factor_eps,new_data_sig_np1[:n_glob,6:8]*factor_sig],
    '2-1': [new_data_eps_np2[:n_glob,6:8]*factor_eps,new_data_sig_np2[:n_glob,6:8]*factor_sig],
}

print('Data shape global', new_data_t_np1.shape[0])


save_path = os.path.join(path, '04_Training\plots')

# plot_nathalie(np.concatenate((new_data_eps_np1, new_data_t_np1), axis = 1), save_path = save_path, tag = 'eps+t')
plot_paper_comp(data, save_path, ticks = True, number = n_glob, id = 'm')

# plot_paper_comp(data_n, save_path, ticks = True, number = n_glob, id='n')

# plot_paper_comp(data_v, save_path, ticks = True, number = n_glob, id='v')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.stats import norm

plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Helvetica",
        "font.size": 12,
        })

def extend_elements(mat_res_rel: pd.DataFrame, names: str):
    """
    Before: mat_res_rel has amount of rows = amount of simulations 
    After: dist_all_pd has amount of rows = amount of elements in all simulations

    Inputs: 
    mat_res_rel     (pd.DataFrame)      One column of data set (Shape: #Simulations, #elements, 1, 1, 8 )
    names           (str)               For naming of the new data set

    Outputs: 
    dist_all_pd     (pd.DataFrame)      One column of data set (Shape: #Simulations * #elements / Simulation, 8)

    """
    # Find amount of elements
    num_elem = np.zeros((mat_res_rel.shape[0],1))
    num_aux = np.zeros((mat_res_rel.shape[0]+1,1))
    for i in range(mat_res_rel.shape[0]):
        num_elem[i] = mat_res_rel[i].shape[0]
        if i == 0: 
            num_aux[i] = 0
        else:
            num_aux[i] = num_aux[i-1] + num_elem[i-1]
    num_elem = num_elem.astype(np.int32)
    tot_num_elem = int(sum(num_elem))
    num_aux[mat_res_rel.shape[0], 0] = tot_num_elem
    # print(num_aux)
    # print(tot_num_elem)
    
    # Create vector with elements expanded
    dist_all = np.zeros((tot_num_elem,8))
    for i in range(mat_res_rel.shape[0]):
        auxil = mat_res_rel[i]
        dist_all[int(num_aux[i]):int(num_aux[i+1]),:] = auxil[:,0,0,:]
        # print(dist_all.shape)


    # Create pandas dataframe
    if names == 'sig':
        key_names = ['nx', 'ny', 'nxy', 'mx', 'my', 'mxy', 'vx', 'vy']
    elif names == 'eps':
        key_names = ['eps_x', 'eps_y', 'eps_xy', 'chi_x', 'chi_y', 'chi_xy', 'gamma_xz', 'gamma_yz']

    dist_all_pd = pd.DataFrame(dist_all)
    range_num = 8    
    for i in range(range_num):
        dist_all_pd = dist_all_pd.rename({i: key_names[i]}, axis = 'columns')

    return dist_all_pd

def extend_elements_t(mat_res_rel: pd.DataFrame, mat_res_rel_i: pd.DataFrame, key: str):
    '''
    Before: mat_res_rel has amount of rows = amount of simulations 
    After: dist_all_pd has amount of rows = amount of elements in all simulations

    This function is identical to extend_elements, except that it takes mat_res_rel[0].shape = () or (8, 8) as input
    (for key = 't' and key = 'De' respectively)
    --> can read other input parameters that would like to be added to data set

    '''
    # Find amount of elements
    num_elem = np.zeros((mat_res_rel_i.shape[0],1))
    num_aux = np.zeros((mat_res_rel_i.shape[0]+1,1))
    for i in range(mat_res_rel_i.shape[0]):
        num_elem[i] = mat_res_rel_i[i].shape[0]
        if i == 0: 
            num_aux[i] = 0
        else:
            num_aux[i] = num_aux[i-1] + num_elem[i-1]
    num_elem = num_elem.astype(np.int32)
    tot_num_elem = int(sum(num_elem))
    num_aux[mat_res_rel_i.shape[0], 0] = tot_num_elem
    # print(num_aux)
    # print(tot_num_elem)

    if key == 't' or key == 'EN':
        if len(mat_res_rel.shape) > 1:
            size_vec = 10
        else:
            size_vec = 1
    elif key == 'De':
        size_vec = 64

    # Create vector with elements expanded
    dist_all = np.zeros((tot_num_elem,size_vec))
    for i in range(mat_res_rel_i.shape[0]):
        auxil = mat_res_rel[i]
        if key == 't':
            dist_all[int(num_aux[i]):int(num_aux[i+1]),:] = auxil
        elif key == 'De':
            dist_all[int(num_aux[i]):int(num_aux[i+1]),:] = auxil[:,0,0,:,:].reshape((auxil.shape[0],64))
        # print(dist_all.shape)
        if key == 'EN':
            dist_all[int(num_aux[i]):int(num_aux[i+1]),:] = auxil.reshape((auxil.shape[0],size_vec))
            # if i == 88:
            #     print('Dist_all_EN, i= 88', dist_all[int(num_aux[i]):int(num_aux[i+1]),:])


    # Create pandas dataframe
    if key == 't':
        key_name = key
    elif key == 'De':
        indices = np.indices((8,8))
        combined_indices = np.array([f"({i},{j})" for i, j in zip(indices[0].flatten(), indices[1].flatten())])
        key_names = np.array([item + 'De_' for item in combined_indices.flatten()]).reshape(1, 64)

    dist_all_pd = pd.DataFrame(dist_all)
    num_columns = dist_all.shape[1]

    if key == 't':
        dist_all_pd = dist_all_pd.rename({0: key_name}, axis = 'columns')
    elif key == 'De':
        rename_dict = {dist_all_pd.columns[i]: key_names[0,i] for i in range(num_columns)}
        dist_all_pd = dist_all_pd.rename(columns=rename_dict)
    
    return dist_all_pd, num_elem

def binning(data: pd.DataFrame, n_bins:int, path, showplt: bool):
    """
    Creates histogram of data
    Careful: only works for 8 columns at a time

    Inputs: 
    show_plt        (bool)              Whether plots shall be displayed in jupyter notebook (not working...)
    
    Outputs: 
    figures are saved to path
    data_per_bin    (dict)              Rows = 0...8: [nx,ny,nxy,mx,my,mxy,vx,vy]
                                        Columns = 0...n_bins
                                        in each column, different lengths of lists of data points exist
    """
    
    plt.ioff()
    data_np = data.to_numpy()
    index_np = np.array([[0, 1, 2],
                         [3, 4, 5],
                         [6, 7, 7]])

    # Figure initiation
    fig1, axs1 = plt.subplots(3, 3, sharey=True, tight_layout=True, figsize= [6,6])
    keys0 = list(data.keys())[0:8]
    if keys0[0] == 'nx':
        name_sig = np.array([['$n_x$', '$n_y$', '$n_{xy}$'],
                            ['$m_x$', '$m_y$', '$m_{xy}$'],
                            ['$v_x$', '$v_y$', '$v_y$']])
        units_sig = np.array([['$[N/mm]$', '$[N/mm]$', '$[N/mm]$'],
                            ['$[kNmm/mm]$', '$[kNmm/mm]$', '$[kNmm/mm]$'],
                            ['$[N/mm]$', '$[N/mm]$', '$[N/mm]$']])
        keys1 = np.char.add(name_sig, units_sig)
        # keys1 = keys1.reshape((9))[0:8]
    elif keys0[0] == 'eps_x':
        name_eps = np.array([[r'$\varepsilon_x$', r'$\varepsilon_y$', r'$\varepsilon_{xy}$'],
                            [r'$\chi_x$', r'$\chi_y$', r'$\chi_{xy}$'],
                            [r'$\gamma_x$', r'$\gamma_y$', r'$\gamma_y$']])
        units_eps = np.array([[r'$[-]$', r'$[-]$', r'$[-]$'],
                            [r'$[1/mm]$', r'$[1/mm]$', r'$[1/mm]$'],
                            [r'$[-]$', r'$[-]$', r'$[-]$']])
        keys1 = np.char.add(name_eps, units_eps)
        # keys1 = keys1.reshape((9))[0:8]


    # Initialising counting vectors
    n_per_bin = np.zeros((8, n_bins))
    range_bin = np.zeros((8, n_bins+1))
    

    # Figure Plotting

    for i in range(3):
        for j in range(3):
            if i == 2 and j == 2:   
                axs1[i,j].set_title(' ')
            else: 
                axs1[i,j].set_xlabel(keys1[i,j])
                n_per_bin[index_np[i,j],:], range_bin[index_np[i,j],:], patches = axs1[i,j].hist(data_np[:, index_np[i,j]], bins = n_bins)
                if j == 0:
                    axs1[i,j].set_ylabel(r'\textit{Frequency}')
        axs1[-1, -1].axis('off')


    # for i in range(3):
    #     axs1[0, i].set_xlabel(keys1[i])
    #     axs1[1, i].set_xlabel(keys1[i+3])
    #     n_per_bin[i,:], range_bin[i,:], patches = axs1[0, i].hist(data_np[:,i], bins=n_bins)
    #     n_per_bin[i+3,:], range_bin[i+3,:], patches = axs1[1, i].hist(data_np[:,i+3], bins=n_bins)
    # axs1[0, 3].set_xlabel(keys1[6])
    # axs1[1, 3].set_xlabel(keys1[7])    
    # n_per_bin[6,:], range_bin[6,:], patches = axs1[0, 3].hist(data_np[:,6], bins=n_bins)
    # n_per_bin[7,:], range_bin[7,:], patches = axs1[1, 3].hist(data_np[:,7], bins=n_bins)


    if showplt:
        plt.show()
    else:
        plt.ioff()

    if list(data.keys())[0] == 'nx':
        fig1.savefig(os.path.join(path, 'hist_before_'+'sig_g' +'.png'))
    elif list(data.keys())[0] == 'eps_x':
        fig1.savefig(os.path.join(path, 'hist_before_'+ 'eps_g'+'.png'))
    else: 
        fig1.savefig(os.path.join(path, 'hist_before_'+ 't'+'.png'))


    data_per_bin = {}
    index_per_bin = {}
    binlist = {}

    for j in range(8):
        binlist[j] = np.c_[range_bin[j, :-1],range_bin[j, 1:]]
        for i in range(len(binlist[j])):
            if i == len(binlist[j])-1:
                l0 = data_np[:,j]
                bl0 = binlist[j]
                l = l0[(l0 >= bl0[i,0]) & (l0 < bl0[i,1])]
                ind = np.where((l0 >= bl0[i,0]) & (l0 < bl0[i,1]))
                data_per_bin[j,i] = l
                index_per_bin[j,i] = ind
                # print(l)
                # print(ind)
            else:
                l0 = data_np[:,j]
                bl0 = binlist[j]
                l = l0[(l0 >= bl0[i,0]) & (l0 < bl0[i,1])]
                ind = np.where((l0 >= bl0[i,0]) & (l0 < bl0[i,1]))
                # print(l)
                data_per_bin[j,i] = l
                index_per_bin[j,i] = ind
            # print(l)
    # print(data_per_bin)
    # data_per_bin = pd.DataFrame.from_dict(data_per_bin)

    return data_per_bin, index_per_bin, binlist


def mask_data(data_per_bin:dict, index_per_bin:dict, binlist: dict, data:pd.DataFrame, offset_fact:float):
    """
    Returns masks for cutting off data set in cutoff function
    Careful: only works for 8 columns at a time

    Inputs: 
    data_per_bin    (dict)              Data points according to bins
                                        Result of hist function
    data            (pd.DataFrame)      Original (roughly normally distributed) data set
    offset_fact     (np.float64)        Width of offset for uniform distribution
                                        (mu+fac*std)

    Outputs: 
    mask            (np.array)          0 where should be cutoff, 1 where not cutoff

    """
    mean = data.mean(axis=0)
    std = data.std(axis = 0)
    offset_up = mean + offset_fact*std
    offset_down = mean - offset_fact*std
    n_bins = int(len(data_per_bin.keys())/8)
    # print(offset_up)
    # print(binlist)

    n_star = np.zeros((8,1))

    # Find the critical mass n_star: 
    # n_star = amount of data points required for a uniform distribution over
    #          mu +/- offset_fact*std
    n_star_up, n_star_down = np.zeros((8,1)), np.zeros((8,1))

    for j in range(8):
        bl0 = binlist[j]
        # print(bl0)
        for i in range(n_bins):
            if (offset_up[j] >= bl0[i,0]) & (offset_up[j] <bl0[i,1]):
                n_star_up[j,0] = len(data_per_bin[j,i])
            else: 
                pass
            if (offset_down[j] >= bl0[i,0]) & (offset_down[j] <bl0[i,1]):
                n_star_down[j,0] = len(data_per_bin[j,i])
            else: 
                pass
        if n_star_up[j] == 0:
            n_star_up[j] = np.nan
        if n_star_down[j] == 0:
            n_star_down[j] = np.nan

    n_star = np.fmin(n_star_down, n_star_up)
    print(n_star)

    data_np = data.to_numpy()

    mask = np.zeros((data_np.shape))
    
    for j in range(1): 
        for i in range(n_bins):
            if len(data_per_bin[j,i]) > n_star[j]:
                # n_2_many = len(data_per_bin[i,j]) - n_star
                # dp0 = data_per_bin[j,i]
                # dp = dp0[0:int(n_star[j])]      # these are the data points we want to keep (not needed after all)
                di0 = index_per_bin[j,i]          # indices of original data set
                di = di0[0:int(n_star[j])]        # indices in the original data set of data points we want to keep 
                # print('Iteration', j, i, di) 
                for k in di:
                    mask[k,j] = 1                 # data points we want to keep are assigned mask "1", others "0"
            else: 
                pass
    # print(mask)
    return mask


def cutoff_data(data:pd.DataFrame, mask_sig:np.array, mask_eps:np.array, option:int, n_bins:int, path: str):
    """
    Returns novel data set with cut-off where clumped, result should yield a uniform distribution.

    Input: 
    data            (pd.DataFrame)      still the individual data sets for sig, eps
    mask_sig        (np.array)          output from mask_data
    mask_eps        (np.array)          output from mask_data
    option          (int)               1 = for sigma or epsilon (8 variables per row)
                                        2 = for t (single-variable per row)
    n_bins          (int)               Amount of bins, makes most sense if it is the same as chosen above
                                        
    Output: 
    new_data       (np.array)           Uniformly distributed "new" data set
    """
    mask_bool_sig = [v != 1 for v in mask_sig]
    mask_bool_eps = [v != 1 for v in mask_eps]

    if option == 1:
        mask_tot = np.logical_and(mask_bool_eps, mask_bool_sig)
        print('Data shape:', data.shape)
        print('Mask shape:', mask_tot.shape)
        new_data = data.where(mask_tot)
        new_data = new_data.dropna()
    elif option == 2:
        mask_tot = np.logical_and(mask_bool_eps, mask_bool_sig)
        print('Data shape:', data.shape)
        # print('Mask shape:', mask_tot.shape)
        mask_tot_single = mask_tot.all(axis = 1)  # checks over rows: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.all.html 
        mask_tot_single = mask_tot_single.reshape((mask_tot_single.shape[0], 1))
        if data.shape[1] > 1:
            mask_tot_single = np.repeat(mask_tot_single, data.shape[1], axis = 1)
        print('Single Mask shape:', mask_tot_single.shape)
        new_data = data.where(mask_tot_single)
        new_data = new_data.dropna()

    new_data_np = new_data.to_numpy()
    print('New Data shape:', new_data_np.shape)
    index_nd = np.array(([[0, 1, 2],
                          [3, 4, 5],
                          [6, 7, 7]]))

    # Figure Plotting
    if option == 1: 
        fig2, axs2 = plt.subplots(3, 3, sharey=True, tight_layout=True, figsize = [6,6])
        keys0 = list(data.keys())[0:8]
        if keys0[0] == 'nx':
            name_sig = np.array([['$n_x$', '$n_y$', '$n_{xy}$'],
                                ['$m_x$', '$m_y$', '$m_{xy}$'],
                                ['$v_x$', '$v_y$', '$v_y$']])
            units_sig = np.array([['$[N/mm]$', '$[N/mm]$', '$[N/mm]$'],
                                ['$[kNmm/mm]$', '$[kNmm/mm]$', '$[kNmm/mm]$'],
                                ['$[N/mm]$', '$[N/mm]$', '$[N/mm]$']])
            keys1 = np.char.add(name_sig, units_sig)
        elif keys0[0] == 'eps_x':
            name_eps = np.array([[r'$\varepsilon_x$', r'$\varepsilon_y$', r'$\varepsilon_{xy}$'],
                                [r'$\chi_x$', r'$\chi_y$', r'$\chi_{xy}$'],
                                [r'$\gamma_x$', r'$\gamma_y$', r'$\gamma_y$']])
            units_eps = np.array([[r'$[-]$', r'$[-]$', r'$[-]$'],
                                [r'$[1/mm]$', r'$[1/mm]$', r'$[1/mm]$'],
                                [r'$[-]$', r'$[-]$', r'$[-]$']])
            keys1 = np.char.add(name_eps, units_eps)

        for i in range(3):
            for j in range(3):
                if i == 2 and j == 2:   
                    axs2[i,j].set_title(' ')
                else: 
                    axs2[i,j].set_xlabel(keys1[i,j])
                    axs2[i,j].hist(new_data_np[:, index_nd[i,j]], bins = n_bins)
                    if j == 0:
                        axs2[i,j].set_ylabel(r'\textit{Frequency}')
            axs2[-1, -1].axis('off')
    
    elif option == 2 and new_data_np.shape[1] == 1: 
        fig2, axs2 = plt.subplots(1,1, sharey=True, tight_layout=True)
        axs2.hist(new_data_np[:,0])
        if max(new_data_np[:,0])<90:
            axs2.set_xlabel('$Simulation$ $Number$' + ' ' + '$[-]$')
            fig2.savefig(os.path.join(path, 'hist_after_'+ 'SN'+'.png'))
        elif max(new_data_np[:,0])<180:
            axs2.set_xlabel('$t$ $[mm]$')
            fig2.savefig(os.path.join(path, 'hist_after_'+ 't'+'.png'))
        else:
            axs2.set_xlabel('$Element$ $Number$' + ' ' + '$[-]$')
            fig2.savefig(os.path.join(path, 'hist_after_'+ 'EN'+'.png'))
        
    else:
        print('The histogram for De is not plotted as it is too large')
    plt.show()

    if list(data.keys())[0] == 'nx' and option == 1:
        fig2.savefig(os.path.join(path, 'hist_after_'+'sig_g' +'.png'))
    elif option == 1: 
        fig2.savefig(os.path.join(path, 'hist_after_'+ 'eps_g'+'.png'))

    return new_data



def D_an(eps:np.array, t: float):
    '''
    returns analytically calculated sig for given eps and t
    E, nu are assumed constant (as we assume analytical formulation for steel)
    '''
    nu = 0.3
    E= 210000

    D_p = (E/(1+nu**2))*np.array([[1, nu, 0], 
                           [nu, 1, 0], 
                           [0, 0, 0.5 * (1 - nu)]])

    Dse = (5/6)*t*(2*E)/(4*(1+nu))


    D_an_1 = np.hstack([t*D_p, 0*D_p, np.zeros((3,2))])
    D_an_2 = np.hstack([0*D_p, (1/12)*(t**3)*D_p, np.zeros((3,2))])
    D_an_3 = np.hstack([np.zeros((2,3)), np.zeros((2,3)), np.array([[Dse, 0], [0, Dse]])])
    D_analytical = np.vstack([D_an_1, D_an_2, D_an_3])

    sig = np.matmul(D_analytical,eps)

    return sig, D_analytical.reshape((1, 64))




def bar_id(EN_df: pd.DataFrame, SN_df: pd.DataFrame, num_elem: np.array):
    EN_arr = EN_df.to_numpy()
    SN_arr = SN_df.to_numpy()

    # create arrays stating amount of available and amount of cut-off elements
    num_elem = num_elem.reshape((num_elem.shape[0]))
    print('Amount of simulations before cut-off:', num_elem.shape)
    num_elem_avail = np.zeros((int(np.max(SN_arr))+1))
    for i in range(int(np.max(SN_arr))+1):
        mask = (SN_arr.astype(int) == i)
        filtered_EN = EN_arr[mask]
        num_elem_avail[i] = filtered_EN.size
    num_elem_cutoff = num_elem[0:int(np.max(SN_arr))+1] - num_elem_avail
    print('Amount of simulations after cut-off:', num_elem_cutoff.shape)

    simulations = np.arange(0, int(np.max(SN_arr))+1)
    # sim_tuple = tuple(x for x in simulations)
    elem_counts = {
        "Available": num_elem_avail,
        "Cutoff": num_elem_cutoff
    }

    # create plot
    fig, ax = plt.subplots()
    bottom = np.zeros(int(np.max(SN_arr))+1)
    width = 0.5

    for boolean, elem_count in elem_counts.items():
        p = ax.bar(simulations, elem_count, width, label = boolean, bottom = bottom)
        bottom += elem_count

    for i, val in enumerate(simulations):
        if num_elem_avail[i] == num_elem[i]:
            ax.text(val, num_elem_avail[i] + num_elem_cutoff[i] + 1, f"{val}", ha='center', fontsize = 10)
    
    ax.set_xlabel('$Simulation$ $Number$')
    ax.set_ylabel('$Number$ $of$ $Elements$ $per$ $Simulation$')
    ax.legend(loc= 'upper right')
    plt.show()
    return


def bar_range(old_sig_df: pd.DataFrame, old_eps_df: pd.DataFrame, old_SN_df: pd.DataFrame, sig_df: pd.DataFrame, eps_df: pd.DataFrame, SN_df: pd.DataFrame, num_elem: np.array):
    '''
    Creates bar plot in which it is indicated which of the original samples are in the range of the (cut-off) 
    training, evaluation and test data.
    A sample is only counted as in-range if BOTH the 8 sig and 8 eps values are in the ranges of their corresponding new (cut-off) data.
    ''' 
    # SN_arr = SN_df.to_numpy()
    old_SN_arr = old_SN_df.to_numpy()
    old_sig_arr = old_sig_df.to_numpy()
    old_eps_arr = old_eps_df.to_numpy()

    # Determine the ranges:
    min_sig = sig_df.min(axis = 0).to_numpy()
    max_sig = sig_df.max(axis = 0).to_numpy()
    min_eps = eps_df.min(axis = 0).to_numpy()
    max_eps = eps_df.max(axis = 0).to_numpy()
    print(min_sig.shape)

    # create arrays stating amount of available and amount of cut-off elements
    num_elem = num_elem.reshape((num_elem.shape[0]))
    num_elem_inrange = np.zeros((int(np.max(old_SN_arr))+1))
    for i in range(int(np.max(old_SN_arr))+1):
        # filter out the samples corresponding to 1 simulation
        mask1 = (old_SN_arr.astype(int) == i).reshape(old_sig_arr.shape[0])
        filtered_sig = old_sig_arr[mask1]
        filtered_eps = old_eps_arr[mask1]
        # filter out the samples within the new sig & eps ranges
        mask2 = np.zeros((filtered_sig.shape[0]), dtype = bool)
        for j in range(filtered_sig.shape[0]):
            mask2[j] = (np.all(filtered_eps[j,:] >= min_eps) and np.all(filtered_eps[j,:] <= max_eps) and
                        np.all(filtered_sig[j,:] >= min_sig) and np.all(filtered_sig[j,:] <= max_sig))
        filtered_sig2 = filtered_sig[mask2]
        num_elem_inrange[i] = filtered_sig2.shape[0]
    num_elem_outrange = num_elem - num_elem_inrange

    simulations = np.arange(0, int(np.max(old_SN_arr))+1)
    # sim_tuple = tuple(x for x in simulations)
    elem_counts = {
        "in-range": num_elem_inrange,
        "out-of-range": num_elem_outrange
    }

    # create plot
    fig, ax = plt.subplots()
    bottom = np.zeros(int(np.max(old_SN_arr))+1)
    width = 0.5

    for boolean, elem_count in elem_counts.items():
        p = ax.bar(simulations, elem_count, width, label = boolean, bottom = bottom)
        bottom += elem_count

    for i, val in enumerate(simulations):
        if num_elem_inrange[i] == num_elem[i]:
            ax.text(val, num_elem_inrange[i] + num_elem_outrange[i] + 1, f"{val}", ha='center', fontsize = 10)
    
    ax.set_xlabel('$Simulation$ $Number$')
    ax.set_ylabel('$Number$ $of$ $Elements$ $per$ $Simulation$')
    ax.legend(loc= 'upper right')
    plt.show()
    return


def boxplot_compar(sig_old: pd.DataFrame, sig_new: pd.DataFrame, indx: int, id: str):
    data = {}
    df1 = sig_old.iloc[:,indx]
    df2 = sig_new.iloc[:,indx]
    data = [df1, df2]
    if id == 'sig':
        name = np.array(['$n_x$', '$n_y$', '$n_{xy}$', '$m_x$', '$m_y$', '$m_{xy}$', '$v_x$', '$v_y$', '$v_y$'])
    elif id == 'eps':
        name = np.array([r'$\varepsilon_x$', r'$\varepsilon_y$', r'$\varepsilon_{xy}$', r'$\chi_x$', r'$\chi_y$', r'$\chi_{xy}$', r'$\gamma_x$', r'$\gamma_y$', r'$\gamma_y$'])
    labels = np.array([np.char.add(name[indx], ', old'), np.char.add(name[indx], ', new')])
    plt.boxplot(data, labels=labels)
    plt.ylabel(name[indx])
    plt.show()
    return
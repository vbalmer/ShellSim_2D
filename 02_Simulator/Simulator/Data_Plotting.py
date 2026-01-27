import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from math import nan
import os

plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Helvetica",
        "font.size": 12,
        })

def countDistinct(arr, n):
    # code copied from here: https://www.geeksforgeeks.org/count-distinct-elements-in-an-array/ 
    # and adapted with rounding

    res = 1
 
    # Pick all elements one by one
    for i in range(1, n):
        j = 0
        for j in range(i):
            if (round(arr[i],2) == round(arr[j],2)):
                print(round(arr[i],2))
                break
 
        # If not printed earlier, then print it
        if (i == j + 1):
            res += 1
 
    return res

def single_analysis(mat_res_row: pd.DataFrame, var: str, var1: str, num_ele: np.array, single: bool):
    '''
    Creates a heatmap of desired simulation outputs on geometrical system
    [Not in use]
    mat_res_row     (pd.df)     One row of the desired plate to be analysed (including its keys!)
    var             (str)       Key name of desired variable to be plotted as heatmap [either sig_g or eps_g]
    var1            (str)       Variable within sig_g, eps_g of interest (for single plot)
    num_ele         (arr)       Amount of elements in x- and y-direction
    single          (bool)      Just plot a single diagram
    '''

    ## Preprocessing the data

    res_s = np.squeeze(mat_res_row.loc['sig_g'])
    res_e = np.squeeze(mat_res_row.loc['eps_g'])
    res_c = mat_res_row.loc['COORD_c']
    res_c = res_c[0]
    # print(res_c)
    # data for x- and y-axis
    x_coord = res_c[:,0]
    y_coord = res_c[:,1]
    x_aux = np.linspace(min(x_coord), max(x_coord),num_ele[0])
    y_aux = np.linspace(min(y_coord), max(y_coord),num_ele[1])


    # data for color (z-axis)
    # Picking the required 
    mask_sigma = np.array([['n_x', 'n_y', 'n_xy', 'm_x', 'm_y', 'm_xy', 'v_x', 'v_y'],
                  [0, 1, 2, 3, 4, 5, 6, 7]])
    mask_eps = np.array([['eps_x', 'eps_y', 'eps_xy', 'chi_x', 'chi_y', 'chi_xy', 'gam_x', 'gam_y'],
                  [0, 1, 2, 3, 4, 5, 6, 7]])
    
    for j in range(mask_sigma.shape[1]):
        if var1 == mask_sigma[0,j]:
            index = int(mask_sigma[1,j])
        if var1 == mask_eps[0,j]:
            index = int(mask_eps[1,j])

    if var == 'sig_g':
        z_value = res_s[:,index]
        # print(z_value_all)
    else: 
        z_value = res_e[:,index]
    # not working for different mesh sizes yet!
    z_value_lay = np.zeros((num_ele[0], num_ele[1]))
    z_value_all = np.zeros((num_ele[0], num_ele[1],8))
    z_value_all_resh = np.zeros((num_ele[0], num_ele[1],2,4))
    for i in range(num_ele[1]):
        z_value_x = z_value[0:num_ele[0]]
        z_value_y = z_value[num_ele[0]:]
    z_value_lay = np.array([z_value_x, z_value_y])
    # print(z_value_lay)


    


    ## Plotting (with contour plot)
    # https://matplotlib.org/stable/gallery/images_contours_and_fields/contour_image.html#sphx-glr-gallery-images-contours-and-fields-contour-image-py 
    # Interpolation schemes: https://matplotlib.org/stable/gallery/images_contours_and_fields/interpolation_methods.html#sphx-glr-gallery-images-contours-and-fields-interpolation-methods-py
    
    # Axes labels: 
    mask_sigma1 = np.array([['n_x', 'n_y', 'n_xy', 'v_x'],
                  ['m_x', 'm_y', 'm_xy','v_y']])
    mask_eps1 = np.array([['eps_x', 'eps_y', 'eps_xy', 'gamma_xz'],
                  ['chi_x', 'chi_y', 'chi_xy','gamma_yz']])

    
    if single:
        X, Y = np.meshgrid(x_aux,y_aux)
        Z = z_value_lay

        norm = cm.colors.Normalize(vmax=abs(Z).max(), vmin=-abs(Z).max())
        cmap = cm.PRGn

        fig, axs = plt.subplots(nrows=1, ncols=1)
        fig.subplots_adjust(hspace=0.3)
        # axs = _axs.flatten()

        cset1 = axs.contourf(X, Y, Z, norm=norm, cmap=cmap)
        fig.colorbar(cset1, ax=axs)

        if var == 'sig_g':
            axs.set_title(mask_sigma[0,index])
        else: 
            axs.set_title(mask_eps[0,index])
        axs.set_xlabel('x')
        axs.set_ylabel('y')
    
    else: 
        pass
        # fig, axs = plt.subplots(2, 4)
        # fig.subplots_adjust(hspace=0.3)
        # X, Y = np.meshgrid(x_aux,y_aux)
        # Z = z_value_lay
        # cset = np.zeros((2,4))
        # for i in range(2):
        #     for j in range(4):
        #         cset[i,j] = axs[i,j].contourf(X, Y, Z[i,j], norm=norm, cmap=cmap)
        #         fig.colorbar(cset1, ax=_axs)

        # if var == 'sig_g':
        #     axs[i,j].set_xlabel(mask_sigma1[i,j])
        # else:
        #     axs[i,j].set_xlabel(mask_eps1[i,j])

        # axs.set_xlabel('x')
        # axs.set_ylabel('y')

    plt.show()

    return





def statistics(mat_res_col: pd.DataFrame, var: str, id: str, n_bins:int, path, single: bool):
    '''
    Creates a histogram for all data in the given data set (over all nodes)
    
    Inputs: 
    mat_res_col:    (pd.DataFrame)      Data Column of Interest
    var:            (str)               Variable within data column of interest
    id:             (str)               either 'eps_g' or 'sig_g' for plotting
    n_bins:         (int)               Amount of bins for histogram, default to 20
    single:         (bool)              Just plot a single diagram

    '''
    # Preprocessing the data
    # print(mat_res_col)
    mask_sigma = np.array([['$n_x$', '$n_y$', '$n_{xy}$','$m_x$', '$m_y$', '$m_{xy}$','$v_x$', '$v_y$'],
                  [0, 1, 2, 3, 4, 5, 6, 7]])
    mask_eps = np.array([[r'\varepsilon_x', r'\varepsilon_y', r'\varepsilon_{xy}', r'\chi_x', r'\chi_y', r'chi_{xy}', r'\gamma_x', r'\gamma_y'],
                  [0, 1, 2, 3, 4, 5, 6, 7]])
    mask_sigma1 = np.array([['$n_x$', '$n_y$', '$n_{xy}$'],
                            ['$m_x$', '$m_y$', '$m_{xy}$'],
                            ['$v_x$', '$v_y$', '$v_y$']])
    mask_eps1 = np.array([[r'$\varepsilon_x$', r'$\varepsilon_y$', r'$\varepsilon_{xy}$'],
                            [r'$\chi_x$', r'$\chi_y$', r'$\chi_{xy}$'],
                            [r'$\gamma_x$', r'$\gamma_y$', r'$\gamma_y$']])
    
    units_sig = np.array([['$[N/mm]$', '$[N/mm]$', '$[N/mm]$'],
                            ['$[Nmm/mm]$', '$[Nmm/mm]$', '$[Nmm/mm]$'],
                            ['$[N/mm]$', '$[N/mm]$', '$[N/mm]$']])
    units_eps = np.array([['$[-]$', '$[-]$', '$[-]$'],
                            ['$[1/mm]$', '$[1/mm]$', '$[1/mm]$'],
                            ['$[-]$', '$[-]$', '$[-]$']])

    # mask_eps1 = np.array([['eps_x', 'eps_y', 'eps_xy', 'gamma_xz'],
    #               ['chi_x', 'chi_y', 'chi_xy','gamma_yz']])

    mask_sigma_t = np.array([['n_x', 'n_y', 'n_xy', 'm_x', 'm_y', 'm_xy', 'v_x', 'v_y'],
                           [0, 1, 2, 3, 4, 5, 6, 7]])
    mask_eps_t = np.array([['eps_x', 'eps_y', 'eps_xy', 'chi_x', 'chi_y', 'chi_xy', 'gam_x', 'gam_y'],
                           [0, 1, 2, 3, 4, 5, 6, 7]])
    
    for j in range(mask_sigma_t.shape[1]):
        if id == 'sig_g':
            if var == mask_sigma_t[0,j]:
                index = int(mask_sigma_t[1,j])
            sig = True
        elif id == 'eps_g':
            if var == mask_eps_t[0,j]:
                index = int(mask_eps_t[1,j])
            sig = False


    # collecting relevant data and reshaping  
    num_elem = np.zeros((mat_res_col.shape[0],1))
    num_aux = np.zeros((mat_res_col.shape[0]+1,1))
    for i in range(mat_res_col.shape[0]):
        num_elem[i] = mat_res_col[i].shape[0]
        if i == 0: 
            num_aux[i] = 0
        else:
            num_aux[i] = num_aux[i-1] + num_elem[i-1]
    num_elem = num_elem.astype(np.int32)
    # print(num_elem)
    tot_num_elem = int(sum(num_elem))
    num_aux[mat_res_col.shape[0], 0] = tot_num_elem
    # print(num_aux)
    # print(tot_num_elem)
    
    dist_x = np.zeros((tot_num_elem,1))
    dist_all = np.zeros((tot_num_elem,8))
    dist_all_resh = np.zeros((tot_num_elem,3,3))
    # print(dist_x.shape)
    # print(dist_all_resh.shape)
    
    for i in range(mat_res_col.shape[0]):
        auxil = mat_res_col[i]

        dist_x[int(num_aux[i]):int(num_aux[i+1]),0] = auxil[:,0,0,index]
        # print(dist_x[int(num_aux[i]):int(num_aux[i+1]),0].shape)

        dist_all[int(num_aux[i]):int(num_aux[i+1]),:] = auxil[:,0,0,:]
        # print(dist_all.shape)
       
    for i in range(tot_num_elem):    
        dist_all_resh[i,0:2,0:3] = dist_all[i,0:6].reshape(2,3)
        dist_all_resh[i,2,0:3] = np.array([dist_all[i,6], dist_all[i,7], dist_all[i,7]])
    
    print(dist_all_resh.shape)
    

    ## Plotting
    if single:
        fig, axs = plt.subplots(1,1, sharey=True, tight_layout=True)
        if sig:
            axs.set_xlabel(mask_sigma[0,index])
            axs.hist(dist_x, bins=n_bins)
            print(dist_x.shape)
        else:
            axs.set_xlabel(mask_eps[0,index])
            axs.hist(dist_x, bins=n_bins)
        axs.set_ylabel(r'\it{Frequency}')
        

    else:
        fig, axs = plt.subplots(3, 3, sharey=True, tight_layout=True, figsize = [7, 7])
        for i in range(3):
            for j in range(3):
                if j==2 and i ==2:
                    axs[i,j].set_title(' ')
                else: 
                    if sig:
                        axs[i,j].set_xlabel(mask_sigma1[i,j]+' '+units_sig[i,j])
                        axs[i,j].hist(dist_all_resh[:,i,j], bins=n_bins)
                    elif not sig:
                        axs[i,j].set_xlabel(mask_eps1[i,j]+' '+units_eps[i,j])
                        axs[i,j].hist(dist_all_resh[:,i,j], bins=n_bins)
                    axs[i,j].set_ylabel(r'\it{Frequency}')
        axs[-1, -1].axis('off')
        axs = plt.gca()
        axs.axis('square')
    plt.show()
    
    if single == False:
        if sig:
            fig.savefig(os.path.join(path, 'hist_all_'+'sig_g' +'.png'))
        else:
            fig.savefig(os.path.join(path, 'hist_all_'+ 'eps_g'+'.png'))

    else: 
        fig.savefig(os.path.join(path, 'hist_'+ var +'.png'))
    return

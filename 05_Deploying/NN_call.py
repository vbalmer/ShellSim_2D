import os
import pickle
import torch
import pandas as pd
import numpy as np
# from Test.FFNN_class_light import *
# from Test.data_work_depl import *
from FFNN_class_light import *
from data_work_depl import *
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from lightning.pytorch import seed_everything
seed_everything(42)

'''
This file contains copies of the functions "transf_units" from data_work.py,
"predict_sig_D", "inp_out_plt" and "D_an" from call_light.py (in 04_Training)
Please adjust functions there if required (only difference here: no logging to wandb).
'''

def transf_units(vec:np.array, id:str, forward:bool, linel = True):
    '''
    Transforms the units for input from simulation to training and back
    vec:        (np.array)          Vector to be transformed
    id:         (str)               Identifier 'sig', 'D' or 'eps-t' depending on the desired transformation of vec
                                    Expected shapes: sig: (n,8), eps-t: (n,9), D: (n, 8,8)
    forward:    (bool)              If true: Transformation is forward (i.e. from N, mm to MN, cm)
                                    If false: Transformation is backward (i.e. from MN, cm to N, mm)
    '''

    if id == 'sig':
        sig_t = np.zeros_like(vec)
        if forward:
            sig_t[:, 0:3] = vec[:, 0:3]*(10**(-6))*(10**(1))
            sig_t[:, 3:6] = vec[:, 3:6]*(10**(-6))*1
            sig_t[:, 6:8] = vec[:, 6:8]*(10**(-6))*(10**(1))
            out_vec = sig_t
        else: 
            sig_t[:, 0:3] = vec[:, 0:3]*(10**(6))*(10**(-1))
            sig_t[:, 3:6] = vec[:, 3:6]*(10**(6))*1
            sig_t[:, 6:8] = vec[:, 6:8]*(10**(6))*(10**(-1))
            out_vec = sig_t

    elif id == 'eps-t':
        eps_t = np.zeros_like(vec)
        if forward:
            eps_t[:, 0:3] = vec[:, 0:3]*1
            eps_t[:, 3:6] = vec[:, 3:6]*(10**1)
            eps_t[:, 6:8] = vec[:, 6:8]*1
            eps_t[:,8:] = vec[:,8:]
            out_vec = eps_t
        else:
            eps_t[:, 0:3] = vec[:, 0:3]*1
            eps_t[:, 3:6] = vec[:, 3:6]*(10**(-1))
            eps_t[:, 6:8] = vec[:, 6:8]*1
            eps_t[:,8:] = vec[:,8:]
            out_vec = eps_t
    
    elif id == 'D':

        D_t = np.zeros_like(vec)
        if linel:
            if forward:
                D_t[:, 0:3, 0:3] = vec[:, 0:3, 0:3]*(10**(-6))*(10**(1))
                D_t[:,3:6, 3:6] = vec[:, 3:6, 3:6]*(10**(-6))*(10**(-1))
                D_t[:, 6:8, 6:8] = vec[:, 6:8, 6:8]*(10**(-6))*(10**(1))
                out_vec = D_t
            else:
                D_t[:, 0:3, 0:3] = vec[:, 0:3, 0:3]*(10**(6))*(10**(-1))
                D_t[:,3:6, 3:6] = vec[:, 3:6, 3:6]*(10**(6))*(10**(1))
                D_t[:, 6:8, 6:8] = vec[:, 6:8, 6:8]*(10**(6))*(10**(-1))
                out_vec = D_t
        else:
            if forward:
                D_t[:, 0:3, 0:3] = vec[:, 0:3, 0:3]*(10**(-6))*(10**(1))
                D_t[:,3:6, 3:6] = vec[:, 3:6, 3:6]*(10**(-6))*(10**(-1))
                D_t[:, 6:8, 6:8] = vec[:, 6:8, 6:8]*(10**(-6))*(10**(1))
                D_t[:,0:3, 3:6] = vec[:, 0:3, 3:6]*(10**(-6))
                D_t[:,3:6, 0:3] = vec[:, 3:6, 0:3]*(10**(-6))
                out_vec = D_t
            else: 
                D_t[:, 0:3, 0:3] = vec[:, 0:3, 0:3]*(10**(6))*(10**(-1))
                D_t[:,3:6, 3:6] = vec[:, 3:6, 3:6]*(10**(6))*(10**(1))
                D_t[:, 6:8, 6:8] = vec[:, 6:8, 6:8]*(10**(6))*(10**(-1))
                D_t[:,0:3, 3:6] = vec[:, 0:3, 3:6]*(10**(6))
                D_t[:,3:6, 0:3] = vec[:, 3:6, 0:3]*(10**(6))
                out_vec = D_t

    return out_vec


def predict_sig_D(random_eps_h:np.array, path_data_name:str, lit_model_path: str, stats: str, transf_type: str, predict = 'sig', sc=False, model_dim = 'ALLDIM'):
    
    '''
    Predicts output sig_h and D, based on random_eps_h input and the trained model given in path_data_name

    random_eps_h    (np.array)      input vector: 8 eps + t --> shape: (1,9)
    path_data_name  (str)           path to data for trained model
    lit_model_path  (str)           path to trained model (lightning)
    stats           (str)           statistics with which normalisation is carried out, can be either 'train' or 'test'
    transf_type     (str-list)      type of transformation for normalisation
    predict         (str)           can be either 'sig', 'D' or 'Dd' -- use this function to predict sig, D (derivative of sig-NN) or Dd (direct D-NN)
    sc              (bool)          True: Scaling included in normalisation, False: No scaling included.
    model_dim       (str)           can be 'ONEDIM_x', 'ONEDIM_y', 'TWODIM' or 'ALLDIM' depending on the network architecture.


    Output: 
    mat_pred         (dict)         Containing 'D_t', 'D_pred' and 'sig_h'
                                    if predict = 'sig': 'D_t' and 'D_pred' == None
                                    if predict = 'D': 'sig_h' == None
    '''

    random_sig_h = np.zeros((1,8))

    # Load statistics information
    with open(os.path.join(path_data_name, 'mat_data_stats.pkl'),'rb') as handle:
        mat_data_stats = pickle.load(handle)
    with open(os.path.join(path_data_name, 'inp.pkl'),'rb') as handle:
        inp = pickle.load(handle)
    
    if stats == 'train':
        stats_y = mat_data_stats['stats_y_train']
        stats_y_sig = {key: value[0:8] for key, value in stats_y.items()}
        stats_X = mat_data_stats['stats_X_train']
        if model_dim != 'ALLDIM':
            print(f"Using reduced model size, according to {model_dim}. Expanding statistics, inputs and predictions with zeros.")
            mat_data_stats = add_zerovals_stats(mat_data_stats, dim = model_dim)
    elif stats =='test':
        UserWarning('Please use train statistics, not test statistics. Aborting calculation.')
        stats_y = mat_data_stats['stats_y_test']
        stats_y_sig = {key: value[0:8] for key, value in stats_y.items()}
        stats_X = mat_data_stats['stats_X_test']

    # Transform into normalised coordinates
    inp_shape = mat_data_stats['stats_X_train']['std'].shape[0]
    out_shape = mat_data_stats['stats_y_train']['std'].shape[0]

    if transf_type == 'mixed':
        transf_type_list_x = ['x-std']*3+['x-range']*3+['x-std']*(inp_shape-6)
        transf_type_list_y = ['y-std']*3+['y-range']*3+['y-std']*66
    elif transf_type == 'st-stitched':
        transf_type_list_x = ['x-std']*inp_shape
        transf_type_list_y = ['y-std']*8+['y-st-stitched']*64
    else: 
        transf_type_list_x = ['x-'+transf_type]*inp_shape
        transf_type_list_y = ['y-'+transf_type]*out_shape

    X_depl_t = transform_data(random_eps_h, mat_data_stats, forward = True, type=transf_type_list_x, sc = sc)
    if X_depl_t.any() > 10 or X_depl_t.any() < -10:
        raise Exception("The normalisation yielded instable results. Please check input data")

    # Create dataset in correct format
    X_depl_tt = torch.from_numpy(X_depl_t)
    X_depl_tt = X_depl_tt.type(torch.float32)
    Y_depl_tt = torch.from_numpy(random_sig_h)
    Y_depl_tt = Y_depl_tt.type(torch.float32)

    # Load the trained model
    if inp['simple_m']:
        if not inp['DeepONet'] and not inp['MoE']:
            model = FFNN(inp)
        elif inp['DeepONet']: 
            model = DeepONet_vb(inp)
        elif inp['MoE']: 
            model = MoE(inp)
        model.load_state_dict(torch.load(lit_model_path, map_location=device))
        model.eval()
        model_D = model
    else:
        if not inp['Split_Net']:
            model_sig = LitFFNN.load_from_checkpoint(lit_model_path)
            model_D = LitFFNN.load_from_checkpoint(lit_model_path['D'])
            model_sig.eval()
            model_D.eval()
        else: 
            model_m = LitFFNN.load_from_checkpoint(lit_model_path['m'])
            model_b = LitFFNN.load_from_checkpoint(lit_model_path['b'])
            model_s = LitFFNN.load_from_checkpoint(lit_model_path['s'])
            model_D = LitFFNN.load_from_checkpoint(lit_model_path['D'])
            model_m.eval()
            model_b.eval()
            model_s.eval()
            model_D.eval()


    # Make predictions of sig_generalised
    if predict == 'sig':
        if inp['simple_m']:
            model.to(device)
            if not inp['DeepONet'] and model_dim == 'ALLDIM':
                predictions_t = model(torch.Tensor(X_depl_t).to(device))
            elif model_dim != 'ALLDIM': 
                predictions_t = predict_with_zerovals(X_depl_t, model, None, dim = model_dim, tag = 'sig')
            elif inp['DeepONet']:
                predictions_t = model(X_depl_tt[:,0:8].to(device), X_depl_tt[:,8].reshape((-1,1)).to(device))          
        else:
            if not inp['Split_Net']:
                predictions_t = model_sig(X_depl_tt.to(device))
            else: 
                predictions_t_m = model_m(torch.Tensor(np.hstack((X_depl_t[:,0:3], X_depl_t[:,8].reshape(-1, 1)))).to(device))
                predictions_t_b = model_b(torch.Tensor(np.hstack((X_depl_t[:,3:6], X_depl_t[:,8].reshape(-1, 1)))).to(device))
                predictions_t_s = model_s(torch.Tensor(np.hstack((X_depl_t[:,6:8], X_depl_t[:,8].reshape(-1, 1)))).to(device))
                predictions_t = torch.hstack((predictions_t_m, predictions_t_b, predictions_t_s))
        # Transform back sigma
        predictions_sig = transform_data(predictions_t.cpu().detach().numpy(), mat_data_stats, forward = False, type = transf_type_list_y, sc=sc)
    else: 
        # only predict D --> sig = None
        predictions_sig = None


    # Make predictions of D-matrix
    if predict == 'D' or predict == 'Dd':
        if inp['DeepONet']:
            # TODO!
            D_pred = np.zeros((X_depl_t.shape[0],8,8))
            D_t = np.zeros((X_depl_t.shape[0],8,8))
        elif predict == 'Dd' and model_dim == 'ALLDIM':
            # Prediction of D with separate direct NN
            D_pred, D_t = np.zeros((X_depl_t.shape[0], 8, 8)), np.zeros((X_depl_t.shape[0], 8, 8))
            D_pred_t = model(torch.Tensor(X_depl_t).to(device))
            D_t[:,:6,:6] = D_pred_t[:,:36].cpu().detach().numpy().reshape((-1,6,6))
            D_t[:,6,6] = D_pred_t[:,36].cpu().detach().numpy().reshape((-1,1))
            D_t[:,7,7] = D_pred_t[:,37].cpu().detach().numpy().reshape((-1,1))

            D_pred_ = transform_data(D_pred_t.cpu().detach().numpy(), mat_data_stats, forward = False, type = transf_type_list_y, sc=sc)
            D_pred[:,:6,:6] = D_pred_[:,:36].reshape((-1,6,6))
            D_pred[:,6,6] = D_pred_[:,36].reshape((-1,1))
            D_pred[:,7,7] = D_pred_[:,37].reshape((-1,1))

        elif predict == 'D': 
            D_pred = np.zeros((X_depl_t.shape[0], 8, 8))
            if model_dim == 'ALLDIM': 
                # prediction of D with derivativation of sig-NN
                X_depl_tt.requires_grad = True
                J = vmap(torch.func.jacrev(model_D.forward), randomness='different')(X_depl_tt)
                # J = torch.cat([torch.autograd.functional.jacobian(model_D, X_depl_tt[i:i+1], create_graph = True) for i in range(len(X_depl_tt))], dim=0)[:,:,0,:]
                D_t = J.cpu().detach().numpy()
                if len(D_t.shape) > 3:
                    D_t = np.squeeze(D_t, axis = 1)
                D_t = D_t[:, :8, :8]
            elif model_dim != 'ALLDIM': 
                D_t = predict_with_zerovals(X_depl_t, None, model_D, dim = model_dim, tag = 'D')


            # Transform back D
            for idx in range(X_depl_t.shape[0]):
                D_t_i = D_t[idx,:,:].reshape((1,64))
                added_transform = np.concatenate((np.zeros((1,8)), D_t_i), axis=1)
                added_transform_ = transform_data(added_transform, mat_data_stats, forward = False, type = transf_type_list_y, sc=sc)
                D_pred_i = added_transform_[:,8:]
                D_pred[idx,:,:] = D_pred_i.reshape((8,8))
            
    
        # Overriding certain values to be zero: 
        D_pred[:,6:8,0:6] = np.zeros((1,2,6))
        D_pred[:,0:6,6:8] = np.zeros((1,6,2))
    
    else: 
        # only predict sig --> D = None
        D_pred = None
        D_t = None

    # Collect relevant data
    mat_pred = {
        'D_pred': D_pred,
        'D_t': D_t,
        'sig_h': predictions_sig
    }

    return mat_pred



def D_an(eps:np.array, t: float):
    '''
    returns analytically calculated sig for given eps and t  (linear elastic)
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

    sig_analytical = np.matmul(D_analytical,eps)

    mat_analytical = {
        'D_a': D_analytical,
        'sig_a': sig_analytical
    }

    return mat_analytical


############## transfer from 1D, 2D to 8D ############################

def add_zerovals_stats(mat_data_stats, dim, small_value = 1e-20):
    '''
    adds zero-values to statistics of NNs that are not the desired shape required for predicting / transforming data
    mat_data_stats  (dict)      if ONEDIM: contains only 1+3 values in x and 1+1 values in y
                                if TWODIM: contains only 3+3 values in x and 3+9 values in y
    dim             (str)       'ONEDIM_x', 'ONEDIM_y', 'TWODIM'
    small_value     (float)     small value that represents zero
    '''

    if dim == 'ONEDIM_x':
        stats_new = {}
        for key in ['stats_y_train', 'stats_y_test']:
            stats_new[key] = {}
            for subkey in mat_data_stats[key].keys():
                a = mat_data_stats[key][subkey][:1]
                b = mat_data_stats[key][subkey][1:]
                stats_new[key][subkey] = np.concatenate((a, small_value*np.ones(7,), b, small_value*np.ones(63,)), axis = 0)
        for key in ['stats_X_train', 'stats_X_test']:
            stats_new[key] = {}
            for subkey in mat_data_stats[key].keys():
                a = mat_data_stats[key][subkey][:1]
                b = mat_data_stats[key][subkey][1:]
                stats_new[key][subkey] = np.concatenate((a, small_value*np.ones(7,), b), axis = 0)
        stats_new
    elif dim == 'ONEDIM_y':
        stats_new = {}
        for key in ['stats_y_train', 'stats_y_test']:
            stats_new[key] = {}
            for subkey in mat_data_stats[key].keys():
                a = mat_data_stats[key][subkey][:1]
                b = mat_data_stats[key][subkey][1:]
                stats_new[key][subkey] = np.concatenate((small_value*np.ones(1,), a, small_value*np.ones(6,), small_value*np.ones(9,), b, small_value*np.ones(54,)), axis = 0)
        for key in ['stats_X_train', 'stats_X_test']:
            stats_new[key] = {}
            for subkey in mat_data_stats[key].keys():
                a = mat_data_stats[key][subkey][:1]
                b = mat_data_stats[key][subkey][1:]
                stats_new[key][subkey] = np.concatenate((small_value*np.ones(1,), a, small_value*np.ones(6,), b), axis = 0)
        stats_new
    elif dim == 'TWODIM':
        stats_new = {}
        for key in ['stats_y_train', 'stats_y_test']:
            stats_new[key] = {}
            for subkey in mat_data_stats[key].keys():
                a = mat_data_stats[key][subkey][:3]
                b = mat_data_stats[key][subkey][3:]
                stats_new[key][subkey] = np.concatenate((a, small_value*np.ones(5,), 
                                                         b[:3], small_value*np.ones(5,),
                                                         b[3:6], small_value*np.ones(5,),
                                                         b[6:], small_value*np.ones(40+5,),), axis = 0)
        for key in ['stats_X_train', 'stats_X_test']:
            stats_new[key] = {}
            for subkey in mat_data_stats[key].keys():
                a = mat_data_stats[key][subkey][:3]
                b = mat_data_stats[key][subkey][3:]
                stats_new[key][subkey] = np.concatenate((a, small_value*np.ones(5,), b), axis = 0)
        stats_new

    else: 
        raise UserWarning('Please do not use this function if model_dim = ALL_DIM.')

    return stats_new


def predict_with_zerovals(X_depl_t, model, model_D, dim, tag, small_value = 1e-20):
    '''
    Carry out prediction and add zero-values in the correct location for a model that is smaller than standard size
    X_depl          (dict)      if ONEDIM: shape = (n, 1+3)
                                if TWODIM: shape = (n, 1+3)
    model           (nn.Module) model for predicting sigma (can be None when predicting D)
    model_D         (nn.Module) model for predicting D (can be None when predicting sig)
    dim             (str)       'ONEDIM_x', 'ONEDIM_y', 'TWODIM'
    small_value     (float)     small value that represents zero
    tag             (str)       'sig' predict sig
                                'D': predict D
    '''

    # for predicting the stresses sigma
    if tag == 'sig':
        if dim == 'ONEDIM_x':
            predictions_t = model(torch.Tensor(np.concatenate((X_depl_t[:,:1], X_depl_t[:,8:]), axis = 1)).to(device))
            predictions_t = torch.cat((predictions_t, small_value*torch.ones((predictions_t.shape[0], 7)).to(device)), axis = 1)
        elif dim == 'ONEDIM_y':
            predictions_t = model(torch.Tensor(np.concatenate((X_depl_t[:,1:2], X_depl_t[:,8:]), axis = 1)).to(device))
            predictions_t = torch.cat((small_value*torch.ones((predictions_t.shape[0], 1)).to(device), predictions_t, small_value*torch.ones((predictions_t.shape[0], 6)).to(device)), axis = 1)
        elif dim == 'TWODIM':
            predictions_t = model(torch.Tensor(np.concatenate((X_depl_t[:,0:3], X_depl_t[:,8:]), axis = 1)).to(device))
            predictions_t = torch.cat((predictions_t, small_value*torch.ones((predictions_t.shape[0], 5)).to(device)), axis = 1)
        predictions_depl = predictions_t

    # for predicting the stiffness D
    elif tag == 'D':
        if dim == 'ONEDIM_x':
            X_depl_tt = torch.from_numpy(X_depl_t)
            X_depl_tt = X_depl_tt.type(torch.float32)
            X_depl_tt.requires_grad = True
            J = vmap(torch.func.jacrev(model_D.forward), randomness='different')(torch.cat((X_depl_tt[:,:1], X_depl_tt[:,8:]), axis = 1))
            # J = torch.cat([torch.autograd.functional.jacobian(model_D, X_depl_tt[i:i+1], create_graph = True) for i in range(len(X_depl_tt))], dim=0)[:,:,0,:]
            if len(J.shape) < 4:
                J = J.unsqueeze(2)
            D_t_ = J.cpu().detach().numpy()
            D_t = small_value*np.ones((D_t_.shape[0],8,8))
            D_t[:,0,0] = D_t_[:,0,0,0]
            predictions_depl = D_t
        elif dim == 'ONEDIM_y':
            X_depl_tt = torch.from_numpy(X_depl_t)
            X_depl_tt = X_depl_tt.type(torch.float32)
            X_depl_tt.requires_grad = True
            J = vmap(torch.func.jacrev(model_D.forward), randomness='different')(torch.cat((X_depl_tt[:,1:2], X_depl_tt[:,8:]), axis = 1))
            # J = torch.cat([torch.autograd.functional.jacobian(model_D, X_depl_tt[i:i+1], create_graph = True) for i in range(len(X_depl_tt))], dim=0)[:,:,0,:]
            if len(J.shape) < 4:
                J = J.unsqueeze(2)
            D_t_ = J.cpu().detach().numpy()
            D_t = small_value*np.ones((D_t_.shape[0],8,8))
            D_t[:,1,1] = D_t_[:,0,0,0]
            predictions_depl = D_t
        elif dim == 'TWODIM': 
            X_depl_tt = torch.from_numpy(X_depl_t)
            X_depl_tt = X_depl_tt.type(torch.float32)
            X_depl_tt.requires_grad = True
            J = vmap(torch.func.jacrev(model_D.forward), randomness='different')(torch.cat((X_depl_tt[:,:3], X_depl_tt[:,8:]), axis = 1))
            # J = torch.cat([torch.autograd.functional.jacobian(model_D, X_depl_tt[i:i+1], create_graph = True) for i in range(len(X_depl_tt))], dim=0)[:,:,0,:]
            if len(J.shape) < 4:
                J = J.unsqueeze(2)
            D_t_ = J.cpu().detach().numpy()
            D_t = small_value*np.ones((D_t_.shape[0],8,8))
            D_t[:,:3,:3] = D_t_[:,:3,0,:3]
            predictions_depl = D_t

    else: 
        UserWarning('Please specify tag = "sig" or tag = "D". This function should not be used if model_dim = "ALL_DIM"')


    return predictions_depl






'''-------------------------------------------------------------NOT IN USE-----------------------------------------------------------'''

def inp_out_plt(eps: str, sig:str, data_model: dict, path: str, path_plots: str):
    '''
    Plots analytical function vs. NN function
    Uses deployment strategy, i.e. one prediction at a time, ignores batches.
    eps             (str)       x-axis variable to be plotted (can be one of the strings listed below)
    sig             (str)       y-axis variable to be plotted (can be one of the strings listed below)
    data_model      (dict)      optional data (eps, t and sig values of e.g. training or test data set) 
    path            (str)       path where model is saved
    save_path       (str)       path where figures are saved
    '''
    # Parameters to be plotted
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Helvetica",
        "font.size": 12,
        })
    
    eps_id = np.array(['eps_x', 'eps_y', 'eps_xy', 'chi_x', 'chi_y', 'chi_xy', 'gam_x', 'gam_y'])
    sig_id = np.array(['n_x', 'n_y', 'n_xy', 'm_x', 'm_y', 'm_xy', 'v_x', 'v_y'])
    eps_name = np.array([r'$\varepsilon_x$', r'$\varepsilon_y$', r'$\varepsilon_{xy}$', 
                         r'$\chi_x$', r'$\chi_y$', r'$\chi_{xy}$',
                         r'$\gamma_{xz}$', r'$\gamma_{yz}$'])
    sig_name = np.array([r'$n_x$', r'$n_y$', r'$n_{xy}$', 
                         r'$m_x$', r'$m_y$', r'$m_{xy}$',
                         r'$v_{xz}$', r'$v_{yz}$'])

    eps_units = np.array(['[-]', '[-]', '[-]', '[1/cm]', '[1/cm]', '[1/cm]', '[-]', '[-]'])
    sig_units = np.array(['[MN/cm]', '[MN/cm]', '[MN/cm]', '[MN]', '[MN]', '[MN]', '[MN/cm]', '[MN/cm]'])
    
    # 0 - sort eps, t and sig data from simulation according to desired eps in ascending order
    if data_model is not None:
        data = np.concatenate((data_model['mat_data_np_TrainEvalTest']['X_train'][:,0:9], data_model['mat_data_np_TrainEvalTest']['y_train'][:,0:8]), axis = 1)
        data_sort = data[data[:, np.where(eps_id == eps)[0][0]].argsort()]
        num_rows = data_sort.shape[0]
    else:
        # here would be just random numbers at which the prediction and analytical solution should be evaluated
        # (if not the training  data points)
        # data_sort = ...
        # num_rows = ...
        pass
    
    path_data_name = os.path.join(path, 'new_data')
    # 1 - calculate corresponding predictions and analytical solutions
    sig_pred = np.zeros((num_rows, 8))
    sig_analytic = np.zeros((num_rows,8))
    for k in range(num_rows):
        mat_pred = predict_sig_D(data_sort[k,0:9].reshape((1,9)), path_data_name, 'train', 'direct')
        sig_pred[k,:] = mat_pred['sig_h']
        data_sort_analytic = data_sort.copy()
        data_sort_analytic[3:6] = data_sort_analytic[3:6]*10**(-1)      #adjust units of chi to 1/mm
        mat_a = D_an(data_sort_analytic[k,0:8], data_sort[k,8])
        sig_analytic[k,:] = mat_a['sig_a']
    # change units of analytical sigma to units of training / evaluation sigma
    sig_analytic[:,0:3], sig_analytic[:,6:8] = sig_analytic[:,0:3]*10**(-5), sig_analytic[:,6:8]*10**(-5)
    sig_analytic[:,3:6] = sig_analytic[:,3:6]*10**(-6)


    # 2 - select desired data for plot
    x_plt = data_sort[:,np.where(eps_id == eps)[0][0]]
    y_plt = sig_pred[:,np.where(sig_id == sig)[0][0]]
    y_plt_analytic = sig_analytic[:,np.where(sig_id == sig)[0][0]]
    y_plt_sim = data_sort[:,9+np.where(sig_id == sig)[0][0]]
    t_plt = data_sort[:,8]


    # 3 - find unique t-values
    unique_t = np.array([50, 60, 80, 140])
    # unique_t = np.unique(t_plt)           # uncomment this row to see actual unique values
    amt_t = len(unique_t)
    cmap = plt.get_cmap('viridis')
    colors = cmap(np.linspace(0, 1, amt_t))

    # 4 - plot
    fig, ax = plt.subplots(1, 1, figsize = (5,5))
    # ax.plot(x_plt, y_plt, label = 'prediction', color = 'lightgrey')
    for j in range(amt_t):
        ax.plot(x_plt[t_plt == unique_t[j]], y_plt[t_plt == unique_t[j]], 
                label = 'prediction, t = ' + np.array2string(unique_t[j]) + 'mm', 
                color = colors[j])
        if j == amt_t-1:
            ax.scatter(x_plt[t_plt == unique_t[j]], y_plt_analytic[t_plt == unique_t[j]], label = 'analytic', color = colors[j], marker = 'x')
            ax.scatter(x_plt[t_plt == unique_t[j]], y_plt_sim[t_plt == unique_t[j]], label = 'simulation', color = 'grey', marker = 'o', facecolors='none')
        else: 
            ax.scatter(x_plt[t_plt == unique_t[j]], y_plt_analytic[t_plt == unique_t[j]], label = 'analytic', color = colors[j], marker = 'x')
            ax.scatter(x_plt[t_plt == unique_t[j]], y_plt_sim[t_plt == unique_t[j]], color = 'grey', marker = 'o', facecolors='none')
    ax.set_xlabel(eps_name[np.where(eps_id == eps)[0][0]]+' '+eps_units[np.where(eps_id == eps)[0][0]])
    ax.set_ylabel(sig_name[np.where(eps_id == eps)[0][0]]+' '+sig_units[np.where(sig_id == sig)[0][0]])
    fig.legend(loc='center right', bbox_to_anchor=(1.4, 0.5))
    # plt.show()

    plt.tight_layout()
    if path_plots is not None:
        filename = os.path.join(path_plots, 'inp-outp_'+eps+'_'+sig+'.png')
        plt.savefig(filename, dpi=100, bbox_inches='tight')
    plt.show()
    plt.close()
    # wandb.log({"inp-outp_"+eps+'_'+sig: wandb.Image(filename)})
    return


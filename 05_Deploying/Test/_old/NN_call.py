import os
import pickle
import torch
import pandas as pd
import numpy as np
from Test.FFNN_class_depl import *
from Test.data_work_depl import *

'''
Do not use this script anymore. Instead call NN from Test.NN_call.py 
(where the predict_sig_D function will have the same effect as this function used to)

'''






# def NN_call(eps_h:np.array, path_data:str):
#     '''
#     Calls the NN and outputs desired values of generalised stresses and D-matrix

#     Input: 
#     eps_h       (np.array)          Generalised strains
#     path_data   (str)               Path where trained model data is saved

#     Output: 
#     sig_h       (np.array)          Generalised stresses
#     D           (np.array)          Material matrix for element (D)
#     '''

#     ## Load the relevant files

#     with open(os.path.join(path_data, 'inp.pkl'),'rb') as handle:
#         inp = pickle.load(handle)
#     with open(os.path.join(path_data, 'mat_data_stats.pkl'),'rb') as handle:
#         mat_data_stats = pickle.load(handle)


#     ## Load the trained model

#     model = FFNN(inp)
#     model.load_state_dict(torch.load(os.path.join(path_data, 'trained_model.pt')))
#     model.eval()


#     ## Make a prediction

#     # random initialisation as test
#     # random_eps_h = 1*10**(-5)*np.ones((1,8))
#     # eps_h_pd = pd.DataFrame(eps_h)
#     # for completeness of DataLoader add random sig (will not be used)
#     random_sig_h = np.zeros((1,8))      
#     # print(random_sig_h.shape)

#     # Transform into normalised coordinates
#     stats_X_train = mat_data_stats['stats_X_train']
#     stats_y_train = mat_data_stats['stats_y_train']
#     X_depl_t = transform_data(eps_h, stats_X_train, forward = True)


#     # Create dataset in correct format
#     X_depl_tt = torch.from_numpy(X_depl_t)
#     X_depl_tt = X_depl_tt.type(torch.float32)
#     Y_depl_tt = torch.from_numpy(random_sig_h)
#     Y_depl_tt = Y_depl_tt.type(torch.float32)

#     data_depl_t = MyDataset(X_depl_tt, Y_depl_tt)
#     deploy_loader = DataLoader(data_depl_t, batch_size=1, shuffle = True)


#     # Make predictions of sig_generalised
#     sig_h_t = np.zeros((1,8))           
#     for i, (features, labels) in enumerate(deploy_loader):
#         predictions_t = model(features)
#         sig_h_t[i] = predictions_t.detach().numpy()
#     # print('sig_h_t', sig_h_t)


#     # Make predictions of D-matrix (i.e. the individual derivatives)
#     for i, (features, labels) in enumerate(deploy_loader):
#         features.requires_grad = True
#         predictions_t = model(features)
#         J = torch.zeros((1,eps_h.shape[1],eps_h.shape[1]))
#         for i in range(8):
#             grd = torch.zeros((1,8))
#             grd[0, i] = 1
#             predictions_t.backward(gradient = grd, retain_graph = True)
#             J[:,:,i] = features.grad
#             features.grad.zero_()
#         D_t = J.detach().numpy()
#     D_t = np.squeeze(D_t, axis = 0)
#     D_t = D_t[0:8, 0:8]
#     # print('D_t', D_t)
#     # print(D_t.shape)


#     # Transform back sig, convert to correct units
#     sig_h = transform_data(sig_h_t, stats_y_train, forward = False)
#     sig_h_f = sig_h
#     sig_h_f[3:6] = sig_h[3:6]*10**3
#     # print('sig_h', sig_h)

#     # Transform back D
#     D_coeff = np.zeros((8,8))
#     for i in range(8):
#         for j in range(8):
#             D_coeff[i,j] = np.divide(stats_y_train['std'][i], stats_X_train['std'][j])
#     D = np.multiply(D_t,D_coeff)
#     # print('D',D)

#     NN_res = {
#          'sig_h': sig_h,
#          'D': D
#     }

#     return NN_res
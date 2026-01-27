'''
bav, 15.04.2025
create convergence images directly from deployment

'''

import os
import pickle
import pandas as pd
from main_utils_vb import all_eps_sig_plots, imshow_plots, all_De_plots

NAME = 'data_20250415_1636_casexx'
NN_hybrid = {
        'predict_sig': True,
        'predict_D': False
}


# Read files:
path_res = os.path.join(os.getcwd(), '05_Deploying\\data_out\\' + NAME)
with open(os.path.join(path_res, 'mat_res_norm.pkl'),'rb') as handle:
		mat_res_pkl = pickle.load(handle)
mat_res_pd = pd.DataFrame(mat_res_pkl) 
if NN_hybrid['predict_sig'] == True:
    with open(os.path.join(path_res, 'mat_res_NN_sig.pkl'),'rb') as handle:
            mat_res_pkl_NN = pickle.load(handle)
    mat_res_pd_NN = pd.DataFrame(mat_res_pkl_NN)
elif NN_hybrid['predict_D'] == True: 
    with open(os.path.join(path_res, 'mat_res_NN_D.pkl'),'rb') as handle:
            mat_res_pkl_NN = pickle.load(handle)
    mat_res_pd_NN = pd.DataFrame(mat_res_pkl_NN)


# Plot figures
eh_cum_NN = mat_res_pd_NN[['eh_cum']].to_numpy()[0,0]
eh_cum_NN_ = mat_res_pd_NN[['eh_cum_']].to_numpy()[0,0]
eh_cum_NLFEA = mat_res_pd[['eh_cum']].to_numpy()[0,0]

if NN_hybrid['predict_sig'] and not NN_hybrid['predict_D']:
    sh_cum_NN = mat_res_pd_NN[['sh_cum']].to_numpy()[0,0]
    sh_cum_NN_ = mat_res_pd_NN[['sh_cum_']].to_numpy()[0,0]    
    sh_cum_NLFEA = mat_res_pd[['sh_cum']].to_numpy()[0,0]
    coord = mat_res_pd_NN[['COORD']]

    # adjust plotting function that if NLFEA vs NN --> plots all three things in one.
    all_eps_sig_plots(7, 7, eh_cum_NN, 0, sh_cum_NN, eh_cum_NN_, sh_cum_NN_,eh_cum_NLFEA,sh_cum_NLFEA, 
                      NN = 'NN', tag = 'max', final = NAME)

if not NN_hybrid['predict_sig'] and NN_hybrid['predict_D']:
    De_cum_NN = mat_res_pd_NN[['De_cum']].to_numpy()[0,0]
    De_cum_NN_ = mat_res_pd_NN[['De_cum_']].to_numpy()[0,0]
    De_cum_NLFEA = mat_res_pd[['De_cum']].to_numpy()[0,0]

    all_De_plots(6, 6, eh_cum_NN, eh_cum_NN_, eh_cum_NLFEA, De_cum_NN, De_cum_NN_, De_cum_NLFEA, tag = 'max')
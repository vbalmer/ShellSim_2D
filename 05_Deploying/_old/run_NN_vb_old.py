import os, sys
import builtins
import streamlit as st
from deploy_utils import HiddenPrints, run_deployment, extend_material_parameters, get_paths


# write a wrapper around normal printing function, such that it's written to streamlit
original_print = builtins.print
def streamlit_print(*args, **kwargs):
    st.write(*args, **kwargs)
builtins.print = streamlit_print



from Main_vb import main_solver
import numpy as np
import time
import pickle
import pandas as pd
import os
import subprocess

###################################################
# 0 - App management
###################################################
st.title('Deployment of ML-FEA hybrid Simulator')
model_no = st.text_input("Model number", "v_137")
epoch_no = st.text_input("Epoch number", "9979")
options = ['1 - Pure shear', '2 - 1D Bending', '3 - 1D Bending + normal force']
scenario = st.radio("Select the scenario to be simulated:", options)
save_folder = st.checkbox("Save the data to a new folder")


###################################################
# 1 - Geometry, Material Inputs
###################################################
# import numpy file from sampler
numpy_sampler = False
single_sample = True

# define location of trained model and input data to be used
path_collection = get_paths(model_no, epoch_no)

# Inputs for solver
if numpy_sampler:
    # If the data comes from the sampler:
    path = os.path.join(os.getcwd(), '01_SamplingFeatures')
    # name = 'data_240724_1752_case4\outfile.npy'
    name = 'data_20241104_1545_case8\outfile.npy'
    features = np.load(os.path.join(path, name))
    mat_tot_dict = {
        'L': features[:,0],         # length
        'B': features[:,1],         # width
        'E_1': features[:,2],       # Young's modulus steel / glass / concrete
        'E_2': features[:,3],       # Young's modulus - / interlayer / reinforcing steel
        'ms': features[:,4],        # mesh_size
        'F': features[:,5],         # force_magnitude
        's': features[:,6],         # scenario 0...6
        't_1': features[:,7],       # thickness of the plate
        't_2': features[:,8],       # thickness of plate
        'nl': features[:,9],        # amount of layers
        'nu_1': features[:,10],     # Poisson's ratio
        'nu_2': features[:,11],     # Poisson's ratio
        'mat': features[:,12]       # Material type     (1 = lin.el., 3 = CMM, 10 = glass)
    }

elif numpy_sampler == False and not single_sample: 
    path = os.path.join(os.getcwd(), '01_SamplingFeatures')
    name = 'output\\data_20241111_0904_case10\\outfile.pkl'
    with open(os.path.join(path, name),'rb') as handle:
            in_dict = pickle.load(handle)
    mat_tot_dict = in_dict

if single_sample:
    if scenario == '1 - Pure shear':
        # pure shear force (scenario 10)
        mat_tot_dict = {
            'L': np.array([7500]),
            'B': np.array([7500]),
            'E_1': np.array([33600]),
            'E_2': np.array([0]),
            'ms': np.array([750]),
            'F': np.array([75000]),       #=10*L
            'F_N': np.array([0]),
            's': np.array([8]),
            't_1': np.array([200]),
            't_2': np.array([0]),
            'nl': np.array([20]),
            'nu_1': np.array([0.2]),
            'nu_2': np.array([0]),
            'mat': np.array([1])
        }

    # # moment + normal force (scenario 11)
    elif scenario == '3 - 1D Bending + normal force':
        mat_tot_dict = {
            'L': np.array([7500]),
            'B': np.array([7500]),
            'E_1': np.array([33600]),
            'E_2': np.array([0]),
            'ms': np.array([750]),
            'F': np.array([0.018]),
            'F_N': np.array([750000]),
            's': np.array([11]),
            't_1': np.array([200]),
            't_2': np.array([0]),
            'nl': np.array([20]),
            'nu_1': np.array([0.2]),
            'nu_2': np.array([0]),
            'mat': np.array([1])
        }

    elif scenario == '2 - 1D Bending':
        # moment (scenario 10)
        mat_tot_dict = {
            'L': np.array([7500]),
            'B': np.array([7500]),
            'E_1': np.array([33600]),
            'E_2': np.array([0]),
            'ms': np.array([750]),
            'F': np.array([0.01]),
            'F_N': np.array([0]),
            's': np.array([10]),
            't_1': np.array([200]),
            't_2': np.array([0]),
            'nl': np.array([20]),
            'nu_1': np.array([0.2]),
            'nu_2': np.array([0]),
            'mat': np.array([1])
        }

mat_tot_raw = pd.DataFrame.from_dict(mat_tot_dict)


if not single_sample:
    # Choose the simulation number(s) that shall be tested
    # import mat_res.pkl from data that was used for training of algorithm
    path_mat_res = os.path.join(os.getcwd(), '02_Simulator')
    # name = 'Simulator\\results\saved_runs\data_240805_1134_case4\mat_res.pkl'        # take 240805 here, as there the SN are included
    # name = 'Simulator\\results\saved_runs\data_20241104_1854_case8\mat_res.pkl'
    name = 'Simulator\\results\saved_runs\data_20241111_1101_case10\mat_res.pkl'
    with open(os.path.join(path_mat_res, name),'rb') as handle:
            mat_res = pickle.load(handle)

    desired_SN = [21, 32, 45, 74, 76, 82]
    mat_tot = mat_tot_raw.iloc[[np.where(mat_res['SN'] == i)[0][0] for i in desired_SN]]
    mat_tot.reset_index(drop=True, inplace=True)
    print('As a check, in mat_tot (from sampling) (t1_tot = ', round(mat_tot['t_1'][0], 1), 'mm) should be equal to t1 in mat_res (directly read) (t1_res = ', round(mat_res['t_1'][desired_SN[0]],1), 'mm).')
    # print(mat_tot)
else: 
    mat_tot = mat_tot_raw
            
print(mat_tot)


##########################################
# 2 - Simulation Inputs
##########################################

conv_plt = True
simple = True
samples = int(mat_tot.shape[0])
n_simple = 1
NN_hybrid = {
        'predict_D': False,                 # If true, solves the system with NN_hybrid solver (prediction of D); if False: "normal" solver
        'predict_sig': True               # If true, solves the system with NN_hybrid solver (prediction of sig); if False: "normal" solver
    }
    # Note: predict_D should not be used in lin.el. case, as there is only one initialisation and this happens with lin.el. model. 
    # => for lin.el. always set predict_D = False
NN_hybrid_2 = {'predict_D': False,
                'predict_sig': False}


##########################################
# 3 - Run Simulation
##########################################

if __name__ == '__main__':
    st.button("Reset", type='primary')
    if st.button("Run simulation"):
        st.write("Standard Simulation is running...")
        mat_res = run_deployment(mat_tot, conv_plt, simple, n_simple, NN_hybrid, path_collection)
        mat_res_pd_NN = pd.DataFrame.from_dict(mat_res)
        mat_res_pd_NN[['ms', 'L', 'B']].head()
        st.write("Simulation with NN is running...")
        mat_res = run_deployment(mat_tot, conv_plt, simple, n_simple, NN_hybrid_2, path_collection)
        mat_res_pd = pd.DataFrame.from_dict(mat_res)
        mat_res_pd[['ms', 'L', 'B']].head()
    else: 
        st.write("Press the button to initialise the simulation")


##########################################
# 4 - Saving files
##########################################

# if data should be saved to folder instead of being overwritten with the next simulation, use save_folder = True
from datetime import datetime
import shutil

if save_folder:
    relative_path = ['data_out\\mat_res_norm.pkl', 'data_out\\mat_res_NN_sig.pkl']
    
    for i in range(len(relative_path)):
        source_folder = os.path.dirname(relative_path[i])
        file_name = os.path.basename(relative_path[i])

        current_time = datetime.now()
        new_folder = current_time.strftime("data_%Y%m%d_%H%M_case"+'xx')
        new_folder_path = os.path.join(source_folder, new_folder)

        os.makedirs(new_folder_path, exist_ok=True)

        destination_path = os.path.join(new_folder_path, file_name)
        shutil.copy(relative_path[i], destination_path)

        print('File', i, 'is copied to', destination_path)
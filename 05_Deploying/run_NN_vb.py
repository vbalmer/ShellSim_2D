'''
This file is the same file as run_NN.py but contains the integration into streamlit 
(that is not entirely working yet)

For pure deployment trials please use the other file: run_NN.py
This file should only be used for the app development.

(vb, 15.04.2025)
'''



import os
import sys
import builtins
import streamlit as st
import numpy as np
import pandas as pd
import pickle
import time
import shutil
import subprocess
import multiprocessing
from datetime import datetime
from deploy_utils import HiddenPrints, run_deployment, extend_material_parameters, get_paths
from Main_vb import main_solver

# Override print function to display output in Streamlit
original_print = builtins.print
def streamlit_print(*args, **kwargs):
    st.write(*args, **kwargs)
builtins.print = streamlit_print

st.title('Deployment of ML-FEA Hybrid Simulator')

##########################
# Input Fields
##########################
model_no = st.text_input("Model number", "v_137")
epoch_no = st.text_input("Epoch number", "9979")
options = ['1 - Pure shear', '2 - 1D Bending', '3 - 1D Bending + normal force']
scenario = st.radio("Select the scenario to be simulated:", options)
save_folder = st.checkbox("Save the data to a new folder")

# Define Paths
path_collection = get_paths(model_no, epoch_no)

##########################
# Define Material Parameters
##########################
single_sample = True
if single_sample:
    if scenario == '1 - Pure shear':
        mat_tot_dict = {
            'L': np.array([7500]), 'B': np.array([7500]),
            'E_1': np.array([33600]), 'E_2': np.array([0]),
            'ms': np.array([750]), 'F': np.array([75000]), 'F_N': np.array([0]),
            's': np.array([8]), 't_1': np.array([200]), 't_2': np.array([0]),
            'nl': np.array([20]), 'nu_1': np.array([0.2]), 'nu_2': np.array([0]),
            'mat': np.array([1])
        }
    elif scenario == '3 - 1D Bending + normal force':
        mat_tot_dict = {
            'L': np.array([7500]), 'B': np.array([7500]),
            'E_1': np.array([33600]), 'E_2': np.array([0]),
            'ms': np.array([750]), 'F': np.array([0.018]), 'F_N': np.array([750000]),
            's': np.array([11]), 't_1': np.array([200]), 't_2': np.array([0]),
            'nl': np.array([20]), 'nu_1': np.array([0.2]), 'nu_2': np.array([0]),
            'mat': np.array([1])
        }
    elif scenario == '2 - 1D Bending':
        mat_tot_dict = {
            'L': np.array([7500]), 'B': np.array([7500]),
            'E_1': np.array([33600]), 'E_2': np.array([0]),
            'ms': np.array([750]), 'F': np.array([0.01]), 'F_N': np.array([0]),
            's': np.array([10]), 't_1': np.array([200]), 't_2': np.array([0]),
            'nl': np.array([20]), 'nu_1': np.array([0.2]), 'nu_2': np.array([0]),
            'mat': np.array([1])
        }
mat_tot = pd.DataFrame.from_dict(mat_tot_dict)


##########################
# Simulation Parameters
##########################
conv_plt = True
simple = True
n_simple = 1
NN_hybrid = {'predict_D': False, 'predict_sig': True}
NN_hybrid_2 = {'predict_D': False, 'predict_sig': False}

def run_simulation(queue, mat_tot, conv_plt, simple, n_simple, NN_hybrid, path_collection):
    """Runs deployment in a separate process and stores result in queue."""
    result = run_deployment(mat_tot, conv_plt, simple, n_simple, NN_hybrid, path_collection)
    queue.put(result)

if __name__ == '__main__':
    st.button("Reset", type='primary')
    if st.button("Run simulation"):
        st.write("Standard Simulation is running...")
        
        # Run simulation in a separate process
        queue = multiprocessing.Queue()
        p = multiprocessing.Process(target=run_simulation, args=(queue, mat_tot, conv_plt, simple, n_simple, NN_hybrid, path_collection))
        p.start()
        p.join()
        
        # Retrieve results
        if not queue.empty():
            mat_res = queue.get()
            mat_res_pd_NN = pd.DataFrame.from_dict(mat_res)
            st.write(mat_res_pd_NN[['ms', 'L', 'B']].head())
        
        st.write("Simulation with NN is running...")
        
        # Run NN simulation in another process
        queue_NN = multiprocessing.Queue()
        p_NN = multiprocessing.Process(target=run_simulation, args=(queue_NN, mat_tot, conv_plt, simple, n_simple, NN_hybrid_2, path_collection))
        p_NN.start()
        p_NN.join()
        
        if not queue_NN.empty():
            mat_res_NN = queue_NN.get()
            mat_res_pd = pd.DataFrame.from_dict(mat_res_NN)
            st.write(mat_res_pd[['ms', 'L', 'B']].head())
    else:
        st.write("Press the button to initialize the simulation")



##########################
# Saving Files
##########################
if save_folder:
    relative_paths = ['data_out/mat_res_norm.pkl', 'data_out/mat_res_NN_sig.pkl']
    for path in relative_paths:
        source_folder = os.path.dirname(path)
        file_name = os.path.basename(path)
        current_time = datetime.now()
        new_folder = current_time.strftime("data_%Y%m%d_%H%M_casexx")
        new_folder_path = os.path.join(source_folder, new_folder)
        os.makedirs(new_folder_path, exist_ok=True)
        destination_path = os.path.join(new_folder_path, file_name)
        shutil.copy(path, destination_path)
        print(f'File {file_name} is copied to {destination_path}')

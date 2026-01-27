import numpy as np
import pyDOE as doe
import pandas as pd
import math
from DataSampling import *
import os
import pickle
from datetime import datetime
import shutil
import time

samples = 90
save_folder = True                  # Save the sampled data in a folder
criterion = 'center'                # can be 'c', 'm', 'cm' or 'corr' (see here: https://pythonhosted.org/pyDOE/randomized.html)
ones = np.ones(samples)
mat = 3


################ Linear elastic ###############
if mat == 1:
    # Sampled parameters
    par_names = ['t_1', 'L/t']
    # min = [15, 150]
    # max = [30, 250]
    min = [200, 30]              # for concrete
    max = [400, 40]              # for concrete
    lhs_sampler = samplers(par_names, min, max, samples, criterion) 
    data = lhs_sampler.lhs()
    np_data = data.to_numpy(dtype = np.float64)
    np_data[:,0] = np.round(np_data[:,0]/5)*5       # [mm]  Round t_1 to nearest 5
    np_data[:,1] = np.round(np_data[:,1]/2)*2     # [-]   Round L/t to nearest 2 (for steel: change to 10)
    sampled_params = dict(zip(par_names, np_data.T))

    # Constant parameters
    constant_params = {
        'E_1': 33600,                          # [N/mm^2]  Young's modulus of plate 
        'F': 0.015,                              # [N/mm^2]  Force applied to plate (10 kN/m^2 = 0.01 N/mm^2)
        's': 10,                                # [-]       Scenario number (type of boundary conditions)        
        'nl': 20,                               # [-]       Number of layers
        'nu_1':0.2,                             # [-]       Poisson's ratio of plate
        'mat':1,                                # [-]       Material law type (1: Linear elastic, 2: Glass, 3: Reinforced concrete)  
        'ms_fac': 0.1,                          # [-]       Mesh size factor
        
        'E_2': 0,                               # these values are not required for lin.el. but need to be defined for the code to work
        't_2': 0,
        'nu_2': 0,
    }
    constant_params = {key: value * ones[i] for i, (key, value) in enumerate(constant_params.items())}

    # Calculated parameters
    L_calc = np.multiply(np_data[:,1], np_data[:,0])   # Calculate L from L/t and t_1
    ms = constant_params['ms_fac']*L_calc
    calculated_params = {
            'L': L_calc,                            # Length of plate [mm]
            'B': L_calc,                            # Width of plate [mm]
            'ms': ms,                               # Mesh size [mm]
        }
    
    filename = 'outfile.pkl'


################ Glass ###############
elif mat == 2:
    # Sampled parameters
    par_names = ['E_1', 'E_2', 't_1', 't_2', 'L/t']
    min = [70000, 0.003, 2, 0.38, 150]              # TODO: Add realistic L/t
    max = [71000, 0.006, 25, 2.28, 200]             # TODO: Add realistic L/t
    lhs_sampler = samplers(par_names, min, max, samples, criterion) 
    data = lhs_sampler.lhs()
    np_data = data.to_numpy(dtype = np.float64)
    np_data[:,0] = np.round(np_data[:,0]/200)*200   # [N/mm^2]  Round E_1 to nearest 200
    np_data[:,1] = np.round(np_data[:,1], 3)        # [N/mm^2]  Round E_2 to 3 decimals
    np_data[:,2] = np.round(np_data[:,2], 0)        # [mm]      Round t_1 to nearest 1
    np_data[:,3] = np.round(np_data[:,3], 1)        # [mm]      Round t_2 to nearest 0.1
    np_data[:,4] = np.round(np_data[:,4]/10)*10     # [-]       Round L/t to nearest 10
    sampled_params = dict(zip(par_names, np_data.T))

    # Constant parameters
    constant_params = {
        'F': 0.01,                              # [N/mm^2]  Force applied to plate (10 kN/m^2 = 0.01 N/mm^2)
        's': 10,                                # [-]       Scenario number (type of boundary conditions)
        'nl': 7,                                # [-]       Number of layers
        'nu_1':0.24,                            # [-]       Poisson's ratio of plate
        'nu_2':0.499,                           # [-]       Poisson's ratio of adhesive
        'mat':2,                                # [-]       Material law type (1: Linear elastic, 2: Glass, 3: Reinforced concrete)
        'ms_fac': 0.1,                          # [-]       Mesh size factor
    }
    constant_params = {key: value * ones[i] for i, (key, value) in enumerate(constant_params.items())}

    # Calculated parameters
    L_calc = ones/np_data[:,1] * np_data[:,0]   # Calculate L from L/t and t_1
    ms = constant_params['ms_fac']*L_calc
    calculated_params = {
            'L': L_calc,                            # [mm] Length of plate
            'B': L_calc,                            # [mm] Width of plate
            'ms': ms,                               # [mm] Mesh size
        }
    
    filename = 'outfile_glass.pkl'


################ Reinfored concrete ###############
elif mat == 3:
    # Sampled parameters
    par_names = ['L/t', 'CC', 'F', 'rho', 't_1']
    min = [30, 0, 0.005, 0.005, 200]
    max = [40, 5, 0.015, 0.030, 400]
    lhs_sampler = samplers(par_names, min, max, samples, criterion)
    data = lhs_sampler.lhs()
    np_data = data.to_numpy(dtype = np.float64)
    np_data[:,0] = np.round(np_data[:,0]/2)*2
    np_data[:,1] = np.round(np_data[:,1], 0)
    np_data[:,2] = np.round(np_data[:,2] / 0.005)*0.005
    np_data[:,3] = np.round(np_data[:,3]/ 0.005)*0.005
    np_data[:,4] = np.round(np_data[:,4]/50)*50
    sampled_params = dict(zip(par_names, np_data.T))

    # Constant parameters
    constant_params = {
        'nl': 20, 
        'mat': 3, 
        's': 10,
        'E_1': 0,
        'E_2': 0,
        'F_N': 0, 
        't_2': 0,
        'nu_1': 0,
        'nu_2': 0
    }
    constant_params = {key: value * ones[i] for i, (key, value) in enumerate(constant_params.items())}

    # Calculated parameters    
    L_calc = np.multiply(np_data[:,0], np_data[:,4])
    calculated_params = {
        'L': L_calc,
        'B': L_calc,
        'ms': L_calc/10,
    }

    filename = 'outfile_RC.pkl'


################ Create plots ###############
save_path = '01_SamplingFeatures\\plots'
plot_dict = {}
plot_dict.update(sampled_params)
plot_dict.update(calculated_params)
hist_from_dict(plot_dict, constant_params, save_path)


############### Outputfile = Input for Simulation ################

outfile = {}
outfile.update(sampled_params)
outfile.update(constant_params)
outfile.update(calculated_params)
print(outfile.keys())

# Save the output file

folder_name = '01_SamplingFeatures\\output'
with open(os.path.join(folder_name, filename), 'wb') as fp:
    pickle.dump(outfile, fp)

time.sleep(5)


if save_folder:
    source_folder = folder_name
    file_name = 'outfile.pkl'

    current_time = datetime.now()
    new_folder = current_time.strftime("data_%Y%m%d_%H%M_case"+str(int(outfile['s'])))
    new_folder_path = os.path.join(source_folder, new_folder)

    os.makedirs(new_folder_path, exist_ok=True)

    destination_path = os.path.join(new_folder_path, file_name)
    with open(destination_path, 'wb') as fp:
        pickle.dump(outfile, fp)

    print('File is copied to', destination_path)



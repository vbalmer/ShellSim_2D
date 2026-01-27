import numpy as np
import os
import sys
from Main_vb import main_solver
import time


# to hide all prints in simulation, when simulating multiple examples after eachother

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout



dict_CC = {
    'CC': [0, 1, 2, 3, 4, 5],                                           # [-] just an identifier
    'Ec': [32e3, 33.6e3, 35e3, 36.3e3, 37.5e3, 39e3],                   # [MPa]
    'tb0': [5.2, 5.8, 6.4, 7.0, 7.6, 8.2],                              # [MPa]
    'tb1': [2.6, 2.9, 3.2, 3.5, 3.8, 4.1],                              # [MPa]
    'ect': [0.08e-3, 0.09e-3, 0.09e-3, 0.1e-3, 0.1e-3, 0.11e-3],        # [-]
    'ec0': [2.2e-3, 2.3e-3, 2.3e-3, 2.4e-3, 2.4e-3, 2.5e-3],            # [-]
    'fcp': [25, 30, 35, 40, 45, 50],	                                # [MPa]    
    'fct': [2.6, 2.9, 3.2, 3.5, 3.8, 4.1],                              # [MPa]
}



def extend_material_parameters(mat_tot_dict_, mat_dict=dict_CC):
    # Add constant steel parameters
    fsy = 435               # [MPa]
    fsu = 470               # [MPa]
    Es = 205e3              # [MPa]
    Esh = 8e3               # [MPa]
    D = 16                  # [mm]

    samples = mat_tot_dict_['L'].shape[0]
    
    mat_tot_dict_.update({'fsy': fsy*np.ones((samples,)), 'fsu': fsu*np.ones((samples,)), 'Es': Es*np.ones((samples,)), 'Esh': Esh*np.ones((samples,)), 'D': D*np.ones((samples,))})
    mat_tot_dict = mat_tot_dict_

    # Add additional concrete parameters
    mat_tot_dict['E_1'] = np.zeros((samples,))
    mat_tot_dict['tb0'] = np.zeros((samples,))
    mat_tot_dict['tb1'] = np.zeros((samples,))
    mat_tot_dict['ect'] = np.zeros((samples,))
    mat_tot_dict['ec0'] = np.zeros((samples,))
    mat_tot_dict['fcp'] = np.zeros((samples,))
    mat_tot_dict['fct'] = np.zeros((samples,))

    for i in range(samples):
        index = int(np.where(mat_dict['CC'] == mat_tot_dict['CC'][i])[0])
        mat_tot_dict['E_1'][i] = mat_dict['Ec'][index]
        mat_tot_dict['tb0'][i] = mat_dict['tb0'][index]
        mat_tot_dict['tb1'][i] = mat_dict['tb1'][index]
        mat_tot_dict['ect'][i] = mat_dict['ect'][index]
        mat_tot_dict['ec0'][i] = mat_dict['ec0'][index]
        mat_tot_dict['fcp'][i] = mat_dict['fcp'][index]
        mat_tot_dict['fct'][i] = mat_dict['fct'][index]

    return mat_tot_dict




def run_simulation(mat_tot, conv_plt, simple, n_simple):
    samples = mat_tot['L'].shape[0]
    t0 = time.time()

    # iterate only over first input sets
    if simple:
        mat_res = [dict() for x in range(n_simple)]
        for i in range(int(n_simple)):
            mat = mat_tot.loc[i,:]
            mat_res[i] = main_solver(mat,conv_plt)
            if i>0 and i%10 == 0:
                print('**********************************************************************')
                print('Data points upto row', i, 'are simulated')
                print('time required for first', i,'points:',time.time()-t0, 'secs') 

    # iterate batchwise over all input sets (here: 3 batches which are created automatically based on size of input vector)
    else:
        mat_res = [dict() for x in range(samples)]
        for i in range(int(samples/3)):
            mat = mat_tot.loc[i,:]
            with HiddenPrints():
                mat_res[int(i)] = main_solver(mat,conv_plt)
            if i>0 and i%10 == 0:
                print('**********************************************************************')
                print('Data points upto row', i, 'are simulated')
                print('time required for first', i,'points:',time.time()-t0, 'secs')
            
        for i in np.linspace(
            int(samples/3), 
            2*int(samples/3), 
            (2*int(samples/3)-int(samples/3))+1
            ):
            mat = mat_tot.loc[i,:]
            with HiddenPrints():
                mat_res[int(i)] = main_solver(mat,conv_plt)
            if i%10 == 0:
                print('**********************************************************************')
                print('Data points upto row', i, 'are simulated')
                print('time required for first', i, 'points:',time.time()-t0, 'secs')

        for i in np.linspace(
            2*int(samples/3), 
            int(samples), 
            (2*int(samples/3)-int(samples/3))+1
            ):
            mat = mat_tot.loc[int(i-1),:]
            with HiddenPrints():
                mat_res[int(i-1)] = main_solver(mat,conv_plt)
            if (i-1)%10 == 0:
                print('**********************************************************************')
                print('Data points upto row', i-1, 'are simulated')
                print('time required for first', i-1, 'points:',time.time()-t0, 'secs')
    
    return mat_res
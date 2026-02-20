### executing plotting functions ####
from plots_paper_utils import *
import os


save_path = os.path.join(os.getcwd(), '05_Deploying\\plots\\test')
SCATTER = False
PLOT_TRAINING = False
PLOT_DEPLOYING = False
PLOT_DATA_SENSITIVITY = True



if SCATTER: 
    plot_scatter_paper('data_20260108_2137_fake', 'data_20260105_1639_fake', save_path)


if PLOT_TRAINING: 
    plot_training_results(["480", "_167"], save_path, include_predict = False)


if PLOT_DEPLOYING:
    path_depl = {
               '2D-1': {
                        '$\\uprho_y$ = 0.75\%': 'data_20260120_1817_casexx',
                        '$\\uprho_y$ = 1.00\%': 'data_20260119_0803_casexx',
                        '$\\uprho_y$ = 1.50\%': 'data_20260120_1637_casexx',
                },
               '2D-2': {'$\\uprho_y$ = 0.75\%': 'data_20260121_1038_casexx',
                        '$\\uprho_y$ = 1.00\%': 'data_20260120_0836_casexx',
                        '$\\uprho_y$ = 1.50\%': 'data_20260120_1945_casexx',
                  },
                '2D-5': {
                        '$\\uprho_y$ = 0.75\%': 'data_20260121_1356_casexx',
                        '$\\uprho_y$ = 1.00\%': 'data_20260120_1210_casexx',
                        '$\\uprho_y$ = 1.50\%': 'data_20260121_1549_casexx',
                },
                '2D-8C': {
                        '$\\uprho_y$ = 0.75\%': 'data_20260203_1607_casexx',
                        '$\\uprho_y$ = 1.00\%': 'data_20260203_0925_casexx',
                        '$\\uprho_y$ = 1.50\%': 'data_20260203_1034_casexx',
                }
        }


    plot_deploying_results(path_depl, save_path, thresh = 2.5)


if PLOT_DATA_SENSITIVITY: 
    path_data = {
        '0': {'Uniform': 'data_20260108_2137_fake',
              'Log.': 'data_20260105_1639_fake',
            },
        '1': {'Log.': 'data_20260105_1639_fake'},
        '2': {'Uniform': 'data_20260108_2137_fake'}
    }

    path_deployment = {
        '0': {'$\\uprho_y$ = 1\%':'data_20260120_1037_casexx'}, 
        '1': {'$\\uprho_y$ = 1\%':'data_20260205_1543_casexx'},
        '2': {'$\\uprho_y$ = 1\%':'data_20260205_1433_casexx'},
    }

    plot_sensitivity(path_data, path_deployment, save_path, thresh = 2.5)
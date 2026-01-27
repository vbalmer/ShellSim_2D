import pickle
import numpy as np
import torch
import os
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
import wandb

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from lightning.pytorch import seed_everything
seed_everything(42)

from data_work import *
from FFNN_class_light import *
from call_light import *
from test_utils import *

os.environ["WANDB_MODE"] = "offline"

# load hyperparams, data
path = os.path.join(os.getcwd(), '04_Training')
path_plots = os.path.join(path, 'plots')
add_path = '_simple_logs\\v_239'
data_model = load_data(path, only_test = True, add_path = add_path)
SCALE = False
DOUBLE_NORM = True

# load model from checkpoint:
inp = data_model['inp'] 
model_test_dict = test_model_instance(inp, path, v_num='239', epoch='_19960')    
# in "epoch" also add identifier: _9273 or _exp1_341 etc.; for MoE: use a list ['MoE_578', 'exp1_1217', 'exp2_4899', 'exp3_4835']

# Create predictions and transform to numpy
if inp['Sobolev']:
    plot_data = make_prediction(inp, model_test_dict, data_model, transf_type = 'st-stitched', sc= SCALE, dn = DOUBLE_NORM)
else:
    plot_data = make_prediction(inp, model_test_dict, data_model, transf_type = 'std', sc = SCALE, dn = DOUBLE_NORM)
stats = data_model['mat_data_stats']

# Create plot for paper
plot_data_label_u = transf_units(plot_data['all_test_labels'], 'sig', forward = False)
plot_data_pred_u = transf_units(plot_data['all_predictions'], 'sig', forward = False)
multiple_diagonal_plots_paper(path_plots, plot_data_label_u, plot_data_pred_u, 'u', stats, 'rse')


# close wandb

wandb.finish()
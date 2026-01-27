import os
import pickle

from NN_call import predict_sig_D
from main_utils_vb import multiple_diagonal_plots_D
from data_work_depl import transf_units



path = os.path.join(os.getcwd(), '04_Training\\new_data\\_simple_logs\\v_283')
with open(os.path.join(path, 'mat_data_np_TrainEvalTest.pkl'),'rb') as handle:
        mat_data_np = pickle.load(handle)
input_j = mat_data_np['X_test']
model_path = os.path.join(path, 'best_trained_model__49879.pt')
save_path = '05_Deploying\\plots\\diagonal_plots_D\\test'


mat_pred = predict_sig_D(input_j, path_data_name = path, lit_model_path = model_path, stats = 'train', transf_type='st-stitched', predict='D', sc=False)
D = transf_units(mat_pred['D_pred'], id='D', forward = False, linel = False).reshape((-1,8,8))
D_true_u = mat_data_np['y_test'][:,8:].reshape((-1,8,8))
D_true = transf_units(D_true_u, id = 'D', forward = False, linel = False)


with open(os.path.join(path, 'mat_data_stats.pkl'),'rb') as handle:
        mat_data_stats = pickle.load(handle)

multiple_diagonal_plots_D(save_path, D, D_true, transf = 'u', stats = mat_data_stats, color = 'nrse', numit = 0, 
                          xlim = None, ylim = None, norms_ = None)



# check deployment: 
# - are the two paths path_data_name and lit_model_path the same? 
# -- they should be the same except for one pointing directly to the model.
# checked it - they are.







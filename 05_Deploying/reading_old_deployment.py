import os
import pickle

# path_depl = '05_Deploying\\data_out\\data_20250425_1426_casexx'


# with open(os.path.join(path_depl, 'mat_res_NN_sig.pkl'),'rb') as handle:
#                 mat_res_NN = pickle.load(handle)


# print(mat_res_NN)



# path_sim = '02_Simulator\Simulator\\results\saved_runs\data_20250604_1857_case10'
path_sim = '05_Deploying\data_out\data_20250714_1512_casexx'

# with open(os.path.join(path_sim, 'mat_res.pkl'), 'rb') as handle:
#         mat_res = pickle.load(handle)

with open(os.path.join(path_sim, 'mat_res_norm.pkl'), 'rb') as handle:
        mat_res = pickle.load(handle)
    
# item = 5
item = 0

print(  f"""
            Length L [mm]: {mat_res['L'][item]}
            Width B [mm]: {mat_res['B'][item]}
            Concrete class CC [-]: {mat_res['CC'][item]}
            Thickness [mm]: {mat_res['t_1'][item]}
            Force F [N/mm2]: {mat_res['F'][item]}
            Reinf.ratio [-]: {mat_res['rho'][item]}
            scenario [-]: {mat_res['s'][item]}""")
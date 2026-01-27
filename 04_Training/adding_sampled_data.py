# adding samples
import numpy as np
import os
import pickle
from datetime import datetime

from data_work import read_data
from plotting_sampled_data_utils import *

mat_data_names = {                                          #sorted in descending order (of epsilon ranges) --> it will be plotted like that
     #'2': '04_Training\\data\\data_20250328_1421_fake',
     #'3': '04_Training\\data\\data_20250428_1150_fake',
    # '4': '04_Training\\data\\data_20250428_1350_fake', 
    # '5': '04_Training\\data\\data_20250428_1438_fake',
    # '1': '04_Training\\data\\data_20250328_1431_fake',
    # '8': '04_Training\\data\\data_20250506_1902_fake',
    # '9': '04_Training\\data\\data_20250507_1826_fake',
    # '7': '04_Training\\data\\data_20250505_1217_fake',
    # '10': '04_Training\\data\\data_20250508_0916_fake',
    # '6': '04_Training\\data\\data_20250430_1501_fake',
    # '11': '04_Training\data\data_20250512_1000_fake',
    # '12': '04_Training\data\data_20250512_1009_fake',
    # '13': '04_Training\data\data_20250512_1035_fake',
    # '14': '04_Training\data\data_20250512_1728_fake', 
    # '15': '04_Training\data\data_20250512_1744_fake',
    # '16': '04_Training\data\data_20250515_1617_fake',
    # '17': '04_Training\data\data_20250515_2018_fake',
    # '18': '04_Training\data\data_20250516_0934_fake',
#     '20': '04_Training\data\data_20250516_1435_fake',
#     '22': '04_Training\data\data_20250521_1224_fake',
#     '23': '04_Training\data\data_20250522_1909_casexx',
#     '24': '04_Training\data\data_20250526_1019_casexx'
#     '25': '04_Training\data\data_20250526_2110_fake',
#     '26': '04_Training\data\data_20250527_1144_casexx',
#     '27': '04_Training\data\data_20250603_2149_fake',
#     '28': '04_Training\data\data_20250605_1144_fake',
#     '29': '04_Training\data\data_20250608_1748_fake',
#     '30': '04_Training\data\data_20250608_1832_fake',
#     'xx': '04_Training\data\data_20250608_1838_fake',
#     '32': '04_Training\data\data_20250815_1055_fake',
#     '33': '04_Training\data\data_20250815_1103_fake', 
#     '46': '04_Training\data\data_20250828_1607_fake',
#     '54': '04_Training\data\data_20250902_1039_fake',
#     '55': '04_Training\data\data_20250903_0924_fake',
#       '57': '04_Training\data\data_20250903_1059_fake',
#       '61': '04_Training\data\data_20251001_1641_fake',
#       '64': '04_Training\data\data_20251103_1552_fake', 
# '71': '04_Training\data\data_20251112_1851_fake'
# '72': '04_Training\data\data_20251112_1900_fake',
# '73': '04_Training\data\data_20251113_1802_fake',
# '78': '04_Training\data\data_20251117_1527_fake',
# '80': '04_Training\data\data_20251120_1813_fake',
# '81': '04_Training\data\data_20251121_1249_fake',
# '82': '04_Training\data\data_20251121_1245_fake',
# '83': '04_Training\data\data_20251121_1437_fake',
# 'xx': '04_Training\data\data_20251117_1529_fake',
# 'xxx': '04_Training\data\data_20251124_1437_fake',
# 'xxxx': '04_Training\data\data_20251124_1159_fake',
# '67': '04_Training\data\data_20251105_1603_fake',
# '85': '04_Training\data\data_20251124_1435_fake',
# '84': '04_Training\data\data_20251124_1155_fake',
# '92': '04_Training\data\data_20251128_1158_fake',
# '1': '04_Training\data\data_20251128_1053_fake_cleaned',
# '2': '04_Training\data\data_20251128_1123_fake_cleaned',
# '3': '04_Training\data\data_20251128_1152_fake_cleaned',
# '4': '04_Training\data\data_20251209_1501_fake'
# '105': '04_Training\data\data_20251209_1850_fake',
# '70': '04_Training\data\data_20251112_1055_fake'
# '107':'04_Training\data\data_20251215_1751_fake',
# '108':'04_Training\data\data_20251215_1540_fake',
# '109':'04_Training\data\data_20251212_1937_fake',
# '110':'04_Training\data\data_20251215_1007_fake',
# '111':'04_Training\data\data_20251215_1318_fake',
# '113': '04_Training\data\data_20251215_1916_fake',
# '114': '04_Training\data\data_20251216_0906_fake',
# '115': '04_Training\data\data_20251217_1210_fake',
# '116': '04_Training\data\data_20251217_1243_fake',
# '117': '04_Training\data\data_20251217_1322_fake',
# '118': '04_Training\data\data_20251217_1549_fake',
# '119': '04_Training\data\data_20251217_1713_fake',
# '120': '04_Training\data\data_20251217_1740_fake'
# '121': '04_Training\data\data_20251218_1912_fake',
# '122': '04_Training\data\data_20251219_0931_fake',
# '123': '04_Training\data\data_20251219_0859_fake',
# '129': '04_Training\data\data_20260105_1639_fake',
# '130': '04_Training\data\data_20260105_1416_fake',
# '131': '04_Training\data\data_20260105_1651_fake',
# '134': '04_Training\data\data_20260106_1333_fake',
# '135': '04_Training\data\data_20260106_1605_fake',
# '136': '04_Training\data\data_20260108_1017_fake',
# '137': '04_Training\data\data_20260108_1050_fake',
# '138': '04_Training\data\data_20260108_1520_fake',
# '142': '04_Training\data\data_20260108_1925_fake',
# '144': '04_Training\data\data_20260108_2137_fake',
# '145': '04_Training\data\data_20260109_1821_fake',
# '146': '04_Training\data\data_20260110_1141_fake',
'147': '04_Training\data\data_20260113_1403_fake',
'148': '04_Training\data\data_20260113_1611_fake',
}

mat_data_paths = get_paths_data(mat_data_names)

################## READ DATA #######################

numpy_data_dict = read_all_data(mat_data_paths)
print('Finished reading data')


################## COMBINE DATA #######################

new_data_eps_np, new_data_sig_np, new_data_t_np, new_data_De_np = combine_all_data(numpy_data_dict, n1 = 1000000, n2 = 500000, n3 = 3000000)
# new_data_eps_np[:,2] = new_data_eps_np[:,2] / 2  # this did not have the anticipated effect - remove it again for now


################## SAVE DATA #######################
# dump them into a new data folder

data_path = '04_Training\data'
folder_name = f"data_{datetime.now().strftime('%Y%m%d_%H%M')}_fake"
save_data_path = os.path.join(data_path, folder_name)
os.makedirs(save_data_path, exist_ok=True)


with open(os.path.join(save_data_path, 'new_data_t.pkl'), 'wb') as fp:
        pickle.dump(new_data_t_np.astype(np.float32), fp)
with open(os.path.join(save_data_path, 'new_data_eps.pkl'), 'wb') as fp:
        pickle.dump(new_data_eps_np.astype(np.float32), fp)
with open(os.path.join(save_data_path, 'new_data_sig.pkl'), 'wb') as fp:
        pickle.dump(new_data_sig_np.astype(np.float32), fp)
with open(os.path.join(save_data_path, 'new_data_De.pkl'), 'wb') as fp:
        pickle.dump(new_data_De_np.astype(np.float32), fp)  
print('Data saved to ', save_data_path)
print('Amount of data points', new_data_t_np.shape[0])
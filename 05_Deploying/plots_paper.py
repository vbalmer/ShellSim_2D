### executing plotting functions ####
from plots_paper_utils import *
import os


save_path = os.path.join(os.getcwd(), '05_Deploying\\plots\\test')
SCATTER = True



if SCATTER == True: 
    plot_scatter_paper('data_20260108_2137_fake', 'data_20260105_1639_fake', save_path)



'''
bav, 15.04.2025
create videos from diagonal plots in sequence
not working currently
'''



import os
from data_work_depl import create_vid, images_to_video

NAME = 'data_20250415_1536_casexx'


new_folder_path = os.path.join(os.getcwd(), '05_Deploying\\data_out\\'+NAME)
in_folder = os.path.abspath('05_Deploying\\plots\\diagonal_plots')
video_name = os.path.join(new_folder_path, 'diag_plots_video.avi')
# create_vid(in_folder, video_name, fps = 12)

images_to_video(in_folder, video_name, fps = 12)
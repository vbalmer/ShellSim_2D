# checking for surjectivity

import numpy as np
from collections import defaultdict
import os
from data_work import read_data

import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

def find_duplicate_row_indices(arr):
    # Round all elements to 1 decimal place
    rounded_arr = np.round(arr/100)*100
    
    # Find unique rows and their corresponding indices
    unique_rows, inverse_indices = np.unique(rounded_arr, axis=0, return_inverse=True)
    row_indices = defaultdict(list)

    for idx, group in enumerate(inverse_indices):
        row_indices[group].append(idx)

    # Filter out groups that only appear once
    duplicate_indices = {k: v for k, v in row_indices.items() if len(v) > 1}
    
    print('Duplicate values', list(duplicate_indices.values()))
    return list(duplicate_indices.values())


def find_mismatched_groups(arr_a, arr_b):
    """Find duplicate groups in arr_a where corresponding rows in arr_b are NOT identical."""
    duplicate_groups = find_duplicate_row_indices(arr_a)
    mismatched_groups = []

    for group in duplicate_groups:
        corresponding_rows = arr_b[group]  # Get corresponding rows in arr_b
        if not np.all(np.round(corresponding_rows, decimals=1) == np.round(corresponding_rows[0], decimals=1)):
            mismatched_groups.append(group)

    return mismatched_groups


def plot_heatmaps(arr_a, arr_b, mismatched_groups, path_plot):
    """Plot heatmaps for arr_a and arr_b, highlighting mismatched groups."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Convert mismatched groups to a row mask
    mismatch_mask = np.zeros(arr_a.shape[0], dtype=bool)
    for group in mismatched_groups:
        mismatch_mask[group] = True  # Highlight these rows

    # Heatmap for arr_a
    sns.heatmap(arr_a, cmap="Blues", cbar=False, ax=axes[0], linewidths=0.5, linecolor="black")
    axes[0].set_title("Array A (Duplicate Groups)")

    # Heatmap for arr_b (with mismatched rows highlighted)
    mask_shape = np.tile(~mismatch_mask[:, None], (1, arr_b.shape[1]))
    sns.heatmap(arr_b, cmap="Reds", cbar=False, ax=axes[1], linewidths=0.5, linecolor="black", mask=mask_shape)
    axes[1].set_title("Array B (Mismatched Rows Highlighted)")

    plt.show()
    plt.savefig(os.path.join(path_plot, 'heatmap_data_surjectivity.png'), dpi=200)

def plot_scatter(arr_a, mismatched_groups, path_plot):
    """Plot row indices for duplicate groups with mismatched ones highlighted."""
    plt.figure(figsize=(8, 6))

    for idx, group in enumerate(find_duplicate_row_indices(arr_a)):
        x_values = [idx] * len(group)
        y_values = group
        color = 'red' if group in mismatched_groups else 'blue'
        plt.scatter(x_values, y_values, label=f"Group {idx}" if idx == 0 else "", c=color)

    plt.xlabel("Duplicate Group Index")
    plt.ylabel("Row Index in Array A")
    plt.title("Duplicate Row Groups (Red = Mismatch in Array B)")
    plt.show()
    plt.savefig(os.path.join(path_plot, 'scatter_data_surjectivity.png'), dpi=200)


if __name__ == '__main__':
    data1 = '04_Training\data\data_20250303_1625_fake'
    path = os.getcwd()
    path_plot = os.path.join(path, '04_Training\\plots')
    path_data1 = os.path.join(path, data1)
    data1_eps_np = read_data(path_data1, 'eps')
    data1_t_np = read_data(path_data1, 't')
    data1_sig_np = read_data(path_data1, 'sig')

    # to check that the code is working, aritificially add two identical rows to the end of the dataset: 
    first_two_rows_sig = data1_sig_np[:2,:]
    first_two_rows_eps = data1_eps_np[:2,:]
    data1_sig_np_ctrl = np.vstack((data1_sig_np, first_two_rows_sig))
    data1_eps_np_ctrl = np.vstack((data1_eps_np, first_two_rows_eps))


    mismatched_eps = find_mismatched_groups(data1_sig_np_ctrl, data1_eps_np_ctrl)
    print('Mismatched eps', mismatched_eps)
    # plot_heatmaps(data1_sig_np, data1_eps_np, mismatched_eps, path_plot)
    # plot_scatter(data1_sig_np, mismatched_eps, path_plot)
import numpy as np
import pyDOE as doe
import pandas as pd
import math
import matplotlib.pyplot as plt
import os
import matplotlib.gridspec as gridspec
import time


class samplers:
    def __init__(self, parnames, min, max, samples, criterion):
        self.parnames = parnames
        self.min = min
        self.max = max
        self.samples = samples
        self.criterion = criterion

    def lhs(self):
        """
        Returns LHS samples.

        :param parnames: List of parameter names
        :type parnames: list(str)
        :param bounds: List of lower/upper bounds,
                        must be of the same length as par_names
        :type bounds: list(tuple(float, float))
        :param int samples: Number of samples
        :param str criterion: A string that tells lhs how to sample the
                                points. See docs for pyDOE.lhs().
        :return: DataFrame
        """
        
        bounds = np.vstack((self.min, self.max))
        bounds = bounds.T
        

        lhs = doe.lhs(len(self.parnames), samples=self.samples, criterion=self.criterion)
        par_vals = {}
        for par, i in zip(self.parnames, range(len(self.parnames))):
            par_min = bounds[i][0]
            par_max = bounds[i][1]
            par_vals[par] = lhs[:, i] * (par_max - par_min) + par_min

        # Convert dict(str: np.ndarray) to pd.DataFrame
        par_df = pd.DataFrame(columns=self.parnames, index=np.arange(self.samples))
        for i in range(self.samples):
            for p in self.parnames:
                par_df.loc[i, p] = par_vals[p][i]

        #logger = logging.getLogger(GA.__name__)
        #logger.info('Initial guess based on LHS:\n{}'.format(par_df))
        return par_df


class plots:
    def __init__(self, random):
        self.random = random
    
    def histogram(self, data, n_bins, parameter, path):
        '''
        ------------------------------------
        Plots simple histogram of given data
        ------------------------------------
        data        (np_array)  generated feature data
        n_bins      (integer)   amount of bins in graph
        parameter   (string)    parameter name
        '''
        fig, axs= plt.subplots(figsize = (8,5))
        axs.hist(data, bins=n_bins)
        axs.set(xlabel = parameter, ylabel = 'frequency')
        fig.savefig(os.path.join(path, 'hist_'+ parameter+'.png'))
        return fig
    


def hist_from_dict(data_dict, const_dict, save_path):
    
    n_cols = len(data_dict)//2 + 2

    # create figure
    fig = plt.figure(figsize=(3*n_cols, 8))
    gs = gridspec.GridSpec(2, n_cols, width_ratios=[1]*(n_cols-1)+[0.3])  # last column for info box

    # Plot each array in its own subplot
    for i, (label, array) in enumerate(data_dict.items()):
        row = i % 2  # Determine the row (0 or 1)
        col = i // 2   # Determine the column (0 or 1 in this layout)
        ax = fig.add_subplot(gs[row, col])
        
        # Plot data
        ax.hist(array, bins=20)
        ax.set_xlabel(label)
        ax.set_ylabel("Frequency")

    # Add the information box in the right column
    info_text = "Constant Values\n"+"\n".join([f"{key}: {value}" for key, value in const_dict.items()])
    info_ax = fig.add_subplot(gs[:, n_cols-1])  # Use both rows in the last column for the box
    info_ax.axis("off")  # Hide axes for the info box
    info_ax.text(
        0, 0.5, info_text,
        verticalalignment='center', horizontalalignment='left',
        fontsize=10, color='black',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='black')
    )

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.3)  # Add space between subplots and info box

    if save_path is not None:
        plt.savefig(os.path.join(save_path, 'hist_sampled.png'))
    plt.close()
    return



def wait_for_file(file_path):
    # Loop until the file exists
    while not os.path.exists(file_path):
        print(f"Waiting for file {file_path} to be saved...")
        time.sleep(1)  # Wait for 1 second before checking again
    print(f"File {file_path} detected!")
### functions for plotting nice figures for a paper ####

import pickle
import os

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl





def read_sampled_data(path):
    '''
    reads datafiles (pickle) created from sampling
    Args: 
        path (str)  path to pickle file

    Returns: 
        mat_data (dict) dict containing two np-arrays
                        each with shape (n,3) 
    '''
    
    mat_data = {}
    names = ['eps', 'sig']
    add_path = '04_Training\\data\\'
    for name in names:
        with open(os.path.join(os.getcwd(), add_path + path+ '\\new_data_'+name+'.pkl'),'rb') as handle:
            mat_data[name] = pickle.load(handle)
        mat_data[name] = mat_data[name][:,:3]

    # convert eps-units to [‰]
    mat_data['eps'] = mat_data['eps']*1e3

    return mat_data

def setup_figure_format():
    # mpl.rcParams["text.usetex"] = True
    mpl.rcParams["text.latex.preamble"] = (
        r"\usepackage{mathptmx}\usepackage{amsmath}\usepackage{upgreek}"
    )
    mpl.rcParams["font.family"] = "Times New Roman"
    mpl.rcParams["mathtext.rm"] = "Times New Roman"
    mpl.rcParams["mathtext.it"] = "Times New Roman:italic"
    mpl.rcParams["mathtext.bf"] = "Times New Roman:bold"
    mpl.rcParams["font.size"] = 10
    mpl.rcParams['mathtext.fontset'] = 'stix'

    return

def plotting_scatter(fig, mat_all, labels, vars, p):
    ncols = len(vars)
    colors = get_colorscale(len(labels))
    axs = [fig.add_subplot(1, ncols, i + 1, projection='3d')
       for i in range(ncols)]
    for i, var in zip(range(ncols), vars):
        for name, color in zip(labels, colors):
            
            axs[i].scatter(mat_all[name][var][::p[name],0], mat_all[name][var][::p[name],1], mat_all[name][var][::p[name],2], 
                            label = name+', $N$ = '+str(int(np.round(mat_all[name][var].shape[0]/1000,0))) + '$\cdot$ 10$^3$', 
                            color = color,  edgecolors='none', alpha = 0.2, s = 2)

    return axs

def final_touches(fig, axs):
    # Axes labels
    axs[0].set_xlabel(r"ε$_x$ [‰]", fontname="Times New Roman")
    axs[0].set_ylabel(r"ε$_y$ [‰]", fontname="Times New Roman")
    axs[0].set_zlabel(r"γ$_{xy}$ [‰]", fontname="Times New Roman")
    axs[1].set_xlabel(r"$\it{n}$$_x$ [kN/m]", fontname="Times New Roman")
    axs[1].set_ylabel(r"$\it{n}$$_y$ [kN/m]", fontname="Times New Roman")
    axs[1].set_zlabel(r"$\it{n}$$_{xy}$ [kN/m]", fontname="Times New Roman")

    # Set maxima and minima
    axs[0].set_xlim(-5,50)
    axs[0].set_ylim(-5,50)
    axs[0].set_zlim(-100,100)
    axs[1].set_xlim(-11000,2000)
    axs[1].set_ylim(-11000,2000)
    axs[1].set_zlim(-4500,4500)

    # Turn in the right way
    axs[0].view_init(elev=30, azim=30)
    axs[1].view_init(elev=30, azim=30)

    # Legend
    fig.subplots_adjust(bottom=0.2, left=0.32, right=0.95, top=0.9, wspace = 0.5)
    handles, labels = [], []
    handles, labels = axs[0].get_legend_handles_labels()
    
    legend = fig.legend(
        handles,
        labels,
        loc='lower center',
        ncol=2,
        bbox_to_anchor=(0.5, 0.02)
    )
    for handle in legend.legend_handles:
        handle.set_sizes([20])
        handle.set_alpha(1) 

    for ax in axs:
        # Layouting gridlines / axes 
        ax.grid(False)

        # ensuring all bounding box lines are plotted.
        for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
            pane.set_facecolor((1,1,1,0))
            pane.set_edgecolor("black")
            pane.set_linewidth(0)
            pane.fill = False
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis.line.set_color("black")
            axis.line.set_linewidth(0)
        
        # draw missing line:
        xmin, xmax = ax.get_xlim3d()
        ymin, ymax = ax.get_ylim3d()
        zmin, zmax = ax.get_zlim3d()

        # 8 corners of the box
        corners = np.array([[xmin, ymin, zmin],
                            [xmax, ymin, zmin],
                            [xmax, ymax, zmin],
                            [xmin, ymax, zmin],
                            [xmin, ymin, zmax],
                            [xmax, ymin, zmax],
                            [xmax, ymax, zmax],
                            [xmin, ymax, zmax]])

        # edges defined by corner indices
        edges = [
            [0,1],[1,2],[2,3],[3,0],  # bottom
            [4,5],[7,4], #[5,6],[6,7],  # top
            [0,4],[1,5],[3,7]#[2,6]   # verticals
        ]

        for e in edges:
            ax.plot(*corners[e].T, color='black', linewidth=0.5)
    
    fig.patch.set_alpha(0)
    # fig.tight_layout(pad = 5)
    
        
    return fig, axs

def get_colorscale(n, cmap_name="viridis"):
        """
        Returns a list of n colors from a given colormap.

        Parameters:
        - n: Number of colors
        - cmap_name: Name of matplotlib colormap

        Returns:
        - List of RGBA colors
        """
        cmap = plt.get_cmap(cmap_name)
        n = n + 1
        return [cmap(i / (n - 1)) for i in range(n)]

def save_figure(fig, save_path, plotname):
    if save_path is not None:
        full_path = os.path.join(save_path, plotname + '.svg')
        fig.savefig(full_path , bbox_inches='tight')    
        print(f'Saved figure {plotname} at {save_path}')
    return




def plot_scatter_paper(data1, data2, save_path):
    '''
    plots scatter plot in 3D for the paper 
    data1: Uniform data
    data2: Log data
    
    '''

    # 1 - Read sampled data
    mat_all = {}
    labels = ['Uniform', 'Log.']
    mat_all[labels[0]] = read_sampled_data(data1)
    mat_all[labels[1]] = read_sampled_data(data2)

    # 2 - Set up figure
    setup_figure_format()
    fig = plt.figure(figsize = (16/2.54, 8/2.54))
    vars = ['eps', 'sig']
    
    # 3 - Plot data in figure
    plotevery = {
        labels[0]: 500,
        labels[1]: 200
    }
    axs = plotting_scatter(fig, mat_all, labels, vars, plotevery)
    
    # 4 - Figure final touches
    fig, axs = final_touches(fig, axs)

    # 5 - Save figure
    plotname = 'scatter_data_simple'
    save_figure(fig, save_path, plotname)

    return
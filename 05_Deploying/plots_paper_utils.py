### functions for plotting nice figures for a paper ####

import pickle
import os
import sys
import pathlib

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec
import matplotlib as mpl
import matplotlib.colors as mcolors
from matplotlib.ticker import StrMethodFormatter, MaxNLocator
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.patches import FancyArrowPatch
import wandb
from matplotlib.patches import ConnectionStyle
from matplotlib.transforms import IdentityTransform

import pandas as pd
import openpyxl

from stress_paths_plots import sample_idx_eps, calculate_sig_D_NLFEA, predict_sig_D_NN
from load_paths_plots import plot_load_paths, trim_vectors, get_loadsteps, get_max_eps, calculate_errors_diagonal, get_max_displ



sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "04_Training"))

from test_utils import make_prediction, test_model_instance
from call_light import predict_D, load_data
from data_work import transf_units




''' AUXILIARY FUNCTIONS SCATTER DATA '''


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
    # convert sig-units to [MN/m]
    mat_data['sig'] = mat_data['sig']*1e-3

    return mat_data

def setup_figure_format():
    mpl.rcParams["text.usetex"] = True
    mpl.rcParams["text.latex.preamble"] = (
        r"\usepackage{newtxtext}\usepackage{newtxmath}\usepackage{amsmath}\usepackage{upgreek}\usepackage{gensymb}\usepackage{textcomp}"
    )
    # mpl.rcParams["text.latex.preamble"] = (r"\usepackage{gensymb}\usepackage{textcomp}")    
    mpl.rcParams["font.family"] = "Times New Roman"
    mpl.rcParams["mathtext.rm"] = "Times New Roman"
    mpl.rcParams["mathtext.it"] = "Times New Roman:italic"
    mpl.rcParams["mathtext.bf"] = "Times New Roman:bold"
    mpl.rcParams["font.size"] = 8
    mpl.rcParams['mathtext.fontset'] = 'stix'

    return

def plotting_scatter(axs, mat_all, labels, vars, p):
    ncols = len(vars)
    colors = get_colorscale(2)
    
    for i, var in zip(range(ncols), vars):
        for name in labels:
            if name == 'Log.':
                fc = (colors[1][:3],0.2)
                ec = (colors[1][:3], 0.8)
            elif name == 'Uniform': 
                fc = (colors[0][:3], 0.2)
                ec = (colors[0][:3], 0.8)
            
            axs[i].scatter(mat_all[name][var][::p[name],0], mat_all[name][var][::p[name],1], mat_all[name][var][::p[name],2], 
                            label = name+', $N$ = '+str(int(np.round(mat_all[name][var].shape[0]/1000,0))) + '$\cdot$ 10$^3$', 
                            facecolor = fc,  edgecolors=ec, linewidth = 0.5, s = 4)


    return axs

def final_touches(fig, axs):
    # Axes labels
    axs[0].set_xlabel(r'$\upvarepsilon_x$ [\textperthousand]') #, fontname="Times New Roman")
    axs[0].set_ylabel(r'$\upvarepsilon_y$ [\textperthousand]') #, fontname="Times New Roman")
    axs[0].set_zlabel(r'$\upgamma_{xy}$ [\textperthousand]') #, fontname="Times New Roman")
    axs[1].set_xlabel(r'$\it{n}$$_x$ [MN/m]') #, fontname="Times New Roman")
    axs[1].set_ylabel(r'$\it{n}$$_y$ [MN/m]') #, fontname="Times New Roman")
    axs[1].set_zlabel(r'$\it{n}$$_{xy}$ [MN/m]') #, fontname="Times New Roman")

    # Set maxima and minima
    axs[0].set_xlim(-5,50)
    axs[0].set_ylim(-5,50)
    axs[0].set_zlim(-100,100)
    axs[1].set_xlim(-11,2)
    axs[1].set_ylim(-11,2)
    axs[1].set_zlim(-4.5,4.5)

    # # Turn in the right way
    # axs[0].view_init(elev=30, azim=30)
    # axs[1].view_init(elev=30, azim=30)

    # Legend
    fig.subplots_adjust(bottom=0.15, left=0.05, right=0.95, top=0.95, wspace = 0.1)
    handles, labels = [], []
    handles, labels = axs[0].get_legend_handles_labels()
    
    legend = fig.legend(
        handles,
        labels,
        loc='lower center',
        ncol=2,
        bbox_to_anchor=(0.5, 0.02)
    )
    legend.get_frame().set_linewidth(0.5)
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

        # move tick labels closer to the plot
        ax.tick_params(pad=0)
        ax.tick_params(axis='both', which='both', direction = 'in')
        ax.tick_params(axis="both", which="both", width=0.5, length = 0.5)
        
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
            [6,7],[7,4], # [4,5],[5,6],  # top
            [0,4],[3,7],[2,6], #[1,5]   # verticals
        ]

        for e in edges:
            ax.plot(*corners[e].T, color='black', linewidth=0.5)

        # make sure that labels don't have numbers after decimal point.
        ax.xaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
        ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.0f}')) 

        # label numbers
        labels_text = ['(a)', '(b)']
        for ax, label in zip(axs, labels_text):
            ax.text2D(0.05, 0.95, label, transform=ax.transAxes, 
                    fontweight='bold', va='top', ha='left')
            
    fig.patch.set_alpha(0)
    
        
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

def save_figure(fig, save_path, plotname, bbox = 'tight'):
    if save_path is not None:
        full_path = os.path.join(save_path, plotname + '.svg')
        fig.savefig(full_path , bbox_inches=bbox)    
        print(f'Saved figure {plotname} at {save_path}')
    return


''' AUXILIARY FUNCTIONS TRAINING '''
def add_subplots_gs(fig, gs):
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[0, 3])

    ax5 = fig.add_subplot(gs[1, 0])
    ax6 = fig.add_subplot(gs[1, 1])
    ax7 = fig.add_subplot(gs[1, 2])
    ax8 = fig.add_subplot(gs[1, 3])

    ax_merged = fig.add_subplot(gs[2, 0:2])
    ax_merged2 = fig.add_subplot(gs[2,2:4])

    ax_merged2.axis('off')

    axs = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax_merged, ax_merged2]

    fig.subplots_adjust(
        left=0.1,
        right=0.97,
        top=0.95,
        bottom=0.05
    )

    add_subplot_number_tr(axs)

    # ax_merged2.text(
    #     0.5, 0.5, 
    #     'e: $RMSE$ in the corresponding units',
    #     transform=ax_merged2.transAxes,
    #     ha='center', va='center',
    #     bbox=dict(facecolor=(1,1,1,0.5), edgecolor='black', linewidth = 0.5, boxstyle='round,pad=0.5')
    # )

    return fig, axs

def add_legend(axs):
    handles, labels = [], []
    for ax in [axs[0], axs[4]]:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    
    legend = axs[-1].legend(
        handles,
        labels,
        loc='center',
        ncol=1,
        bbox_to_anchor=(0.5, 0.5)
    )
    legend.get_frame().set_linewidth(0.5)
    for i, handle in enumerate(legend.legend_handles):
        if i == 0:
            handle.set_sizes([10])
            handle.set_alpha(1) 

def add_subplot_number_tr(axs):
    labels_text = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)']
    for ax, label in zip(axs[:-1], labels_text):  # Skip the last one (ax_merged2)
        ax.text(-0.1, 1.1, label, transform=ax.transAxes, 
                fontsize=8, fontweight='bold', va='top', ha='left')
    return

def save_data_test(plot_data_orig, plot_data_d_orig, path_train, NN):
    plot_data_test = {
        'sig': plot_data_orig,
        'D': plot_data_d_orig
    }
    filename = os.path.join(os.path.join(path_train + '\\new_data\\_simple_logs\\v_'+NN[0]), 'plot_data_test.pkl')
    with open(filename, 'wb') as fp:
        pickle.dump(plot_data_test, fp)

    print(f'Saved data at {filename}')
    
    return

def load_data_test(path_train, NN):
    with open(os.path.join(os.path.join(path_train +'\\new_data\\_simple_logs\\v_'+NN[0]), 'plot_data_test.pkl'),'rb') as handle:
            plot_data_test = pickle.load(handle)
    plot_data_orig = plot_data_test['sig']
    plot_data_d_orig = plot_data_test['D']
    
    return plot_data_orig, plot_data_d_orig

def get_plot_data(NN, include_predict):
    path_train = os.path.join(os.getcwd(), '04_Training')
    if include_predict: 
        # get relevant data
        wandb.init(mode="offline")
        data_model = load_data(path_train, only_test = True, add_path = '_simple_logs\\v_'+NN[0])
        inp = data_model['inp']
        model_test_dict = test_model_instance(inp, path_train, v_num=NN[0], epoch=NN[1])
        transf_type_ = 'st-stitched'
        SCALE, DOUBLE_NORM = False, False

        # make prediction for sigma
        plot_data_orig = make_prediction(inp, model_test_dict, data_model, transf_type = transf_type_, sc= SCALE, dn = DOUBLE_NORM)

        # make prediction for D
        data_model['eval_model'] = model_test_dict['standard']
        plot_data_d_orig = predict_D(data_model, transf_type=transf_type_,sc=SCALE, dn=DOUBLE_NORM)

        save_data_test(plot_data_orig, plot_data_d_orig, path_train, NN)

        plot_data_u['sim'] = transf_units(plot_data_orig['all_test_labels'], 'sig', forward = False)
        plot_data_u['NN'] = transf_units(plot_data_orig['all_predictions'], 'sig', forward = False)
        plot_data_d_u['sim'] = transf_units(plot_data_d_orig['D_sim'], 'D', forward = False, linel = False)
        plot_data_d_u['NN'] = transf_units(plot_data_d_orig['D_pred'], 'D', forward = False, linel = False)

    else:
        plot_data_orig, plot_data_d_orig = load_data_test(path_train, NN)
        plot_data_u, plot_data_d_u = {}, {}
        plot_data_u['sim'] = transf_units(plot_data_orig['all_test_labels'], 'sig', forward = False)
        plot_data_u['NN'] = transf_units(plot_data_orig['all_predictions'], 'sig', forward = False)
        plot_data_d_u['sim'] = transf_units(plot_data_d_orig['D_sim'], 'D', forward = False, linel = False)
        plot_data_d_u['NN'] = transf_units(plot_data_d_orig['D_pred'], 'D', forward = False, linel = False)


    return plot_data_u, plot_data_d_u

def final_touches_diag_scatter(axs):
    # set labels
    pad = 2
    axs[0].set_xlabel(r'$n_x$ [MN/m]', labelpad = pad)
    axs[0].set_ylabel(r'$\tilde{n}_x$ [MN/m]', labelpad = pad)
    axs[1].set_xlabel(r'$n_y$ [MN/m]', labelpad = pad)
    axs[1].set_ylabel(r'$\tilde{n}_y$ [MN/m]', labelpad = pad)
    axs[2].set_xlabel(r'$D_{m,11}$ [GN/m]', labelpad = pad)
    axs[2].set_ylabel(r'$\tilde{D}_{m,11}$ [GN/m]', labelpad = pad)
    axs[3].set_xlabel(r'$D_{m,12}$ [GN/m]', labelpad = pad)
    axs[3].set_ylabel(r'$\tilde{D}_{m,12}$ [GN/m]', labelpad = pad)
    
    # add grid
    for i in range(6):
        axs[i].grid(True)

    # adjust linewidths
    lw = 0.5
    for ax in axs:
        for spine in ax.spines.values():
            spine.set_linewidth(lw)
        ax.tick_params(width=lw)
        ax.grid(True, linewidth=lw)
        ax.tick_params(pad=2)
        ax.tick_params(axis='both', which='both', direction = 'in')
        ax.tick_params(axis="both", which="both", width=0.5, length = 2)



    # for cbar in cbars:
    #     cbar.outline.set_linewidth(lw)
    #     cbar.ax.tick_params(width=lw)
    #     cbar.solids.set_alpha(0.8)
    #     cbar.ax.text(
    #         0.5, 1.05,
    #         '$RSE$',
    #         ha = 'center', va = 'bottom',
    #         transform = cbar.ax.transAxes
    #     )
    # for ax_cb in [axs[2], axs[5]]:
    #     pos = ax_cb.get_position()
    #     ax_cb.set_position([pos.x0-0.04, pos.y0, pos.width, pos.height])

    return axs

def add_colorbar(errors, d_unit, s_unit):
    vmin1 = np.min(errors['rse'][:,0:2]*s_unit)
    vmax1 = np.max(errors['rse'][:,0:2]*s_unit)
    norms1 = mcolors.Normalize(vmin=vmin1,vmax=vmax1)

    vmin2 = np.min(errors['rse'][:,2:4]*d_unit)
    vmax2 = np.max(errors['rse'][:,2:4]*d_unit)
    norms2 = mcolors.Normalize(vmin=vmin2,vmax=vmax2)
    
    return [norms1, norms1, norms2, norms2]

def calculate_errors(plot_data_u, plot_data_d_u):
    num_cols_plt = 4

    r_squared2 = np.zeros((1,num_cols_plt))
    rse = np.zeros((plot_data_u['sim'].shape[0], num_cols_plt))
    rmse, aux_ =np.zeros((1,num_cols_plt)), np.zeros((1,num_cols_plt))

    Y_col = np.vstack((plot_data_u['sim'][:,0], plot_data_u['sim'][:,1], plot_data_d_u['sim'][:,0,0], plot_data_d_u['sim'][:,0,1])).T
    Y_pred = np.vstack((plot_data_u['NN'][:,0], plot_data_u['NN'][:,1], plot_data_d_u['NN'][:,0,0], plot_data_d_u['NN'][:,0,1])).T

    for i in range(num_cols_plt):
        r_squared2[:,i] = np.corrcoef(Y_col[:,i], Y_pred[:,i])[0, 1]**2
        rse[:,i] = np.sqrt((Y_pred[:,i]-Y_col[:,i])**2)
        rmse[:,i] = np.sqrt(np.mean((Y_pred[:,i] - Y_col[:,i]) ** 2))
        # aux_[:,i] = np.sqrt(np.mean((mean_train[:,i]*np.ones(Y_col.shape) - Y_col) ** 2))

        errors = {
            'rse': rse,
            'rmse': rmse,
            # 'rrmse': rrmse
            'r_squared2':r_squared2
        }


    return errors

def plot_diagonal_scatter(fig, axs, NN, include_predict = False):
    # 1 - collect desired data
    plot_data_u, plot_data_d_u = get_plot_data(NN, include_predict)   
    n_every = 100
    sig_sim = plot_data_u['sim'][::n_every]
    sig_pred = plot_data_u['NN'][::n_every]
    D_sim = plot_data_d_u['sim'][::n_every]
    D_pred = plot_data_d_u['NN'][::n_every]

    # 2 - calculate errors
    errors = calculate_errors(plot_data_u, plot_data_d_u)

    # 3 - plot data
    d_unit = 1e-6
    s_unit = 1e-3
    units = [s_unit, s_unit, d_unit, d_unit]
    # norms = add_colorbar(errors, d_unit, s_unit)
    cmap1 = plt.cm.plasma
    values = np.linspace(0.2,0.8,10)
    colors = [cmap1(v) for v in values]
    scatters = []
    Y_sim = np.array([sig_sim[:,0]*s_unit, sig_sim[:,1]*s_unit, D_sim[:,0,0]*d_unit, D_sim[:,0,1]*d_unit])
    Y_pred = np.array([sig_pred[:,0]*s_unit, sig_pred[:,1]*s_unit, D_pred[:,0,0]*d_unit, D_sim[:,0,1]*d_unit])
    axs_ = [axs[0], axs[1], axs[2], axs[3]]
    for i in range(4):
        scatter = axs_[i].scatter(Y_sim[i], Y_pred[i], s= 2, alpha = 0.2, edgecolors = 'none',facecolor=colors[0],
                                label = f'Test Data, $N$ = {Y_pred[0].shape[0]*n_every*1e-3:.0f} $\cdot 10^{3}$')
                                #  c = errors['rse'][::n_every,i]*units[i], cmap = 'plasma', norm = norms[i])
        axs_[i].plot([np.min([np.min(Y_sim[i]), np.min(Y_pred[i])]), np.max([np.max(Y_sim[i]), np.max(Y_pred[i])])], 
                    [np.min([np.min(Y_sim[i]), np.min(Y_pred[i])]), np.max([np.max(Y_sim[i]), np.max(Y_pred[i])])],
                                  color='white', linestyle='--', linewidth = 0.5)
        scatters.append(scatter)
        axs_[i].text(0.4, 0.08, f'$RMSE$ = {errors["rmse"][0, i]*units[i]:.2f}',transform=axs_[i].transAxes,
                color='black', bbox=dict(facecolor=(1, 1, 1, 0.5), linewidth = 0.5, edgecolor = 'black', boxstyle='round,pad=0.15'))

    # cbars = []
    # cbar0 = fig.colorbar(scatters[0], cax=axs[2], orientation='vertical')
    # cbar1 = fig.colorbar(scatters[2], cax=axs[5], orientation='vertical')
    # cbars.append(cbar0)
    # cbars.append(cbar1)
    
    
    return axs #, cbars



def get_data_physics_influence():
    mat_data = {
        'points': np.array([3300, 33, 3, 0.65]),
        'sobolev_rrmse': np.array([0.6, 2, 6, 6]),
        'sobolev_rmse': np.array([21, 70, 213, 215]),
        'no_sobolev_rrmse':np.array([0.2, 1, 6, 9]),
        'no_sobolev_rmse':np.array([8, 30, 206, 292]),
    }
    return mat_data

def plot_physics_influence(ax, mat_data):
    ax.plot(mat_data['points']*1e3, mat_data['sobolev_rmse']*1e-3, lw = 0.5, marker='o', ms = '2', color = 'black', label = 'Sobolev')
    ax.plot(mat_data['points']*1e3, mat_data['no_sobolev_rmse']*1e-3, lw = 0.5, marker='o', ms = '2',color = 'black', linestyle = 'dashed', label = 'No Soblev')
    ax.set_ylabel('max. $RMSE$ $\\hat{\\upsigma}$ [MN/m]')
    ax.set_xlabel('$N$ [-]', labelpad = 2)
    ax.set_xscale('log')
    leg = ax.legend(frameon = True)
    leg.get_frame().set_linewidth(0.5)
    ax.set_ylim(0, )

    ax.annotate('NN v480', 
        xy=(mat_data['points'][0]*1e3, mat_data['sobolev_rmse'][0]*1e-3),              # Point to annotate
        xytext=(mat_data['points'][0]*1e3-2.7e6, mat_data['sobolev_rmse'][0]*1e-3+0.07),          # Where to place the text
        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=-0.3', lw=0.25),
        bbox=dict(boxstyle='round', facecolor=(1,1,1,0.5), edgecolor='lightgrey', 
              linewidth=0.5))


    return


def plot_stress_paths(axs, NN, geom = [300, 0.025, 0.025, 1]):

    # Step 1: determine the values to be plotted
    path = os.path.join(os.getcwd(), '04_Training\\new_data\\_simple_logs\\v_'+NN[0])
    inp_vector = sample_idx_eps([0], geom = geom, model_path = path, model_dim = 'TWODIM', range_factor = 1, small_value = 1e-9)
    sig_D_NLFEA = calculate_sig_D_NLFEA(inp_vector, rho_sublayer = True)
    sig_D_NN = predict_sig_D_NN(inp_vector, path, epnum = NN[1], NN_comp = None, model_dim = 'TWODIM')
    
    
    # Step 2: Plot desired values
    d_unit = 1e-6
    s_unit = 1e-3
    e_unit = 1e3

    cmap1 = plt.cm.plasma
    values = np.linspace(0.2,0.8,3)
    colors = [cmap1(v) for v in values]

    kwargs_NN = {
        'color': colors[0],
        'lw': 0.5,
        'linestyle': 'dashed',
        'label':'NN prediction',
        }
    kwargs_NLFEA = {
        'color': colors[2],
        'lw': 0.5,
        'linestyle': 'solid',
        'label':'NLFEA calculation',
    }

    for key in inp_vector.keys():
        if inp_vector[key] is not None:
            eps = inp_vector[key][:,0]*e_unit
            axs[0].plot(eps, sig_D_NLFEA[key]['sh_NLFEA'][:,0,0]*s_unit, **kwargs_NLFEA)
            axs[0].plot(eps, sig_D_NN[key]['sh_NN'][:,0]*s_unit, **kwargs_NN)
            axs[1].plot(eps, sig_D_NLFEA[key]['sh_NLFEA'][:,1,0]*s_unit, **kwargs_NLFEA)
            axs[1].plot(eps, sig_D_NN[key]['sh_NN'][:,1]*s_unit, **kwargs_NN)
            axs[2].plot(eps, sig_D_NLFEA[key]['D_NLFEA'][:,0,0]*d_unit, **kwargs_NLFEA)
            axs[2].plot(eps, sig_D_NN[key]['D_NN'][:,0,0]*d_unit, **kwargs_NN)
            axs[3].plot(eps, sig_D_NLFEA[key]['D_NLFEA'][:,0,1]*d_unit, **kwargs_NLFEA)
            axs[3].plot(eps, sig_D_NN[key]['D_NN'][:,0,1]*d_unit, **kwargs_NN)
    return

def final_touches_stress_path(axs):
    pad = 2
    for i in range(4):
        axs[i].set_xlabel(r'$\upvarepsilon_x$ [‰]', labelpad = pad)
    axs[0].set_ylabel(r'$n_x$ [MN/m]', labelpad = pad)
    axs[1].set_ylabel(r'$n_y$ [MN/m]', labelpad = pad)
    axs[2].set_ylabel(r'$D_{m,11}$ [GN/m]', labelpad = pad)
    axs[3].set_ylabel(r'$D_{m,12}$ [GN/m]', labelpad = pad)

    axs[1].set_ylim(-0.1, 0.1)
    axs[3].set_ylim(-0.2, 0.2)
    axs[2].set_ylim(-0.4,12)
    
    for i in range(4):
        axs[i].set_xlim(-3,5)
    for ax in axs:
        ax.tick_params(pad=2)
        ax.tick_params(axis='both', which='both', direction = 'in')
        ax.tick_params(axis="both", which="both", width=0.5, length = 2)

    # make sure not more than 2 decimals after point
    axs[1].yaxis.set_major_formatter(StrMethodFormatter('{x:.1f}'))
    axs[2].yaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
    axs[3].yaxis.set_major_formatter(StrMethodFormatter('{x:.1f}'))

    # make sure only three ticks per y-axis in n_y and D_m12 plots:
    axs[1].yaxis.set_major_locator(MaxNLocator(nbins=3))
    axs[3].yaxis.set_major_locator(MaxNLocator(nbins=3))


    return



''' AUXILIARY FUNCTIONS DEPLOYING '''

def add_subplots_gs_depl(fig, gs):
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])

    ax7 = fig.add_subplot(gs[2, 0])
    ax8 = fig.add_subplot(gs[2, 1])
    ax9 = fig.add_subplot(gs[2, 2])

    ax10 = fig.add_subplot(gs[3, 0])
    ax11 = fig.add_subplot(gs[3, 1])
    ax12 = fig.add_subplot(gs[3, 2])

    axs = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12]

    for ax in axs:
        ax.grid(True, lw = 0.5)

    # adjust linewidths
    lw = 0.5
    for ax in axs:
        for spine in ax.spines.values():
            spine.set_linewidth(lw)
        ax.tick_params(width=lw)
        ax.grid(True, linewidth=lw)
    
    add_subplot_number_depl(axs)

    return fig, axs

def add_subplot_number_depl(axs):
    labels_text = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)', '(j)', '(k)', '(l)']
    for ax, label in zip(axs, labels_text):
        ax.text(-0.1, 1.1, label, transform=ax.transAxes, 
                fontsize=8, fontweight='bold', va='top', ha='left', clip_on=False)
    return

def plot_base_cases(axs, path_depl, thresh, type_ = 'eps', color = None):
    load_steps = {}
    mat_displ = {}
    errors = {}
    for key0, i in zip(path_depl.keys(), range(3)): 
        if key0 == '2D-8C': 
            pass # only consider base study cases
        else:
            load_steps[key0]= {}
            mat_displ[key0] = {}
            errors[key0] = {}
            for key1 in path_depl[key0].keys(): # rho_y_i
                load_steps[key0][key1], mat_displ[key0][key1] = plot_load_paths('05_Deploying\\data_out\\'+path_depl[key0][key1], case_study = key0, 
                                                                                    until_load_level = None, save_path = None, type = type_)
                if color is None:
                    # only determine the suitable max load for the base cases (the function doesn't work for combined cases)
                    load_steps[key0][key1] = suitable_max_load(mat_displ, load_steps, thresh, key0, key1)
                if key0 == '2D-5':
                    # remove last two points for pure compression in x-direction.
                    load_steps[key0][key1] = load_steps[key0][key1][:-2]
                    mat_displ[key0][key1]['NN'] = mat_displ[key0][key1]['NN'][:-2]
                    mat_displ[key0][key1]['NLFEA'] = mat_displ[key0][key1]['NLFEA'][:-2]
            if color is not None: 
                # sensitivity plot:
                plot_individual_base_case_sensitivity(axs[i], key0, load_steps, mat_displ, thresh, color = color[i])
                key_star = list(path_depl[key0].keys())[0]
                errors[key0][key_star] = {}
                mat_displ[key0][key_star]['NLFEA'] = mat_displ[key0][key_star]['NLFEA'][:-1]
                mat_displ[key0][key_star]['NN'] = mat_displ[key0][key_star]['NN'][:-1]
                for dim in range(2):    
                    errors[key0][key_star][str(dim)] = calculate_errors_diagonal(mat_displ, key0, key1=key_star, i = dim)
                axs[i].text(
                        0.45, 0.2,                 # x,y position in axes coordinates (0–1)
                        r"$RMSE$ $\hat{\upvarepsilon}_x = $ " + f"{np.round(errors[key0][key_star]['0']['rmse'][0][0], 2):.2f} \n"+
                        r"$RMSE$ $\hat{\upgamma}_{xy} = $ " + f"{np.round(errors[key0][key_star]['1']['rmse'][0][1], 2):.2f}",
                        transform=axs[i].transAxes,     # makes coordinates relative to the axes
                        verticalalignment="top",
                        bbox=dict(
                            boxstyle="round,pad=0.2",
                            fc="white",
                            ec="black",
                            lw =0.5,
                            alpha=0.8
                        )
                )
            else: 
                # deployment plot:
                plot_individual_base_case(axs[i], key0, load_steps, mat_displ, thresh)
                

    return

def suitable_max_load(mat_displ, load_steps, thresh, key0, key1):
    idx_NLFEA = trim_vectors(mat_displ[key0][key1]['NLFEA'][:,0], threshold = thresh)
    idx_NN = trim_vectors(mat_displ[key0][key1]['NN'][:,0], threshold = thresh)
    idx = min(idx_NLFEA, idx_NN)
    mat_displ[key0][key1]['NN'] = mat_displ[key0][key1]['NN'][:idx,0]
    mat_displ[key0][key1]['NLFEA'] = mat_displ[key0][key1]['NLFEA'][:idx,0]
    load_steps[key0][key1] = load_steps[key0][key1][:idx]
    print(f'First and last load level for case {key0} and {key1} are {load_steps[key0][key1][0]} kN/m and {load_steps[key0][key1][-1]} kN/m')
    return load_steps[key0][key1]

def plot_individual_base_case(ax, key0, load_steps, mat_displ, thresh, type_ = 'eps', color = None):
    if color is None:
        cmap1 = plt.cm.viridis
        values = np.linspace(0.2,0.8,3)
        colors = [cmap1(v) for v in values]
    else: 
        colors = color
    

    unit_s = 1e-3

    if type_ != 'eps':
        raise UserWarning('This plot is not implemented for plotting displacements u, please use type = "eps".')

    for j, key1 in enumerate(load_steps[key0].keys()):
        ax.plot(mat_displ[key0][key1]['NLFEA'], np.abs(load_steps[key0][key1])*unit_s, color = colors[j], 
                marker = 'o', markeredgewidth=0.5, markerfacecolor = 'none', ms = 3, lw = 0.5, label = 'NLFEA, ' +key1)
        ax.plot(mat_displ[key0][key1]['NN'], np.abs(load_steps[key0][key1])*unit_s, color = colors[j], 
                marker = 'x', ms = 3, markeredgewidth=0.5, lw = 0.5, linestyle = '--', label = 'NN, ' +key1)

    return

def final_touches_base_cases(fig, ax):
    ax[0].set_xlabel(r'$\hat{\upgamma}_{xy}$ [‰]')
    ax[0].set_ylabel('$n_{xy}$ [MN/m]')
    ax[1].set_xlabel(r'$\hat{\upvarepsilon}_{y}$ [‰]')
    ax[1].set_ylabel('$n_{y}$ [MN/m]')
    ax[2].set_xlabel(r'-$\hat{\upvarepsilon}_{x}$ [‰]')
    ax[2].set_ylabel('-$n_{x}$ [MN/m]')

    # ensure no numbers after decimal point
    ax[0].yaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
    ax[1].yaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
    ax[2].yaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
    ax[0].xaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
    ax[1].xaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
    ax[2].xaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
    # make sure only three ticks per y-axis in n_y and D_m12 plots:
    ax[0].yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax[1].yaxis.set_major_locator(MaxNLocator(nbins=3))
    # make sure only three ticks per x-axis:
    ax[0].xaxis.set_major_locator(MaxNLocator(integer = True, nbins=3))
    ax[1].xaxis.set_major_locator(MaxNLocator(integer = True, nbins=4))
    ax[2].xaxis.set_major_locator(MaxNLocator(integer = True, nbins=3))


    # set axes limits
    ax[0].set_xlim(0)
    ax[1].set_xlim(0,4.5)
    ax[2].set_xlim(0,2.5)
    ax[0].set_ylim(0,2.5)
    ax[1].set_ylim(0,2.5)
    ax[2].set_ylim(0,)

    # ticks to the inside
    for axi in ax:
        axi.tick_params(pad=2)
        axi.tick_params(axis='both', which='both', direction = 'in')
        axi.tick_params(axis="both", which="both", width=0.5, length = 2)
    

    # legend
    handles, labels = ax[0].get_legend_handles_labels()
    
    legend = fig.legend(
        handles,
        labels,
        loc='lower center',
        ncol=3,
        bbox_to_anchor=(0.5, 0.02)
    )
    legend.get_frame().set_linewidth(0.5)

    return

def plot_diag_base_cases(ax, path_depl, thresh, type_ = 'eps'):
    load_steps = {}
    mat_displ = {}
    errors = {}
    for key0, i in zip(['2D-1', '2D-2', '2D-5'], range(3)): # case study
        load_steps[key0] = {}
        mat_displ[key0] = {}
        errors[key0] = {}
        for key1 in path_depl[key0].keys(): # rho_i
            load_steps[key0][key1], mat_displ[key0][key1] = plot_load_paths('05_Deploying\\data_out\\'+path_depl[key0][key1], case_study = key0, 
                                                                            until_load_level = None, save_path = None, type = type_)
            load_steps[key0][key1] = suitable_max_load(mat_displ, load_steps, thresh, key0, key1)
            if key0 == '2D-5':
                    load_steps[key0][key1] = load_steps[key0][key1][:-2]
                    mat_displ[key0][key1]['NN'] = mat_displ[key0][key1]['NN'][:-2]
                    mat_displ[key0][key1]['NLFEA'] = mat_displ[key0][key1]['NLFEA'][:-2]
        plot_individual_diagonal_base_case(ax[i], key0, load_steps, mat_displ, errors)
    return

def plot_individual_diagonal_base_case(ax, key0, load_steps, mat_displ, errors):
    cmap1 = plt.cm.viridis
    values = np.linspace(0.2,0.8,3)
    colors = [cmap1(v) for v in values]

    for key1,j in zip(load_steps[key0].keys(), range(3)):
        face_color = (colors[j][:3], 0.5)
        ax.scatter(mat_displ[key0][key1]['NLFEA'], mat_displ[key0][key1]['NN'], facecolors = face_color,
                    edgecolors = colors[j], marker = 'o', label = key1, s = 9, linewidth = 0.5)
        errors[key0][key1] = calculate_errors_diagonal(mat_displ, key0, key1)

    keys1 = list(mat_displ[key0].keys())
    minimum = 0 #np.minimum(np.min(mat_displ[key0][key1]['NLFEA']), np.min(mat_displ[key0][key1]['NN']))
    maximum = 15 #np.maximum(np.max(mat_displ[key0][keys1[0]]['NLFEA']), np.max(mat_displ[key0][keys1[0]]['NN']))
    diagonal = np.arange(minimum, maximum)
    ax.plot(diagonal, diagonal, color = 'grey', linestyle = '--', lw = 0.5)

    ax.text(
            0.05, 0.95,                 # x,y position in axes coordinates (0–1)
            f"$e_{{\\uprho_y = 0.75\\%}}$: {np.round(errors[key0][keys1[0]]['rmse'][0][0], 2)}\n"
            f"$e_{{\\uprho_y = 1.00\\%}}$: {np.round(errors[key0][keys1[1]]['rmse'][0][0], 2)}\n"
            f"$e_{{\\uprho_y = 1.50\\%}}$: {np.round(errors[key0][keys1[2]]['rmse'][0][0], 2)}",
            transform=ax.transAxes,     # makes coordinates relative to the axes
            verticalalignment="top",
            bbox=dict(
                boxstyle="round,pad=0.2",
                fc="white",
                ec="black",
                lw =0.5,
                alpha=0.8
            )
        )

    return

def final_touches_diag_base_case(fig, axs):
    axs[0].set_xlabel(r'$\hat{\upgamma}_{xy,NLFEA}$ [‰]')
    axs[1].set_xlabel(r'$\hat{\upvarepsilon}_{y,NLFEA}$ [‰]')
    axs[2].set_xlabel(r'-$\hat{\upvarepsilon}_{x,NLFEA}$ [‰]')

    axs[0].set_ylabel(r'$\hat{\upgamma}_{xy,NN}$ [‰]')
    axs[1].set_ylabel(r'$\hat{\upvarepsilon}_{y,NN}$ [‰]')
    axs[2].set_ylabel(r'-$\hat{\upvarepsilon}_{x,NN}$ [‰]')

    # ensure no numbers after decimal point
    axs[0].yaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
    axs[1].yaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
    axs[2].yaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
    axs[0].xaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
    axs[1].xaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
    axs[2].xaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
    
    axs[0].xaxis.set_major_locator(MaxNLocator(integer = True, nbins=3))
    axs[1].xaxis.set_major_locator(MaxNLocator(integer = True, nbins=4))
    axs[2].xaxis.set_major_locator(MaxNLocator(integer = True, nbins=3))


    # ticks to the inside
    for ax in axs:
        ax.tick_params(pad=2)
        ax.tick_params(axis='both', which='both', direction = 'in')
        ax.tick_params(axis="both", which="both", width=0.5, length = 2)


    # axes limits
    axs[0].set_xlim(0,11)
    axs[0].set_ylim(0,11)
    axs[1].set_xlim(0,4.5)
    axs[1].set_ylim(0,4.5)
    axs[2].set_xlim(0,2.5)
    axs[2].set_ylim(0,2.5)
    
    axs[2].yaxis.set_major_locator(MaxNLocator(nbins=3))
    
    return



def plot_case_2D8C(ax, path_depl_2D8C):
    raise UserWarning('Deprecated function, not in use.')
    load_steps = {}
    mat_displ = {}

    for key in path_depl_2D8C.keys():    # rho_y_i

        # 1 - Determine load steps
        full_path_2D8C = os.path.join(os.getcwd(), '05_Deploying\\data_out\\'+path_depl_2D8C[key])
        load_steps[key] = get_loadsteps(full_path_2D8C)
    
        # 2 - Determine strains
        eps_NN    = get_max_eps([0,1,2], load_steps[key], full_path_2D8C, tag = 'NN')
        eps_NLFEA = get_max_eps([0,1,2], load_steps[key], full_path_2D8C, tag = 'norm')
        mat_displ[key] = {
            'NN': eps_NN,
            'NLFEA': eps_NLFEA
        }

    # 3 - Plot
    cmap1 = plt.cm.viridis
    values = np.linspace(0.2,0.8,3)
    colors = [cmap1(v) for v in values]

    ax = np.array(ax).reshape((3,3))
    for i in range(3): 
        for j in range(3):
            for k, key in zip(range(3), load_steps.keys()):
                ax[i,j].plot(mat_displ[key]['NLFEA'][:,i], np.abs(load_steps[key]), color = colors[k], 
                        marker = 'o', markerfacecolor = 'none', ms = 2, lw = 0.5, linestyle = 'solid', label = 'NLFEA, ' +key)
                ax[i,j].plot(mat_displ[key]['NN'][:,i], np.abs(load_steps[key]), color = colors[k], 
                        marker = 'x', ms = 2, lw = 0.5, linestyle = '--', label = 'NN, ' +key)
    

    return

def final_touches_2D8C(fig, ax):
    raise UserWarning('Deprecated function, not in use.')
    ax = np.array(ax).reshape((3,3))
    for i in range(3): 
        ax[0,i].set_xlabel(r'$\varepsilon_{x}$ [‰]')
        ax[1,i].set_xlabel(r'$\varepsilon_{y}$ [‰]')
        ax[2,i].set_xlabel(r'$\gamma_{xy}$ [‰]')

        ax[i,0].set_ylabel('$n_{x}$ [kN/m]')
        ax[i,1].set_ylabel('$n_{y}$ [kN/m]')
        ax[i,2].set_ylabel('$n_{xy}$ [kN/m]')
        
    
    return

def plot_case_2D8C_slim(ax, path_depl_2D8C):
    '''
    only for 3 subplots (instead of all 9)
    '''

    load_steps = {}
    mat_displ = {}

    for key in path_depl_2D8C.keys():    # rho_y_i

        # 1 - Determine load steps
        full_path_2D8C = os.path.join(os.getcwd(), '05_Deploying\\data_out\\'+path_depl_2D8C[key])
        load_steps[key] = get_loadsteps(full_path_2D8C)
    
        # 2 - Determine strains
        eps_NN    = get_max_eps([0,1,2], load_steps[key], full_path_2D8C, tag = 'NN')
        eps_NLFEA = get_max_eps([0,1,2], load_steps[key], full_path_2D8C, tag = 'norm')
        mat_displ[key] = {
            'NN': eps_NN,
            'NLFEA': eps_NLFEA
        }

    # 3 - Plot
    cmap1 = plt.cm.viridis
    values = np.linspace(0.2,0.8,3)
    colors = [cmap1(v) for v in values]
    unit_s = 1e-3

    for i in range(3): 
        for k, key in zip(range(3), load_steps.keys()):
            ax[i].plot(mat_displ[key]['NLFEA'][:,i], np.abs(load_steps[key])*unit_s, color = colors[k], 
                    marker = 'o', markerfacecolor = 'none', markeredgewidth=0.5, ms = 3, lw = 0.5, linestyle = 'solid', label = 'NLFEA, ' +key)
            ax[i].plot(mat_displ[key]['NN'][:,i], np.abs(load_steps[key])*unit_s, color = colors[k], 
                    marker = 'x', ms = 3, markeredgewidth=0.5, lw = 0.5, linestyle = '--', label = 'NN, ' +key)

    

    return

def final_touches_2D8C_slim(fig, ax):
    ax[0].set_xlabel(r'$\hat{\upvarepsilon}_{x}$ [‰]')
    ax[1].set_xlabel(r'$\hat{\upvarepsilon}_{y}$ [‰]')
    ax[2].set_xlabel(r'$\hat{\upgamma}_{xy}$ [‰]')

    ax[0].set_ylabel('$n_{x} = n_{y} = n_{xy}$ [MN/m]')
    ax[1].set_ylabel('$n_{x} = n_{y} = n_{xy}$ [MN/m]')
    ax[2].set_ylabel('$n_{x} = n_{y} = n_{xy}$ [MN/m]')

    # ticks to the inside
    for axi in ax:
        axi.tick_params(pad=2)
        axi.tick_params(axis='both', which='both', direction = 'in')
        axi.tick_params(axis="both", which="both", width=0.5, length = 2)

    # ensure no numbers after decimal point
    ax[0].xaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
    ax[1].xaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
    ax[2].xaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
    
    ax[0].xaxis.set_major_locator(MaxNLocator(integer = True, nbins=4))
    ax[1].xaxis.set_major_locator(MaxNLocator(integer = True, nbins=3))
    ax[2].xaxis.set_major_locator(MaxNLocator(integer = True, nbins=3))

    # set axes limits
    ax[0].set_xlim(0,20)
    ax[1].set_xlim(0,2.1)
    ax[2].set_xlim(0,14)
    ax[0].set_ylim(0,)
    ax[1].set_ylim(0,)
    ax[2].set_ylim(0,)

    return

def plot_diag_2D8C(ax, path_depl_2D8C, thresh, type_ = 'eps'):
    load_steps = {}
    mat_displ = {}
    errors = {}
    for key0 in ['2D-8C']:
        load_steps[key0] = {}
        mat_displ[key0] = {}
        errors[key0] = {}
        for key1 in path_depl_2D8C.keys(): # rho_i
            load_steps[key0][key1], mat_displ[key0][key1] = plot_load_paths('05_Deploying\\data_out\\'+path_depl_2D8C[key1], case_study = '2D-8C', 
                                                                            until_load_level = None, save_path = None, type = type_)
        # load_steps[key1] = suitable_max_load(mat_displ, load_steps, thresh, key0, key1)
    
    cmap1 = plt.cm.viridis
    values = np.linspace(0.2,0.8,3)
    colors = [cmap1(v) for v in values]

    for i in range(3):
        key0 = '2D-8C'
        for k, key in zip(range(3), load_steps[key0].keys()):
            face_color = (colors[k][:3], 0.5)
            ax[i].scatter(mat_displ[key0][key]['NLFEA'][:,i], mat_displ[key0][key]['NN'][:,i], facecolors = face_color,
                           edgecolors = colors[k], marker = 'o', label = key, s = 9, linewidth = 0.5)
            errors[key0][key] = {}
            errors[key0][key][str(i)] = calculate_errors_diagonal(mat_displ, key0, key, i)

        keys1 = list(mat_displ[key0].keys())
        minimum = 0 #np.minimum(np.min(mat_displ[key0][key1]['NLFEA']), np.min(mat_displ[key0][key1]['NN']))
        maximum = 22 #np.maximum(np.max(mat_displ[key0][keys1[0]]['NLFEA']), np.max(mat_displ[key0][keys1[0]]['NN']))
        diagonal = np.arange(minimum, maximum)
        ax[i].plot(diagonal, diagonal, color = 'grey', linestyle = '--', lw = 0.5)

        ax[i].text(
                0.05, 0.95,                 # x,y position in axes coordinates (0–1)
                f"$e_{{\\uprho_y = 0.75\\%}}$: {np.round(errors[key0][keys1[0]][str(i)]['rmse'][0][0], 2)}\n"
                f"$e_{{\\uprho_y = 1.00\\%}}$: {np.round(errors[key0][keys1[1]][str(i)]['rmse'][0][0], 2)}\n"
                f"$e_{{\\uprho_y = 1.50\\%}}$: {np.round(errors[key0][keys1[2]][str(i)]['rmse'][0][0], 2)}",
                transform=ax[i].transAxes,     # makes coordinates relative to the axes
                verticalalignment="top",
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    fc="white",
                    ec="black",
                    lw =0.5,
                    alpha=0.8
                )
            )
        
    return

def final_touches_2D8C_diag_slim(fig, axs):
    axs[0].set_xlabel(r'$\hat{\upvarepsilon}_{x,NLFEA}$ [‰]')
    axs[1].set_xlabel(r'$\hat{\upvarepsilon}_{y,NLFEA}$ [‰]')
    axs[2].set_xlabel(r'$\hat{\upgamma}_{xy,NLFEA}$ [‰]')
    
    axs[0].set_ylabel(r'$\hat{\upvarepsilon}_{x,NN}$ [‰]')
    axs[1].set_ylabel(r'$\hat{\upvarepsilon}_{y,NN}$ [‰]')
    axs[2].set_ylabel(r'$\hat{\upgamma}_{xy,NN}$ [‰]')

    # ticks to the inside
    for ax in axs:
        ax.tick_params(pad=2)
        ax.tick_params(axis='both', which='both', direction = 'in')
        ax.tick_params(axis="both", which="both", width=0.5, length = 2)

    # axes limits
    axs[0].set_xlim(0,20)
    axs[0].set_ylim(0,20)
    axs[1].set_xlim(0,2.1)
    axs[1].set_ylim(0,2.1)
    axs[2].set_xlim(0,14)
    axs[2].set_ylim(0,14)

    # ensure no number after decimal (y-axis)
    axs[0].yaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
    axs[1].yaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
    axs[2].yaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
    axs[1].yaxis.set_major_locator(MaxNLocator(integer = True, nbins=3))
    axs[2].yaxis.set_major_locator(MaxNLocator(integer = True,nbins=3))
    
    # ensure no numbers after decimal point (x-axis)
    axs[0].xaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
    axs[1].xaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
    axs[2].xaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
    
    axs[0].xaxis.set_major_locator(MaxNLocator(integer = True, nbins=4))
    axs[1].xaxis.set_major_locator(MaxNLocator(integer = True, nbins=3))
    axs[2].xaxis.set_major_locator(MaxNLocator(integer = True, nbins=3))
    

    return


''' AUXILIARY FUNCTIONS SENSITIVITY'''

def add_subplots_gs_sensitivity(fig, gs):
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
    ax2 = None
    ax3 = fig.add_subplot(gs[0, 2], projection='3d')
    ax4 = None
    ax5 = fig.add_subplot(gs[0, 4])

    ax6 = fig.add_subplot(gs[1, 0], projection='3d')
    ax7 = None
    ax8 = fig.add_subplot(gs[1, 2], projection='3d')
    ax9 = None
    ax10 = fig.add_subplot(gs[1, 4])

    ax11 = fig.add_subplot(gs[2, 0], projection='3d')
    ax12 = None
    ax13 = fig.add_subplot(gs[2, 2], projection='3d')
    ax14 = None
    ax15 = fig.add_subplot(gs[2, 4])

    axs = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12, ax13, ax14, ax15]

    add_subplot_number_sen(axs)

    return fig, axs

def add_subplot_number_sen(axs):
    labels_text = ['(a)', None, '(b)', None, '(c)', 
                    '(d)', None, '(e)', None, '(f)',
                    '(g)', None, '(h)', None, '(i)']
    for ax, label in zip(axs, labels_text): 
        if ax is not None:
            if isinstance(ax, Axes3D):
                ax.text2D(0.05, 0.95, label, transform=ax.transAxes, 
                            fontweight='bold', va='top', ha='left')
            else: 
                ax.text(-0.1, 1.2, label, transform=ax.transAxes, 
                        fontweight='bold', va='top', ha='left')
    return


def final_touches_scatter_sensitivity(fig, axs, nrows = 3):
    # Axes labels
    for i in range(nrows):
        pad = -10
        axs[i,0].set_xlabel(r'$\hat{\upvarepsilon}_x$ [\textperthousand]', labelpad = pad)
        axs[i,0].set_ylabel(r'$\hat{\upvarepsilon}_y$ [\textperthousand]', labelpad = pad)
        axs[i,0].set_zlabel(r'$\hat{\upgamma}_{xy}$ [\textperthousand]', labelpad = pad)
        axs[i,1].set_xlabel(r'$\it{n}$$_x$ [MN/m]', labelpad = pad)
        axs[i,1].set_ylabel(r'$\it{n}$$_y$ [MN/m]', labelpad = pad)
        axs[i,1].set_zlabel(r'$\it{n}$$_{xy}$ [MN/m]', labelpad = pad)

        # Set maxima and minima
        axs[i,0].set_xlim(-5,50)
        axs[i,0].set_ylim(-5,50)
        axs[i,0].set_zlim(-100,100)
        axs[i,1].set_xlim(-11,2)
        axs[i,1].set_ylim(-11,2)
        axs[i,1].set_zlim(-4.5,4.5)

    # Legend
    fig.subplots_adjust(bottom=0.12, left=0, right=1, top=1)
    handles, labels = [], []
    handles, labels = axs[0,0].get_legend_handles_labels()
    
    legend = fig.legend(
        handles,
        labels,
        loc='lower center',
        ncol=2,
        bbox_to_anchor=(0.35, 0.05)
    )
    legend.get_frame().set_linewidth(0.5)
    for handle in legend.legend_handles:
        handle.set_sizes([20])
        handle.set_alpha(1) 

    for ax in axs.ravel():
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

        # move tick labels closer to the plot
        ax.tick_params(axis='x',pad=-4)
        ax.tick_params(axis='y',pad=-4)
        ax.tick_params(axis='z',pad=-3)
        ax.tick_params(axis='both', which='both', direction = 'in')
        ax.tick_params(axis="both", which="both", width=0.5, length = 0.5)
        # make sure no numbers after decimal point
        ax.xaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
        ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
        ax.zaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))          
        
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
            [6,7], [7,4], # [4,5], [5,6],  # top
            [0,4], [2,6], [3,7]   # [1,5], # verticals
        ]

        for e in edges:
            ax.plot(*corners[e].T, color='black', linewidth=0.5)
    
    fig.patch.set_alpha(0)
    
        
    return fig, axs

def final_touches_sensitivity_depl(fig, axs, nrows = 3):
    for i in range(nrows):
        axs[i,0].set_xlabel(r'$\hat{\upvarepsilon}$ [\textperthousand]', labelpad = 1)
        axs[i,0].set_ylabel(r'$n_x$ [MN/m]', labelpad = 1)
    

    handles, labels = [], []
    handles, labels = axs[0,0].get_legend_handles_labels()
    
    legend = fig.legend(
        handles,
        labels,
        loc='lower center',
        ncol=1,
        bbox_to_anchor=(0.85, 0.02)
    )
    legend.get_frame().set_linewidth(0.5)

    for ax in axs.ravel(): 
        # Layouting gridlines / axes 
        ax.grid(True, linewidth=0.5)
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
        ax.tick_params(pad=2)
        ax.tick_params(axis='both', which='both', direction = 'in')
        ax.tick_params(axis="both", which="both", width=0.5, length = 0.5)
        #shrink plot size slightly
        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0+ pos.height * 0.15, pos.width*0.9, pos.height * 0.6])
        ax.set_xlim(0,)
        ax.set_ylim(0,)
        ax.xaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
        ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.1f}'))

    return

def plot_individual_base_case_sensitivity(ax, key0, load_steps, mat_displ, thresh, type_ = 'eps', color = None):
    
    colors = color
    bright_factor = 0.5
    colors_1 = []
    for color in colors:
        rgb = mcolors.to_rgba(color)[:3]
        lighter = tuple(c + (1 - c) * bright_factor for c in rgb)
        colors_1.append(lighter)

    unit_s = 1e-3

    if type_ != 'eps':
        raise UserWarning('This plot is not implemented for plotting displacements u, please use type = "eps".')

    for j, key1 in enumerate(load_steps[key0].keys()):
        ax.plot(mat_displ[key0][key1]['NLFEA'][:-1,0], np.abs(load_steps[key0][key1][:-1])*unit_s, color = colors[j], 
                marker = 'o', markeredgewidth=0.5, markerfacecolor = 'none', ms = 3, lw = 0.5, label = r"NLFEA $f(\hat{\upvarepsilon}_x)$") # +key1)
        ax.plot(mat_displ[key0][key1]['NN'][:-1,0], np.abs(load_steps[key0][key1][:-1])*unit_s, color = colors[j], 
                marker = 'x', ms = 3, markeredgewidth=0.5, lw = 0.5, linestyle = '--', label = r"NN $f(\hat{\upvarepsilon}_x)$") # +key1)
        
        ax.plot(mat_displ[key0][key1]['NLFEA'][:-1,1], np.abs(load_steps[key0][key1][:-1])*unit_s, color = colors_1[j], 
                marker = 'o', markeredgewidth=0.5, markerfacecolor = 'none', ms = 3, lw = 0.5, label = r"NLFEA $f(\hat{\upgamma}_{xy})$") # +key1)
        ax.plot(mat_displ[key0][key1]['NN'][:-1,1], np.abs(load_steps[key0][key1][:-1])*unit_s, color = colors_1[j], 
                marker = 'x', ms = 3, markeredgewidth=0.5, lw = 0.5, linestyle = '--', label = r"NN $f(\hat{\upgamma}_{xy})$") # +key1)

    return

def add_arrows(axs):
    axs = axs.ravel()
    starts = [[(2.5, 0.5), (1.5, 0.4)],
              [(2.5, 0.5), (1.5, 0.4)],
              [(2.5, 0.5), (1.5, 0.4)]]
    ends = [[(1.3, 0.94), (0, 0.94)],
            [(1.3, 0.94), (0, 0.94)],
            [(1.3, 0.94), (0, 0.94)]]
    labels = [[r"$f(\hat{\upvarepsilon}_x)$", r"$f(\hat{\upgamma}_{xy})$"],
              [r"$f(\hat{\upvarepsilon}_x)$", r"$f(\hat{\upgamma}_{xy})$"],
              [r"$f(\hat{\upvarepsilon}_x)$", r"$f(\hat{\upgamma}_{xy})$"]]
    curvatures = [[-0.35, -0.2],
                  [-0.35, -0.2],
                  [-0.35, -0.2]]
    t_n = [[0.99, 0.99], 
           [0.99, 0.9], 
           [0.99, 0.9]]
    
    for i in range(3):
        for start, end, label, rad, t_i in zip(starts[i], ends[i], labels[i], curvatures[i], t_n[i]):
            rad = rad
            arrow1 = FancyArrowPatch(
                    start, end,
                connectionstyle=f"arc3,rad={rad}",
                arrowstyle="Simple,head_length=4,head_width=2,tail_width=0.25",
                linewidth=0,
                color="gray",
                zorder = 10
            )
            axs[i].add_patch(arrow1)

            mid = place_arrowhead(start, end, rad, t_i, head_length_pts=4)

            arrow2 = FancyArrowPatch(
                start, mid,
                connectionstyle=f"arc3,rad={rad}",
                arrowstyle="Simple,head_length=4,head_width=2,tail_width=0",
                linewidth=0,
                color="gray",
                zorder = 10
            )

            axs[i].add_patch(arrow2)
            axs[i].text(*start, label, ha = 'left', color = 'gray')

def place_arrowhead(start, end, rad, t, head_length_pts):
    # Compute Bezier points
    start = np.array(start)
    end = np.array(end)
    con = ConnectionStyle.Arc3(rad=rad)
    path = con.connect(start, end)
    P0, P1, P2 = path.vertices
    
    # Adjust t backward to account for arrowhead length
    if t<0.95:
        t_adjusted = t - 0.08  # Tune this value (try 0.01 to 0.05)
    else: 
        t_adjusted = t-0.02
    
    mid = (1-t_adjusted)**2 * P0 + 2*(1-t_adjusted)*t * P1 + t_adjusted**2 * P2
    return mid



''' ..........................  MAIN FUNCTIONS .......................... '''

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
    mpl.rcParams["axes.labelpad"] = -0.5
    fig = plt.figure(figsize = (17/2.54, 8/2.54))
    vars = ['eps', 'sig']
    
    # 3 - Plot data in figure
    plotevery = {
        labels[0]: 500,
        labels[1]: 200
    }
    axs = [fig.add_subplot(1, len(vars), i + 1, projection='3d')
       for i in range(len(vars))]
    axs = plotting_scatter(axs, mat_all, labels, vars, plotevery)
    
    # 4 - Figure final touches
    fig, axs = final_touches(fig, axs)
    

    # 5 - Save figure
    plotname = 'scatter_data_simple'
    save_figure(fig, save_path, plotname, bbox = None)

    return


def plot_training_results(NN, save_path= None, include_predict = False):
    '''
    Plots results of the training of the NN
    
    :param mat_multi_NN: data for the plot at the bottom (sensitivity analysis for data vs physics loss)
    :param NN: list containing ['xxx', '_yyy'], where xxx is the version number and yyy the epoch number of the corresponding NN.
    :param save_path: path where plot should be saved
    '''

    # 1 - create figure 
    setup_figure_format()
    fig = plt.figure(figsize = (17/2.54,12/2.54))
    gs = GridSpec(3, 4, figure = fig, width_ratios=[1,1,1,1], height_ratios=[1,1,1], wspace = 0.35, hspace = 0.35)
    fig, axs = add_subplots_gs(fig, gs)
    fig.subplots_adjust(bottom=0.1, left=0.08, right=0.98, top=0.95)

    # 2 - plot diagonal plots (test plots NN)
    axs = plot_diagonal_scatter(fig, axs, NN, include_predict)
    axs = final_touches_diag_scatter(axs)


    # 3 - plot stress path
    plot_stress_paths(axs[4:8],NN)
    final_touches_stress_path(axs[4:8])
    add_legend(axs)

    # 4 - plot physics loss for different data amounts
    mat_data = get_data_physics_influence()
    plot_physics_influence(axs[8], mat_data)



    # 5 - save figure
    save_figure(fig, save_path, 'training_results', bbox = None)

    return
    

def plot_deploying_results(path_depl, save_path, thresh):

    # 1 - create figure
    setup_figure_format()
    fig = plt.figure(figsize = (17/2.54,24/2.54))
    gs = GridSpec(4, 3, figure = fig, height_ratios=[1,1,1,1], wspace = 0.2, hspace = 0.2)
    fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.1)
    fig, axs = add_subplots_gs_depl(fig, gs)

    # 2a - plot base cases load-def.
    plot_base_cases(axs[0:3], path_depl, thresh)
    final_touches_base_cases(fig, axs[0:3])

    # 2b - plot base cases diagonal
    plot_diag_base_cases(axs[3:6], path_depl, thresh)
    final_touches_diag_base_case(fig, axs[3:6])

    # 3a - plot 2D-8C case load-def
    plot_case_2D8C_slim(axs[6:9], path_depl['2D-8C'])
    final_touches_2D8C_slim(fig, axs[6:9])

    # 3b - plot 2D-8C case diagonal
    plot_diag_2D8C(axs[9:12], path_depl['2D-8C'], thresh)
    final_touches_2D8C_diag_slim(fig, axs[9:12])

    for ax in axs:
        ax.xaxis.labelpad = -1
        ax.yaxis.labelpad = -0.1

    # 4 - save figure
    save_figure(fig, save_path, 'deploying_results', bbox = None)


    return


def plot_sensitivity(path_data, path_deployment, save_path, thresh):

    # 1 - create figure
    setup_figure_format()
    fig = plt.figure(figsize = (17/2.54, 20/2.54))
    gs = GridSpec(3,5, figure = fig, height_ratios = [1,1,1], width_ratios=[1,0.01,1,0.15,1], wspace = 0.1, hspace=0)
    fig, axs = add_subplots_gs_sensitivity(fig, gs)

    # 2a - read scatter data
    mat_all = {}
    for key0 in path_data.keys():
        mat_all[key0] = {}
        for key1 in path_data[key0].keys():
            mat_all[key0][key1] = read_sampled_data(path_data[key0][key1])

    # 2b - plot scatter data
    axs = np.array((axs)).reshape((3,5))
    for key0 in path_data.keys():
        plotevery = {}
        for key1 in path_data[key0].keys():
            if key1 == 'Uniform':
                plotevery[key1] = 500
            elif key1 == 'Log.':
                plotevery[key1] = 200
        plotting_scatter(np.hstack((axs[int(key0),0:1], axs[int(key0), 2:3])), 
                         mat_all[key0], labels = path_data[key0].keys(), vars = ['eps', 'sig'], p = plotevery)
    final_touches_scatter_sensitivity(fig, np.concatenate((axs[0:3,0:1], axs[0:3,2:3]), axis =1), nrows =3)


    # 3 - plot deployments
    # mpl.rcParams["axes.labelpad"] = 0
    color = [['black'], [get_colorscale(2)[1]], [get_colorscale(2)[0]]]
    plot_base_cases(axs[0:3,4], path_deployment, thresh, type_ = 'eps', color=color)
    final_touches_sensitivity_depl(fig, axs[0:3,4:5])
    add_arrows(axs[0:3,4:5])



    # 4 - save figure
    save_figure(fig, save_path, 'sensitivity', bbox = None)



    return



'''.......................... SAVING ERRORS ..........................'''

def save_errors_to_excel(path_depl_all):
    mat_displ = {}
    mat_eps = {}
    errors_displ = {}
    errors_eps = {}

    for key0 in path_depl_all.keys():
        mat_displ[key0] = {}
        mat_eps[key0] = {}
        errors_displ[key0] = {}
        errors_eps[key0] = {}
        for key1 in path_depl_all[key0].keys():
            path_here = '05_Deploying\\data_out\\'+path_depl_all[key0][key1][0]
            index = path_depl_all[key0][key1][1]

            load_steps = get_loadsteps(path_here)[:index]
            displ_NN    = get_max_displ([0,1], load_steps, path_here, tag = 'NN')
            displ_NLFEA = get_max_displ([0,1], load_steps, path_here, tag = 'norm')
            mat_displ[key0][key1] = {
                    'NN': displ_NN,
                    'NLFEA': displ_NLFEA
            }
            eps_NN    = get_max_eps([0,1,2], load_steps, path_here, tag = 'NN')
            eps_NLFEA = get_max_eps([0,1,2], load_steps, path_here, tag = 'norm')
            mat_eps[key0][key1] = {
                    'NN': eps_NN,
                    'NLFEA': eps_NLFEA
            }
        print('Collected all eps and u data.')
        for key1 in path_depl_all[key0].keys():
            errors_displ[key0][key1] = [[],[]]
            errors_eps[key0][key1] = [[],[],[]]
            for i in range(2):
                    e = calculate_errors_diagonal(mat_displ, key0, key1, i)
                    errors_displ[key0][key1][i] = e['rmse'][0][i]
            for i in range(3):
                    e = calculate_errors_diagonal(mat_eps, key0, key1, i)
                    errors_eps[key0][key1][i] = e['rmse'][0][i]
        print('Calculated all RMSE')
                    
    export_to_excel(errors_displ, 'u')
    export_to_excel(errors_eps, 'eps')
    
    return


def export_to_excel(data, id):  
        if id == 'eps':
            sub_headers = ["εx", "εy", "γxy"]
        elif id == 'u':
            sub_headers = ["ux", "uy"]
        rho_labels = ["ρ_y = 1%", "ρ_y = 0.75%", "ρ_y = 1.5%"]

        # Build MultiIndex columns
        columns = pd.MultiIndex.from_product(
        [rho_labels, sub_headers]
        )

        # Build rows
        rows = []
        for _, rho_dict in data.items():
                row_vals = [val for vals in rho_dict.values() for val in vals]
                rows.append(row_vals) 

        df = pd.DataFrame(rows, index=data.keys(), columns=columns)

        # Export to Excel — MultiIndex columns become merged header rows automatically
        df.to_excel("05_Deploying\\error_calc_"+id+".xlsx", merge_cells=True)
        print(f'Saved excel for {id}')
        return



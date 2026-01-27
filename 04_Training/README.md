
## Overview of files:

Most relevant files: 
- `train_vb.py`: Execute this from a terminal to train a model
    - dependencies: `data_work.py`, `FFNN_class_light.py`, `train_utils.py`
- `test_vb.py`: Execute this from a terminal to test a model
    - dependencies: `data_work.py`, `FFNN_class_light.py`, `call_light.py`, `test_utils.py`
- `sampler_analytical.py`: Analytical ("local") sampling based on Simulation files provided by an
    - dependencies: `sampler_utils.py` (which itself depends on `Stresses_mixreinf.py`, `defcplx.py`)

Less relevant files: 
-  `test_vb_paper.py`: Execute this from a terminal to test a model and generate the image required for the paper
    - dependencies: `data_work.py`, `FFNN_class_light.py`, `call_light.py`, `test_utils.py`
- `Plots_Sampling_Comparison.py`: Execute this from a terminal to create scatter / histogram comparison plots for paper
    - dependencies: `data_work.py`

Least relevant files: 
- `adding_sampled_data`: combines two data sets
- `TrainingXGBoost.ipynb`: not relevant anymore (was used as a baseline comparison)
- all python files that are stored in subfolders of this one are not directly in use.

## Overview of data stored:

Relevant: 

- Folder `data`
  - contains input data
  - locally sampled: "[]..._fake"
  - globally sampled: "[]..._casexx"
  - currently used dataset, from locally sampled data: `04_Training\data\data_20241212_1605_fake`
- Folder `new_data`:
  - `_simple_logs`: subfolder with the relevant trained models (subfolders, with version numbers)
  - otherwise: contains a lot of folders that are not relevant anymore
- Folder `plots`: 
  - dump-folder for plots
  - also where image for paper is saved (sampling)

Less relevant: 
- Folder `logs` and `lightning logs`:
  - logs created with pytorch lightning (are no longer required / not important anymore)
- Folder `mike`:
  - contains code from mike's guidance (old)
- Folder `notes`:
  - contains notes of the old experiments (lightning). No longer relevant
- Folder `wandb`:
  - stores information about all wandb runs. Never used this.


## Explanation of individual input files (only relevant ones)

#### `train_vb.py` : Run this file to execute a training. Adjust the following Variables if required
0. Read in data
    - `NAME`: data set name (raw data). The most recent one for lin.el. material law is: `'04_Training\data\data_20241212_1605_fake'`
    - `SAVE_FOLDER`: set to true if you would like to save this model to separate version folder (including inp files). This flag only works for single runs, not for sweeps.
    - `SWEEP`: whether to run the training as a sweep or as a single run 
    - *nbins*: amount of bins for visualisation of histograms (not very relevant)
1.  Train-Eval-Test Split
    - no external input required.
    - can turn on / off plotting of histograms by commenting in / out the plotting functions
2. Normalisation
    - `type = ['std']` or `type = ['range']`: careful, this also needs to be adjusted at every other location, where a data transformation takes place. This is not always straight-forward visible and not automatically adjusted everywhere (e.g. in test_vb.py)
3. Define input parameters
    - `sobolev`: set this to true for trainings including Sobolev loss.
    - `Split_Net_all`: set this to true for Split_Net_all (is this code working?)
    - `inp`: main input file for changing variables / network architectures for single runs (`SWEEP = False`)
      - *Network architecture / characteristics*:
        - input_size, out_size, hidden_layers: network architecture
        - batch_size = data set size (fixed, cannot be adjusted)
        - num_epochs, activation, learning rate, dropout rate, BatchNorm = freely choosable; kfold: outdated (raises Error)
        - learning rate scheduler: standard or plateau (=StepLR or ReduceLROnPlateau)
        - num_samples, fourier mapping: for Fourrier Mapping
        - loss_type, splitloss, weights: Loss Function
      - *Training type*: 
        - simple_m: **ALWAYS LEAVE AT TRUE** (ensures that code is carried out with simple pytorch train function, not with lightning)
      - *Network type*:
        - Sobolev, w_s: For Network with sobolev loss
        - Pretrain, hidden_layers_new, BatchNorm_new, activation_new: For adding a pretrained network in architecture
        - DeepONet: if set to true, use input_size = 8
        - MoE: this function still needs checking for relevance / for whether it is fully working as anticipated.
    - `constant_inp`: values that are constant in the sweep
4. Define main train function
    - No inputs need to be made here.
5. Train normally or define sweep parameters and run sweep
    - `SWEEP = False`: no changes required here
    - `SWEEP = True`: 
      - change sweep_config to the desired sweeping parameters
      - note: you cannot sweep over DeepONet = True, False (as the True and False require different input_sizes)
      - change count in agent for amount of sweeps.


## Explanation of individual dependency files (only relevant ones)


#### `FFNN_class_light.py` contains information related to FFNN model:

- Definition of loss functions and fourrier mapping:
  - class `FourierMapping` for Fourier Mapping
  - classes `RMSELoss`, `MSELoss`, `wMSELoss` contain different loss functions that could be selected for training
  
- Definition of Architectures:
  - class `FFNN` for definition of feed-forward NN, takes inp as argument --> flexibility in w, d, act, ...
    - function `forward` defines forward pass through net
  - class `BranchNet`, `TrunkNet` and `DeepONet_vb`: for defining the DeepONet architecture
  - class `FFNN_pretrain` for defining a FFNN with a pretrained FFNN model
  - classes `Expert`, `Gating` and `MoE` for defining a Mixture of Experts model

- Definition of Sobolev Losses
  - `Custom Losses`

- Definition of Lightning Modules
  - `LitFFNN` (not in use)
  - `LitFFNN_doub` (not in use)


#### `data_work.py` contains all functions related to data transformation / plotting:
- Loading Data: 
   class `MyDataset` for torch DataSet definition
  - function `read_data` for reading data points from given path and converting to numpy format
  - function `save_data` for saving data points that are already split into train, eval and test
  - function `data_to_torch` for transforming data from numpy to torch dataloaders and saving
- Histograms, Statistics:
  - function `histogram` that enables plotting histograms based on numpy DataSets
  - function `histogram_torch` enables plotting histograms of torch Datasets
  - function `statistics_pd` calculates relevant statistical parameters that are constant for entire script (pandas)
  - function `statistics` calculates relevant statistical parameters that are constant for entire script
  - function `transform_data` for the moment only standard normalisation
    could be augmented by max-min or other normalisation types
- Initialising wandb, unit transformations
  - function `init_wandb` for smoother initialisation of wandb in main training script
  - function `transf_units` for transforming units from simulation to training units and back
- Postprocessing:
  - function `calculate_errors` for calculating the errors required in the test plots
  - function `multiple_diagonal_plots_wrapper` for MoE execution of multiple diagonal plots (for all sub-nets)
  - function `multiple_diagonal_plots` for plotting the test data in 45°-plots, including R^2 values
  - function `multiple_diagonal_plots_Dnz` for plotting the test data in 45°-plots for the non-zero entries of D
  - function `multiple_diagonal_plots_paper` for the plot required in the paper
- Plot saving / copying:
  - function `get_latest_version_folder`
  - function `copy_files_with_incremented_version`
  - function `copy_files_to_plots_folder`
- Plotting raw sampled data:
  - function `plots_mike`
  - function `plot_nathalie`
  - function `plot_paper_comp`



#### `call_light.py` contains all functions relevant for querying the trained model
- Relevant for testing:
  - function `load_data` loads all created data during training; used for testing
  - function `predict_D` predicts D based on loaded data and model, in batch-wise format, used in test_vb.py


- Not in use / less relevant:
  - function `predict_sig_D` predicts both sig and D based on loaded data and model but every row at a time
      --> not in use for training / testing, keep on radar for deploying
  - function `predict_sig_XGB`: predicts sigma with XGB trained model (old, not in use)
  - function `D_an` calculates analytical solution (lin.el.); only referenced in inp-out plot (= not in use)
  - function `inp_out_plt` plots eps-sig for queried values (analytical, simulated and predicted) for different t, 
    based on deployment predictions (predict_sig_D)


Note: everything is connected to `wandb`, where loss plots are automatically logged


#### `train_utils.py` contains functions aiding training in train_vb.py
- Model creation
  - function `model_instance`: creates a model instance
  - function `model_print`: prints model to wandb (for later checking)
  - function `data_split_MoE`: splits data for different Experts of MoE
- Training with lightning: 
  - funtion `trainer_instance`: creates a lightning trainer instance, not in use
- Training with pytorch: 
  - function `simple_train`: defines the relevant parameters that need to be passed to training loop, depending on whether kfold is required or not.
  - function `simple_train_aux`: contains the training loop in pytorch

#### `test_utils.py` contains functions aiding testing in test_vb.py
- function `test_model_instance`: create instance of model used for testing
- function `make_prediction`: creates sigma predictions for testing. Output format such that it can be used by sigma-diagonal plots
- function `data_split_MoE_test`: splits data for MoE-separate testing of individual experts


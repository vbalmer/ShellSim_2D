## Overview Simulation Process
(c) Simulation coded by Andreas Näsbom and adapted for this case study by Vera Balmer (ETH Zurich, 27.05.2024)

This document provides an overview of the files in this folder "Simulator" including a user guide on how to get started with the simulation and data generation process. In general terms, the set of files allows for generating a data set for a simple plate of dimensions <code>b x h</code> in a parametrised manner for a layered shell element with different material properties.

The most relevant files are the following:

<ul>
  <li><code>run.ipynb</code></li>
  <li><code>DataSet.ipynb</code></li>
</ul>
which access:
<ul>
  <li><code>Mesh_gmsh_vb.py</code>, adjusted: created <code>input_definition</code> function to replace local variables, parametrised<br>

  the original version is also provided in this folder (<code>Mesh_gmsh.py</code>) </li>
  <li><code>Main_vb.py</code>, adjusted: created <code>main</code> function, inserted input_definition function to replace local variables, parametrised<br>
  the original version is also provided in this folder (<code>Main.py</code>)</li>
  <li><code>fem_vb.py</code>, adjusted: created a class from functions, got rid of locally defined variables<br>
  the (almost) original version is also provided in this folder (<code>fem.py</code>)</li>
  <li><code>PostProcess_vb.py</code>, adjusted: created a class from functions, got rid of locally defined variables<br> 
  the original version is also provided in this folder (<code>PostProcess.py</code>)</li>
</ul> 

As well as the subfolder <code>results</code>, which contains all results of the <code>i</code>'th simulation as well as the important output file <code>mat_res.pkl</code>, which compiles the relevant parameters for ALL simulations.

IMPORTANT NOTE: For the <code>CMM.py and el_tensors.py</code>, there are still some variables that are imported in a wrong manner (i.e. locally from mesh_gmsh instead of as part of a function), which needs adjustment if the CMM material model is to be used in the future (but for now just use the lin.el. version). Also, the el_tensors.py is not required in any of the other files --> might be deleted (ev. ask Andreas).

### 1 Getting Started

To get started, use the script <code>run.ipynb</code> which will execute the entire simulation process, following these basic steps: 

<ol>
  <li><em>[don't change:]</em> Definition of <code>HiddenPrints</code> Class</li>
  <li>Importing relevant libraries and defining path to sampled inputs, here these inputs are saved in a superfolder: <code>06_LinElData>01_SamplingFeatures>LHS.ipynb</code></li>
  <li>Defining input parameters for data set generation<br>
  If new features / input parameters are desired, they can be added to <code>mat_tot_dict</code> here directly or as additional import from the LHS and will also need to be linked in the </code>Mesh_gmsh_vb.py</code> or <code>Main_vb.py</code> file to the corresponding parameter </li>
  <li><em>[don't change:]</em> Running simulation in batches</li>
</ol> 

Use the script <code>DataSet.ipynb</code> to create statistical plots for the simulated output data:

<ol>
  <li><em>[don't change:]</em> Read the data pickle file <code>mat_res.pkl</code></li>
  <li>Importing relevant libraries and functions</li>
  <li> <strong>[tbd]</strong> Execute the function <code>single_analysis</code> for analysing all variables for one single example in the data set</li>
  --> works for the example of $n_x$ but not more yet
  <li> Execute the function <code>statistics</code> for analysing one variable for all examples in the data set</li>
</ol> 


### 2 Rough code descriptions

#### 2.1 Main_vb.py

This file contains the main functions for running the simulation:
<ul>
  <li><code>un_thn</code>: auxiliary function</li>
  <li><code>plot_convergence</code>: allows to plot convergence for every single simulation</li>
  <li><code>main_solver</code>: main solver function which is called from run file, contains the following steps:</li>
    <ol>
      <li>Run <code>input_definition</code> function (see <code>Mesh_gmsh_vb.py</code>) </li>
      <li>Initialise Iteration and Convergence Plots</li>
      <li>Nonlinear Solution: Iteration with secant or tangent stiffness</li>
      <li>Postprocessing (saving data) and Time Management</li>
    </ol>
</ul> 


#### 2.2 Mesh_gmsh_vb.py

This file contains the definition of material as well as geometrical input parameters:

<ul>
  <li><code>gauss_points</code>: auxiliary function for gauss integration points</li>
  <li><code>jacobi</code>: auxiliary function for the calculation of jacobian</li>
  <li><code>vcos</code>, <code>find_copl</code>, <code>dotproduct</code>, <code>length</code>, <code>angle</code>: auxiliary functions </li>
  <li>Some previous example calculations of Andreas</li>
  <li><code>input_definition</code>: main input definition function, including following steps:</li>
    <ol>
      <li>Geometry and Mesh: Definition of mesh size with parameter <code>ms</code> (here <code>ms = L/2</code>) </li>
      <li>Loads & BCs</li>
      <li>Material & further parameters</li>
      <li>Postprocess Mesh</li>
    </ol>
</ul> 


#### 2.3 fem_vb.py

This file contains some functions that are called from the main_vb.py file: <br>
<em> Note: The fem.py file contained in this folder is NOT the original fem.py by Andreas. The latter can be found in the subfolder 02_Simulator/_ogVersion_Andreas</em>

<ul>
  <li>class <code>gettime</code>: time measurement</li>
  <li>class <code>ki</code>: retains old stiffness matrix (is this used anywhere??)</li>
  <li>class <code>fem_func</code>: executes main fem functions listed below</li>
    <ul>
      <li>function <code>gauss_points</code>:returns Gauss weights and points</li>
      <li>function <code>get_et</code>: returns stiffness matrix <code>ET</code> based on</li>
        <ul>
          <li><code>cm_klij</code>: Material type (1 = lin.el., 3 = CMM)</li>
          <li><code>k</code>: Element number</li>
          <li><code>l</code>: Layer number</li>
          <li><code>[ex_klij, ey_klij, gxy_klij]</code>: strain state</li>
        </ul>
      <li>function <code>jacobi</code>: returns jacobian, inverse of jacobian and determinant of jacobian</li>
      <li>function <code>b_kij</code>: returns displacement matrices <code>B_m, B_s, B_b</code></li>
      <li>function <code>find_b</code>: returns displacement matrix <code>B</code></li>
      <li>function <code>rotLG</code>: returns rotation matrices <code>Tk, Tkr</code></li>
      <li>function <code>dh_kij</code>: returns d-matrices</li>
      <li>function <code>k_k</code>: returns stiffness matrix at element <code>k</code></li>
      <li>function <code>k_glob</code>: returns global stiffness matrix</li>
      <li>function <code>c_dof</code>: returns dofs for static condensation</li>
      <li>function <code>v_stat_con</code>: returns statically condensed force vector</li>
      <li>function <code>m_stat_con</code>: returns statically condensed stiffness matrix</li>
      <li>function <code>m_assemble</code>: returns global stiffness matrix w.r.t. specific element</li>
      <li>function <code>v_assemble</code>: returns global force vector w.r.t. specific element</li>
      <li>function <code>f_assemble</code>: vector of applied outer forces</li>
      <li>function <code>f0_assemble</code>: returns vector of external forces caused by internal stresses</li>
      <li>function <code>find_node</code>: returns node numbers for given coordinates</li>
      <li>function <code>find_nodes</code>: returns nodes connected to given element</li>
      <li>function <code>find_node_range</code>: ??</li>
      <li>function <code>find_el_range</code>: ??</li>
      <li>function <code>find_el</code>: return elements connected to given node</li>
      <li>function <code>find_dofs_n</code>: return dofs for given node</li>
      <li>function <code>find_dofs_k</code>: return dofs for given element</li>
      <li>function <code>find_v_el</code>: calculate values of a vector at all nodes of regarded element</li>
      <li>function <code>find_fi</code>: calculate vector of inner forces</li>
      <li>function <code>find_eh_kij</code>: returns strains for entire element and gauss point</li>
      <li>function <code>find_eh</code>: returns generalised strains for entire element</li>
      <li>function <code>find_sh</code>: returns generalised stresses</li>
      <li>function <code>find_e_klij</code>: returns strains at gauss points</li>
      <li>function <code>find_e</code>:?? </li>
      <li>function <code>find_e0</code>: ??</li>
      <li>function <code>find_sh0_kij</code>: </li>
      <li>function <code>find_s</code>: find stresses </li>
      <li>function <code>find_s0</code>: </li>
      <li>function <code>find_s0_klij</code>: </li>
      <li>function <code>find_ss</code>: </li>
      <li>function <code>find_sc</code>: </li>
      <li>function <code>solve_sys</code>: solves system </li>
      <li>function <code>solve_0</code>: solves system for initial iteration (linear elasticity)</li>
    </ul>
</ul>




### 3 Detailed code descriptions [for the nerds :D]

#### 3.1 <code>main_solver</code> function

Inputs: <br>

<ul>
  <li><code>mat</code> [dict]: Containing all potential input features, specifically:

  | Parameter             | Type    | Description
  | --------------------  | ------- | ---------------------------------
  | <code>L</code>        | float64 | length of sample plate
  | <code>B</code>        | float64 | width
  | <code>E</code>        | float64 | Young's Modulus 
  | <code>ms</code>       | float64 | Mesh size
  </li>
  <li><code>conv_plt</code> [bool]: To turn on or off plotting of convergence plots</li>
  <li><code>simple</code> [bool]: To turn on or off the simple mode for debugging </li>
</ul>

                 

Outputs:<br>
<ul>
  <li><code>mat_res</code> [dict]: Containing all relevant output features for further use, specifically:

  | Parameter             | Type    | Description
  | --------------------  | ------- | ---------------------------------
  | <code>L</code>        | float64 | length of sample plate
  | <code>B</code>        | float64 | width 
  | <code>COORD</code>    | dict    | coordinates at center of  each element
  | <code>sig_g</code>    | np-arr  | generalised stressses at midpoint of each element
  |                       |         | $[N_x,N_y,N_{xy},M_x,M_y,M_{xy},Q_x,Q_y]$
  | <code>eps_g</code>    | np-arr  | generalised strains at midpoint of each element
  |                       |         | $[\varepsilon_x,\varepsilon_y,\varepsilon_{xy},\chi_x,\chi_y,\chi_{xy},\gamma_x,\gamma_y]$</li>
  <!-- | <code>GEOMA</code>    | dict    | geometrical information (?) -->
  <!-- | <code>NODESG</code>   | dict    | nodal points (?) -->
</ul>

Outputs in <code>POST</code> variable:
<ul>
  <li><code>xn, yn, zn</code>: Node coordinates</li>
  <li><code>ux, uy, uz, thx, thy, thz</code>: Displacements, Rotations</li>
  <li><code>relunx, reluny, relunz, relthnx, relthny, relthnz</code></li>
  <li><code>RNx, RNy, RNxy, RMx, RMy, RMxy</code></li>
  <li><code>Nx, Ny, Nxy, Mx, My, Mxy, Qx, Qy</code>: Generalised Stresses</li>
  <li><code>exinfa, exsupa, eyinfa, eysupa, gxyinfa, gxysupa</code></li>
  <li><code>e3infa, e1infa, thinfa</code>: Strains in lowest layer (?)</li>
  <li><code>ssxinfa, ssyinfa, spxinfa, spxinfa, spyinfa</code></li>
  <li><code>e3supa, e1supa, thsupa</code>: Strains in highest layer (?) </li>
  <li><code>ssxsupa, ssysupa, spxsupa, spxsupa, spysupa</code></li>
  <li><code> relexinfa, relexsupa, relexsupa, releyinfa, releysupa, relgxyinfa, relgxysupa</code></li>
  <li>Missing, to be added by Vera: Generalised strains</li>
</ul>

Content of function:

<ol>
    <li>Run <code>input_definition</code> function (see <code>Mesh_gmsh_vb.py</code>) </li>
        contains definition of parameter <code>it_type</code> (use secant or tangent stiffness for nonlinear simulation)
    <li>Initialise Iteration and Convergence Plots</li>
        <ul>
            <li>Assembly of force vector and condensed DOFs</li>
            <li>Solution for linear elasticity</li>
        </ul>
    <li>Nonlinear Solution: Iteration with secant or tangent stiffness</li>
        <ul>
            <li>Definition of parameter <code>numit</code> (numbers of iterations for one simulation)</li>
            <li>Solution with secant stiffness (<code>it_type = 2</code>) Convergence control, Update Strains, Displacements and Rotations </li>
            <li>Solution with tangent stiffness (<code>it_type = 1</code>) Convergence control, Update Strains, Displacements and Rotations</li>
            <li>Collect solution values for printing and saving</li>
        </ul>
    <li>Postprocessing and Time Management</li>
</ol>

#### 3.2 <code>input_definition</code> function

Inputs: 

<ul>
  <li><code>mat</code>: See above (same input as for <code>main_solver</code>)</li>
</ul>

Output: 

<ul>
  <li><code>MATK, MASK </code>: Parameters related to material (?)</li>
  <li><code>NODESG, ELS, COORD, GEOMA, GEOMK</code>: Parameters related to geometry</li>
  <li><code>na, gauss_order, it_type</code>: Parameters related to solver type</li>
  <li><code>BC</code>: Boundary conditions</li>
</ul>


Content of function: 
<ol>
  <li>Geometry and Mesh: Definition of mesh size with parameter <code>ms</code> (here <code>ms = 
      L/2</code>) </li>
    <ul>
      <li>Definition of mesh points and lines<br>
      Definition of mesh areas</li>
      <li>Definition of number of layers <code>nl</code> per area, currently <code>nl=20</code> </li>
      <li>Assembly into gmsh model and embedded lines<br>
      Performing meshing</li>
    </ul>
  <li>Loads & BCs</li>
    <ul>
        <li>Definition of BC<br>
        Every boundary condition needs to be entered in terms of a np-array with the following
        entries: <code>[xmin,xmax,ymin,ymax,zmin,zmax,BC_ux,BC_uy,BC_uz,BC_thx,BC_thy,BC_thz]</code>,
        where the first 6 entries correspond to the location of the BC and the next to their prescribed 
        deformation / rotation. The value <code>1234</code> is set if a fixed BC is desired.</li>
        <li>Definition of the loads<br>
        Every load needs to be entered in terms of a np-array with the following
        entries:
          <ul>
          <li>Load_el: Global Element Loads $[N/mm^2]$<br>
          <code>[xmin,xmax,ymin,ymax,zmin,zmax,direction(x=1,y=2,z=3,thx=4,thy=5,thz=6),magnitude[N/mm2]] 
          </code></li>
          <li>Load_n: Global Nodal Loads $[N]$<br>
          <code>[xmin,xmax,ymin,ymax,zmin,zmax,direction(x=1,y=2,z=3,thx=4,thy=5,thz=6),magnitude[N]] 
          </code></li>
          </ul>
        </li>
      </ul>
  <li>Material & further parameters</li>
    <ul>
      <li>Choice of analysis: Linear elastic (<code>cma = 1</code>) or CMM (<code>cma = 3</code>)<br>
      For the steel simulation example, the parameter is permanently set to <code>cma = 1</code>
      </li>
      <li>Definition of iteration type <code>it_type</code> (see above) and gauss order 
      <code>gauss_order</code> (only 1 or 2) </li>
      <li>Input definition of material parameters, these are collected in the matrix <code>MATA</code></li>
    </ul>
  <li>Postprocess Mesh</li>
    <ul>
      <li>[tbd!]</li>
    </ul>
</ol>
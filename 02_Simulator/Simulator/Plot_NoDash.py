import matplotlib.pyplot as plt
import numpy as np
import pickle
import pandas as pd
import os
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import AxesGrid


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica",
    "text.latex.preamble": r'\usepackage{wasysym}'
})

def read(mat_res_raw):

    BC = mat_res_raw['BC']
    COORD = mat_res_raw['COORD']
    ELEMENTS = mat_res_raw['ELEMENTS']
    fe = mat_res_raw['fe']
    GEOMA = mat_res_raw['GEOMA']
    GEOMK = mat_res_raw['GEOMK']
    MASK = mat_res_raw['MASK']
    NODESG = mat_res_raw['NODESG']
    POST = mat_res_raw['POST']
    gauss_order = mat_res_raw['gauss_order']
    na = mat_res_raw['na']
    ux = mat_res_raw['ux'] 
    uy = mat_res_raw['uy'] 
    uz = mat_res_raw['uz']
    thx = mat_res_raw['thx']
    thy = mat_res_raw['thy']
    thz = mat_res_raw['thz']
    eps_g = mat_res_raw['eps_g']

    return BC, COORD, ELEMENTS, fe, GEOMA, GEOMK, MASK, NODESG, POST, gauss_order, na, ux, uy, uz, thx, thy, thz, eps_g


def plot_nodes(mat_res_raw):
    """ ------------------------------------- Create Nodes for Dash Plot  -------------------------------------------
        ----------------------------------------------- INPUT: ------------------------------------------------------
        - ELEMENTS, COORD,u
        ---------------------------------------------- OUTPUT: ------------------------------------------------------
        - nodes[Entire/Area][Undeformed,Deformed]: global node coordinates in undef. and def. state
        - elements[entire/area]: Element connectivity
        - [nx,ny,nz] = nodes[Entire][Undeformed]
        - Indicators_1 = [Entire, a1, a2,...,na]
    -----------------------------------------------------------------------------------------------------------------"""

    BC, COORD, ELEMENTS, fe, GEOMA, GEOMK, MASK, NODESG, POST, gauss_order, na, ux, uy, uz, thx, thy, thz, eps_g = read(mat_res_raw)


    # 0 Initiate
    nodes = {}
    elements = {}
    nodes['Entire']={}
    elements['Entire']=ELEMENTS

    # 1 Iteration through areas
    # - numelsia: number of elements in area ia
    # - elements[ia]: node connectivity of elements in area ia, shape (numelsia, 4)
    for ia in range(na):
        nodes[str(ia)]={}
        numelsia = len(np.array(GEOMK["ak"])[np.where(np.array(GEOMK["ak"])==ia)])
        elements[str(ia)]=np.reshape(ELEMENTS[np.where(np.array(GEOMK["ak"])==ia),:],(numelsia,4))

    # 2 Assignment of node coordinates for entire structure
    # - nx, ny, nz: node coordinates of all nodes
    # - nodes[Entire][Undeformed]: [nx,ny,nz]
    nx=np.array(COORD['n'][0][:,0]).reshape(len(ux),1)
    ny=np.array(COORD['n'][0][:,1]).reshape(len(ux),1)
    nz=np.array(COORD['n'][0][:,2]).reshape(len(ux),1)
    nodes['Entire']['Undeformed'] = [nx,ny,nz]

    # 3 Scale factor for plot scaling of deformations
    # - f: Scale factor
    # - nodes[Entire][Deformed] = [nx,ny,nz]+f*[ux,uy,uz]
    # if np.max(abs(nz))>0:
    #     f = np.max(abs(nz))/np.max(abs(uz))
    # else:
    #     f = np.max(abs(ny))/np.max(abs(uz))
    f = max(np.max(abs(nx)),np.max(abs(ny)),np.max(abs(nz)))/max(np.max(abs(ux)),np.max(abs(uy)),np.max(abs(uz)))/5
    nodes['Entire']['Deformed'] = [nx+f*ux,ny+f*uy,nz+f*uz]

    # 4 Assignment of node coordinates for each area
    # - nodes[ia][splot] for splot = [Deformed,Undeformed]: global node coordinates for individual areas
    for ia in range(na):
        nodes[str(ia)]['Undeformed'] = [nx[MASK[ia]], ny[MASK[ia]], nz[MASK[ia]]]
        nodes[str(ia)]['Deformed'] = [nx[MASK[ia]] + f * ux[MASK[ia]], ny[MASK[ia]] + f * uy[MASK[ia]],
                                      nz[MASK[ia]] + f * uz[MASK[ia]]]
        indicators1 = ['Entire']
        Indicators1 = list.append(indicators1, str(ia))
    return nodes, elements, nx, ny, nz, Indicators1



def plot_meshes(mat_res_raw):
    """ --------------------------------- Create Node Connectivity and Coordinates  ---------------------------------
        ----------------------------------------------- INPUT: ------------------------------------------------------
        - nodes, elements from plot_nodes() function
        ---------------------------------------------- OUTPUT: ------------------------------------------------------
        - [xmesh, ymesh, zmesh] at positions [aplot (area),splot (def/undef)]: Mesh Points in correct order for
          plotting of elements: xmesh = [xi_e0, xi_e1,....xi_ene]
    -----------------------------------------------------------------------------------------------------------------"""

    # 0 Initiate
    xmesh = {}
    ymesh = {}
    zmesh = {}

    # 1 Iteration through areas
    indicators1 = [str(0)]
    for aplot in indicators1:

        # 1.0 Initiation of meshpoint vectors
        xmesh[str(aplot)]= {}
        ymesh[str(aplot)] = {}
        zmesh[str(aplot)] = {}

        # 1.1 Iteration through splot = [Deformed,Undeformed]
        indicators2 = ['Undeformed','Deformed']
        nodes, elements, nx, ny, nz, Indicators1 = plot_nodes(mat_res_raw)
        for splot in indicators2:

            # 1.1.1 NODES_plt = coordinates of nodes to be plotted
            #       ELEMENTS_plt = node connectivity of plotted elements
            NODES_plt = nodes['Entire'][str(splot)]
            ELEMENTS_plt = elements[str(aplot)]

            # 1.1.2 Number of elements
            num_elements = len(ELEMENTS_plt[:, 0])

            # 1.1.3 Initiation of xall,yall,zall: vectors of plotted coordinates for nodes in correct element order
            xall = np.array([])
            yall = np.array([])
            zall = np.array([])

            # 1.1.4 Iteration through elements of area aplot
            for k in range(num_elements):

                # 1.1.4.1 Nodes constituting regarded element
                nk = ELEMENTS_plt[k, :][ELEMENTS_plt[k, :]<10**5]

                # 1.1.4.2 Coordinates of nodes nk
                nodes_x = NODES_plt[0][nk]
                nodes_y = NODES_plt[1][nk]
                nodes_z = NODES_plt[2][nk]

                # 1.1.4.3 Append first node to end to plot edge n4-n1 (n3-n1 for tris) as well
                nodes_x = np.append(nodes_x, nodes_x[0])
                nodes_x = np.append(nodes_x, None)
                nodes_y = np.append(nodes_y, nodes_y[0])
                nodes_y = np.append(nodes_y, None)
                nodes_z = np.append(nodes_z, nodes_z[0])
                nodes_z = np.append(nodes_z, None)

                # 1.1.4.4 Assemble nodes_x/y/z of regarded element to all-vectors containing all points for regarded
                #         area
                xall = np.append(xall,nodes_x)
                yall = np.append(yall,nodes_y)
                zall = np.append(zall,nodes_z)

            # 1.1.5 Assign x/y/zall to x/y/zmesh lists with options aplot = [Entire/0/1/2/../na] and
            #       splot = [Deformed/Undeformed]
            xmesh[str(aplot)][str(splot)] = xall
            ymesh[str(aplot)][str(splot)] = yall
            zmesh[str(aplot)][str(splot)] = zall
    return xmesh,ymesh,zmesh


def colorgrids(x,y,c,ia, mat_res_raw):
    """ ---------------------------------------- Create Colorgrids  -------------------------------------------------
        ----------------------------------------------- INPUT: ------------------------------------------------------
        - x,y: Local coordinates of regarded points
        - c: Color at given coordinates x&y
        - ia: area number
        ---------------------------------------------- OUTPUT: ------------------------------------------------------
        - xgrid,ygrid: local grid points at which colorgrid is defined
        - Ci: Color values at coordinates x/ygrid
    -----------------------------------------------------------------------------------------------------------------"""
    BC, COORD, ELEMENTS, fe, GEOMA, GEOMK, MASK, NODESG, POST, gauss_order, na, ux, uy, uz, thx, thy, thz, eps_g = read(mat_res_raw)

    # 1 Local node coordinates of area ia
    nxL = np.array(COORD['n'][2][ia][:, 0]).reshape(len(ux), 1)[MASK[ia]].flatten()
    nyL = np.array(COORD['n'][2][ia][:, 1]).reshape(len(ux), 1)[MASK[ia]].flatten()

    # 1.1 minimum and maximum coordinates
    xmin = min(nxL)
    ymin = min(nyL)
    xmax = max(nxL)
    ymax = max(nyL)

    # 2 Create Grid
    # 2.1 Grid Spacing
    dmesh = GEOMA["meshsa"][ia][0]

    # 2.2 Creation of meshgrid
    xgrid = np.linspace(xmin, xmax, int((xmax - xmin) / dmesh) * gauss_order)
    ygrid = np.linspace(ymin, ymax, int((ymax - ymin) / dmesh) * gauss_order)
    if xgrid.size == 0 or ygrid.size == 0:
        xgrid = np.linspace(xmin, xmax, int((xmax - xmin) / (dmesh/2)) * gauss_order)
        ygrid = np.linspace(ymin, ymax, int((ymax - ymin) / (dmesh/2)) * gauss_order)
    X, Y = np.meshgrid(xgrid, ygrid)

    # 2.3 Assign color data to grid

    # # 2.3.1 Extrapolate Data (on nodes or integration points) to nodes by using method 'nearest'
    # cgrid = griddata((x, y), c, (nxL, nyL), method='nearest')

    # # 2.3.1 Interpolate previously extrapolated data to grid points for plotting. Using 'cubic' allows for
    # #       arbitrary shapes to be contour filled
    # C = griddata((nxL, nyL), cgrid, (X, Y),method='cubic')


    # return [xgrid,ygrid,C]
    Xflat = X   #.flatten()
    Yflat = Y   #.flatten()

    from scipy.interpolate import Rbf
    rbf3 = Rbf(x, y, c, function="linear", smooth=1)
    cgrid = rbf3(Xflat,Yflat)
    return [Xflat,Yflat,cgrid]



def plot_colors(mat_res_raw):
    """ ----------------------------------- Create Information for Colorplots  --------------------------------------
        ----------------------------------------------- INPUT: ------------------------------------------------------
        - COORD, MASK, Solution output
        ---------------------------------------------- OUTPUT: ------------------------------------------------------
        - colors[ia][c] = [x_ip,y_ip,c], c = value for searched parameter at integration/center point in local coord
    -----------------------------------------------------------------------------------------------------------------"""

    BC, COORD, ELEMENTS, fe, GEOMA, GEOMK, MASK, NODESG, POST, gauss_order, na, ux, uy, uz, thx, thy, thz, eps_g = read(mat_res_raw)


    # 0 Initiate
    colors = {}

    # 1 Iteration throuhg areas
    for ia in range(na):

        # 1.0 Initiate colors
        colors[str(ia)]={}

        # 1.1 Local x- and y-coordinates of nodes of area ia
        xa = np.array(COORD['n'][2][ia][:, 0]).reshape(len(ux), 1)[MASK[ia]].flatten()
        ya = np.array(COORD['n'][2][ia][:, 1]).reshape(len(ux), 1)[MASK[ia]].flatten()

        # 1.2 Create colors-entries for Deformations and relative error of deformations (in nodes)
        for i in range(12):
            v=[ux, uy, uz,thx,thy,thz,POST["relunx"][0],POST["reluny"][0],POST["relunz"][0],POST["relthnx"][0],POST["relthny"][0],POST["relthnz"][0]][i]
            titles = ['ux','uy','uz','thx','thy','thz','relunx','reluny','relunz','relthnx','relthny','relthnz']
            colors[str(ia)][titles[i]] = colorgrids(xa,ya,v[MASK[ia]].flatten(),ia, mat_res_raw)

        # 1.3 Entries for residual forces (in element center points)
        colors[str(ia)]['RNx'] = colorgrids(xa, ya, POST['RNx'][1][ia], ia, mat_res_raw)
        colors[str(ia)]['RNy'] = colorgrids(xa, ya, POST['RNy'][1][ia], ia, mat_res_raw)
        colors[str(ia)]['RNxy'] = colorgrids(xa, ya, POST['RNxy'][1][ia], ia, mat_res_raw)

        # 1.4 Entries for residual moments (in element center points)
        colors[str(ia)]['RMx'] = colorgrids(xa, ya, POST['RMx'][1][ia], ia, mat_res_raw)
        colors[str(ia)]['RMy'] = colorgrids(xa, ya, POST['RMy'][1][ia], ia, mat_res_raw)
        colors[str(ia)]['RMxy'] = colorgrids(xa, ya, POST['RMxy'][1][ia], ia, mat_res_raw)

        # 1.5 Entries for moments (in element center points)
        colors[str(ia)]['Mx'] = colorgrids(COORD['c'][3][ia][:,0],COORD['c'][3][ia][:,1],POST['Mx'][ia],ia, mat_res_raw)
        colors[str(ia)]['My'] = colorgrids(COORD['c'][3][ia][:, 0], COORD['c'][3][ia][:, 1], POST['My'][ia], ia, mat_res_raw)
        colors[str(ia)]['Mxy'] = colorgrids(COORD['c'][3][ia][:, 0], COORD['c'][3][ia][:, 1], POST['Mxy'][ia], ia, mat_res_raw)

        # 1.6 Entries for membrane and shear forces (in element center points)
        colors[str(ia)]['Nx'] = colorgrids(COORD['c'][3][ia][:,0],COORD['c'][3][ia][:,1],POST['Nx'][ia],ia, mat_res_raw)
        colors[str(ia)]['Ny'] = colorgrids(COORD['c'][3][ia][:, 0], COORD['c'][3][ia][:, 1], POST['Ny'][ia], ia, mat_res_raw)
        colors[str(ia)]['Nxy'] = colorgrids(COORD['c'][3][ia][:, 0], COORD['c'][3][ia][:, 1], POST['Nxy'][ia], ia, mat_res_raw)
        colors[str(ia)]['Qx'] = colorgrids(COORD['c'][3][ia][:,0],COORD['c'][3][ia][:,1],POST['Qx'][ia],ia, mat_res_raw)
        colors[str(ia)]['Qy'] = colorgrids(COORD['c'][3][ia][:, 0], COORD['c'][3][ia][:, 1], POST['Qy'][ia], ia, mat_res_raw)

        # 1.5 b Generalised Normal Strains
        
        colors[str(ia)]['epsx'] = colorgrids(COORD['c'][3][ia][:,0],COORD['c'][3][ia][:,1],eps_g[:, :, :, 0].flatten(),ia, mat_res_raw)
        colors[str(ia)]['epsy'] = colorgrids(COORD['c'][3][ia][:,0],COORD['c'][3][ia][:,1],eps_g[:, :, :, 1].flatten(),ia, mat_res_raw)
        colors[str(ia)]['epsxy'] = colorgrids(COORD['c'][3][ia][:,0],COORD['c'][3][ia][:,1],eps_g[:, :, :, 2].flatten(),ia, mat_res_raw)

        # 1.6 b Generalised moment and shear strains

        colors[str(ia)]['chix'] = colorgrids(COORD['c'][3][ia][:,0],COORD['c'][3][ia][:,1],eps_g[:, :, :, 3].flatten(),ia, mat_res_raw)
        colors[str(ia)]['chiy'] = colorgrids(COORD['c'][3][ia][:,0],COORD['c'][3][ia][:,1],eps_g[:, :, :, 4].flatten(),ia, mat_res_raw)
        colors[str(ia)]['chixy'] = colorgrids(COORD['c'][3][ia][:,0],COORD['c'][3][ia][:,1],eps_g[:, :, :, 5].flatten(),ia, mat_res_raw)
        colors[str(ia)]['gamx'] = colorgrids(COORD['c'][3][ia][:,0],COORD['c'][3][ia][:,1],eps_g[:, :, :, 6].flatten(),ia, mat_res_raw)
        colors[str(ia)]['gamy'] = colorgrids(COORD['c'][3][ia][:,0],COORD['c'][3][ia][:,1],eps_g[:, :, :, 7].flatten(),ia, mat_res_raw)        

        # 1.7 Entries for strains and steel stresses (in integration points) in top and bottom layer
        colors[str(ia)]['ex'] = {}
        colors[str(ia)]['ex']['sup'] = colorgrids(COORD['ip'][3][ia][:, 0], COORD['ip'][3][ia][:, 1], POST['exsupa'][ia], ia, mat_res_raw)
        colors[str(ia)]['ex']['inf'] = colorgrids(COORD['ip'][3][ia][:, 0], COORD['ip'][3][ia][:, 1], POST['exinfa'][ia], ia, mat_res_raw)
        colors[str(ia)]['ey'] = {}
        colors[str(ia)]['ey']['sup'] = colorgrids(COORD['ip'][3][ia][:, 0], COORD['ip'][3][ia][:, 1], POST['eysupa'][ia], ia, mat_res_raw)
        colors[str(ia)]['ey']['inf'] = colorgrids(COORD['ip'][3][ia][:, 0], COORD['ip'][3][ia][:, 1], POST['eyinfa'][ia], ia, mat_res_raw)
        colors[str(ia)]['gxy'] = {}
        colors[str(ia)]['gxy']['sup'] = colorgrids(COORD['ip'][3][ia][:, 0], COORD['ip'][3][ia][:, 1], POST['gxysupa'][ia], ia, mat_res_raw)
        colors[str(ia)]['gxy']['inf'] = colorgrids(COORD['ip'][3][ia][:, 0], COORD['ip'][3][ia][:, 1], POST['gxyinfa'][ia], ia, mat_res_raw)
        colors[str(ia)]['e3'] = {}
        colors[str(ia)]['e3']['sup'] = colorgrids(COORD['ip'][3][ia][:, 0], COORD['ip'][3][ia][:, 1], POST['e3supa'][ia], ia, mat_res_raw)
        colors[str(ia)]['e3']['inf'] = colorgrids(COORD['ip'][3][ia][:, 0], COORD['ip'][3][ia][:, 1], POST['e3infa'][ia], ia, mat_res_raw)
        colors[str(ia)]['e1'] = {}
        colors[str(ia)]['e1']['sup'] = colorgrids(COORD['ip'][3][ia][:, 0], COORD['ip'][3][ia][:, 1], POST['e1supa'][ia], ia, mat_res_raw)
        colors[str(ia)]['e1']['inf'] = colorgrids(COORD['ip'][3][ia][:, 0], COORD['ip'][3][ia][:, 1], POST['e1infa'][ia], ia, mat_res_raw)
        colors[str(ia)]['th'] = {}
        colors[str(ia)]['th']['sup'] = colorgrids(COORD['ip'][3][ia][:, 0], COORD['ip'][3][ia][:, 1], POST['thsupa'][ia], ia, mat_res_raw)
        colors[str(ia)]['th']['inf'] = colorgrids(COORD['ip'][3][ia][:, 0], COORD['ip'][3][ia][:, 1], POST['thinfa'][ia], ia, mat_res_raw)
        colors[str(ia)]['ssx'] = {}
        colors[str(ia)]['ssx']['sup'] = colorgrids(COORD['ip'][3][ia][:, 0], COORD['ip'][3][ia][:, 1], POST['ssxsupa'][ia], ia, mat_res_raw)
        colors[str(ia)]['ssx']['inf'] = colorgrids(COORD['ip'][3][ia][:, 0], COORD['ip'][3][ia][:, 1], POST['ssxinfa'][ia], ia, mat_res_raw)
        colors[str(ia)]['ssy'] = {}
        colors[str(ia)]['ssy']['sup'] = colorgrids(COORD['ip'][3][ia][:, 0], COORD['ip'][3][ia][:, 1], POST['ssysupa'][ia], ia, mat_res_raw)
        colors[str(ia)]['ssy']['inf'] = colorgrids(COORD['ip'][3][ia][:, 0], COORD['ip'][3][ia][:, 1], POST['ssyinfa'][ia], ia, mat_res_raw)
        colors[str(ia)]['spx'] = {}
        colors[str(ia)]['spx']['sup'] = colorgrids(COORD['ip'][3][ia][:, 0], COORD['ip'][3][ia][:, 1], POST['spxsupa'][ia], ia, mat_res_raw)
        colors[str(ia)]['spx']['inf'] = colorgrids(COORD['ip'][3][ia][:, 0], COORD['ip'][3][ia][:, 1], POST['spxinfa'][ia], ia, mat_res_raw)
        colors[str(ia)]['spy'] = {}
        colors[str(ia)]['spy']['sup'] = colorgrids(COORD['ip'][3][ia][:, 0], COORD['ip'][3][ia][:, 1], POST['spysupa'][ia], ia, mat_res_raw)
        colors[str(ia)]['spy']['inf'] = colorgrids(COORD['ip'][3][ia][:, 0], COORD['ip'][3][ia][:, 1], POST['spyinfa'][ia], ia, mat_res_raw)

        # 1.8 Entries for relative strain errors (in integration points) in top and bottom layer
        colors[str(ia)]['relex'] = {}
        colors[str(ia)]['relex']['sup'] = colorgrids(COORD['ip'][3][ia][:, 0], COORD['ip'][3][ia][:, 1], POST['relexsupa'][ia], ia, mat_res_raw)
        colors[str(ia)]['relex']['inf'] = colorgrids(COORD['ip'][3][ia][:, 0], COORD['ip'][3][ia][:, 1], POST['relexinfa'][ia], ia, mat_res_raw)
        colors[str(ia)]['reley'] = {}
        colors[str(ia)]['reley']['sup'] = colorgrids(COORD['ip'][3][ia][:, 0], COORD['ip'][3][ia][:, 1], POST['releysupa'][ia], ia, mat_res_raw)
        colors[str(ia)]['reley']['inf'] = colorgrids(COORD['ip'][3][ia][:, 0], COORD['ip'][3][ia][:, 1], POST['releyinfa'][ia], ia, mat_res_raw)
        colors[str(ia)]['relgxy'] = {}
        colors[str(ia)]['relgxy']['sup'] = colorgrids(COORD['ip'][3][ia][:, 0], COORD['ip'][3][ia][:, 1], POST['relgxysupa'][ia], ia, mat_res_raw)
        colors[str(ia)]['relgxy']['inf'] = colorgrids(COORD['ip'][3][ia][:, 0], COORD['ip'][3][ia][:, 1], POST['relgxyinfa'][ia], ia, mat_res_raw)
    return colors


def find_node_range(xmin, xmax, ymin, ymax, zmin, zmax, mat_res_raw):

    COORD = mat_res_raw['COORD']
    NODESG = COORD["n"][0]
    nodesx = NODESG[:,0]
    nodesy = NODESG[:,1]
    nodesz = NODESG[:,2]
    ind1 = np.array(np.where(nodesx<=xmax)).ravel()
    ind2 = np.array(np.where(nodesx>=xmin)).ravel()
    indx = np.intersect1d(ind1,ind2)

    ind1 = np.array(np.where(nodesy <= ymax)).ravel()
    ind2 = np.array(np.where(nodesy >= ymin)).ravel()
    indy = np.intersect1d(ind1, ind2)

    ind1 = np.array(np.where(nodesz <= zmax)).ravel()
    ind2 = np.array(np.where(nodesz >= zmin)).ravel()
    indz = np.intersect1d(ind1, ind2)

    indxy = np.intersect1d(indx,indy)
    ind = np.intersect1d(indxy,indz)

    return ind


def plot_boundaries(mat_res_raw, fig):

    BC, COORD, ELEMENTS, fe, GEOMA, GEOMK, MASK, NODESG, POST, gauss_order, na, ux, uy, uz, thx, thy, thz, eps_g = read(mat_res_raw)

    mesh_size_plt = GEOMA["meshsa"][0][0]
    # B_plt = GEOMA["Ba"][0]
    B_plt = 1600
    order = 1
    lw = 5 * np.max(mesh_size_plt) / B_plt / order

    def plot_triangle(coord, d, i):
        d = d / order
        xtr = [coord[0]-d/2,coord[0]-d/2,coord[0]+d/2,coord[0]+d/2,coord[0]-d/2,coord[0],coord[0]+d/2,coord[0]+d/2,coord[0],coord[0]-d/2]
        ytr = [coord[1]+d/2,coord[1]-d/2,coord[1]-d/2,coord[1]+d/2,coord[1]+d/2,coord[1],coord[1]+d/2,coord[1]-d/2,coord[1],coord[1]-d/2]
        ztr = [coord[2]-d,coord[2]-d,coord[2]-d,coord[2]-d,coord[2]-d,coord[2],coord[2]-d,coord[2]-d,coord[2],coord[2]-d]
        fig.scatter(x=xtr,
                    y=ytr,
                    z=ztr,
                    # mode='lines',
                    # line=dict(color='rgb(0,0,0)', width=lw),
                    # row=1, col=1,
                    # showlegend=False
                    )
        if BC[i, 6] == 1:
            fig.add_cone(x=[coord[0]+d/3], y=[coord[1]], z=[coord[2]], u=[-d], v=[0], w=[0], colorscale=[[0, 'black'], [1, 'black']], showscale=False,
                            showlegend=False)
        if BC[i, 7] == 1:
            fig.add_cone(x=[coord[0]], y=[coord[1]+d/3], z=[coord[2]], u=[0], v=[-d], w=[0], colorscale=[[0, 'black'], [1, 'black']], showscale=False,
                            showlegend=False)
        if BC[i, 8] == 1:
            fig.add_cone(x=[coord[0]], y=[coord[1]], z=[coord[2]+d/3], u=[0], v=[0], w=[-d], colorscale=[[0, 'black'], [1, 'black']], showscale=False,
                            showlegend=False)
        if BC[i, 9] == 1:
            fig.add_cone(x=[coord[0]-d/3], y=[coord[1]], z=[coord[2]], u=[2*d/3], v=[0], w=[0], colorscale=[[0, 'grey'], [1, 'grey']], showscale=False,
                            showlegend=False)
            fig.add_cone(x=[coord[0]-d/2], y=[coord[1]], z=[coord[2]], u=[2*d/3], v=[0], w=[0], colorscale=[[0, 'grey'], [1, 'grey']], showscale=False,
                            showlegend=False)
        if BC[i, 10] == 1:
            fig.add_cone(x=[coord[0]], y=[coord[1]-d/3], z=[coord[2]], u=[0], v=[2*d/3], w=[0], colorscale=[[0, 'grey'], [1, 'grey']], showscale=False,
                            showlegend=False)
            fig.add_cone(x=[coord[0]], y=[coord[1]-d/2], z=[coord[2]], u=[0], v=[2*d/3], w=[0], colorscale=[[0, 'grey'], [1, 'grey']], showscale=False,
                            showlegend=False)
        if BC[i, 11] == 1:
            fig.add_cone(x=[coord[0]], y=[coord[1]], z=[coord[2]-d/3], u=[0], v=[0], w=[2*d/3], colorscale=[[0, 'grey'], [1, 'grey']], showscale=False,
                            showlegend=False)
            fig.add_cone(x=[coord[0]], y=[coord[1]], z=[coord[2]-d/2], u=[0], v=[0], w=[2*d/3], colorscale=[[0, 'grey'], [1, 'grey']], showscale=False,
                            showlegend=False)
        
        return

    numb = len(BC[:, 0])
    for i in range(numb):
        if BC[i, 6] + BC[i, 7] + BC[i, 8] + BC[i, 9] + BC[i, 10] > 0:
            xmin = BC[i, 0]
            xmax = BC[i, 1]
            ymin = BC[i, 2]
            ymax = BC[i, 3]
            zmin = BC[i, 4]
            zmax = BC[i, 5]
            nodesi = find_node_range(xmin, xmax, ymin, ymax, zmin, zmax)
            for j in nodesi:
                j = int(j)
                coord = NODESG[j, :]
                plot_triangle(coord, mesh_size_plt / 2, i)
    return



def plot_forces(mat_res_raw, fig):

    BC, COORD, ELEMENTS, fe, GEOMA, GEOMK, MASK, NODESG, POST, gauss_order, na, ux, uy, uz, thx, thy, thz, eps_g = read(mat_res_raw)

    f_max = max(abs(fe))
    max_conesize = GEOMA["meshsa"][0][0]*3
    max_linesize = int(GEOMA["meshsa"][0][0])*3
    for i in range(len(fe)):
        fe_i = fe[i]
        if abs(fe_i) > 0.01:
            n_i = int(i/6)
            [x,y,z] = COORD["n"][0][n_i]
            dir = i - n_i*6
            factor = max(1/4,abs(fe_i/f_max))*np.sign(fe_i)
            if dir == 0 or dir == 3:
                " Force in x - direction "
                x = x - int(max_conesize * factor) / 3
                u = max_conesize*factor
                v = [0]
                w = [0]

                xl = [x,x-int(max_linesize*factor)*2/3]
                yl = [y,y]
                zl = [z,z]
            elif dir == 1 or dir == 4:
                " Force in y - direction "
                y = y - int(max_conesize * factor) / 3
                u = [0]
                v = max_conesize * factor
                w = [0]

                xl = [x, x]
                yl = [y, y - int(max_linesize * factor)*2/3]
                zl = [z, z]
            elif dir == 2 or dir == 5:
                " Force in z - direction "
                z = z - int(max_conesize * factor)/3
                u = [0]
                v = [0]
                w = max_conesize * factor

                xl = [x, x]
                yl = [y, y]
                zl = [z, z - int(max_linesize * factor)*2/3]
            if abs(factor) < 0.251:
                if dir < 2.5:
                    color = 'orangered'
                else:
                    color = 'green'
            else:
                if dir < 2.5:
                    color = 'red'
                else:
                    color = 'blue'
            # fig.add_cone(x=[x], y=[y], z=[z], 
            #              u=u, v=v, w=w, colorscale=[[0, color], [1,color]], showscale = False, showlegend=False)
            fig.scatter(xl,
                        yl,
                        zl,
                        # mode='lines',
                        # line=dict(color='rgb(255,0,0)',width=abs(int(max_conesize * factor))/100),
                        # row=1, col=1,
                        # showlegend=False
                        )


def plot_node_nr(mat_res_raw, fig):

    BC, COORD, ELEMENTS, fe, GEOMA, GEOMK, MASK, NODESG, POST, gauss_order, na, ux, uy, uz, thx, thy, thz, eps_g = read(mat_res_raw)

    nn = len(NODESG[:, 0])
    for n in range(nn):
        fig.add_scatter3d(x=[NODESG[n][0]],
                            y=[NODESG[n][1]],
                            z=[NODESG[n][2]],
                            mode="markers+text",
                            text=str(n),
                            row=1, col=1,
                            showlegend=False)
    fig.update_traces(textposition='top center')

def plot_coordsys(mat_res_raw, fig):
    
    BC, COORD, ELEMENTS, fe, GEOMA, GEOMK, MASK, NODESG, POST, gauss_order, na, ux, uy, uz, thx, thy, thz, eps_g = read(mat_res_raw)

    max_conesize = GEOMA["meshsa"][0][0]/2
    max_linesize = int(GEOMA["meshsa"][0][0])/2
    fig.add_scatter3d(x=[0,0],
                        y=[0,0],
                        z=[0,max_linesize],
                        mode='lines',
                        line=dict(color='rgb(0,0,255)', width=1),
                        row=1, col=1,
                        showlegend=False)
    fig.add_cone(x=[0], y=[0], z=[max_linesize], u=[0], v=[0], w=[max_conesize], colorscale=[[0, 'blue'], [1, 'blue']], showscale=False,
                    showlegend=False)
    fig.add_scatter3d(x=[0],
                        y=[0],
                        z=[max_linesize*6/5],
                        mode="text",
                        textfont = dict(family = "times", color = "blue"),
                        text="<i>z</i>",
                        row=1, col=1,
                        showlegend=False)
    fig.add_scatter3d(x=[0,0],
                        z=[0,0],
                        y=[0,max_linesize],
                        mode='lines',
                        line=dict(color='rgb(0,0,255)', width=1),
                        row=1, col=1,
                        showlegend=False)
    fig.add_cone(x=[0], z=[0], y=[max_linesize], u=[0], w=[0], v=[max_conesize], colorscale=[[0, 'blue'], [1, 'blue']], showscale=False,
                    showlegend=False)
    fig.add_scatter3d(x=[0],
                        z=[0],
                        y=[max_linesize*6/5],
                        mode="text",
                        textfont = dict(family = "times", color = "blue"),
                        text="<i>y</i>",
                        row=1, col=1,
                        showlegend=False)
    fig.add_scatter3d(y=[0,0],
                        z=[0,0],
                        x=[0,max_linesize],
                        mode='lines',
                        line=dict(color='rgb(0,0,255)', width=1),
                        row=1, col=1,
                        showlegend=False)
    fig.add_cone(y=[0], z=[0], x=[max_linesize], v=[0], w=[0], u=[max_conesize], colorscale=[[0, 'blue'], [1, 'blue']], showscale=False,
                    showlegend=False)
    fig.add_scatter3d(y=[0],
                        z=[0],
                        x=[max_linesize*6/5],
                        mode="text",
                        textfont = dict(family = "times", color = "blue"),
                        text="<i>x</i>",
                        row=1, col=1,
                        showlegend=False)

    coordloc = np.array([[1,0,0],[0,1,0],[0,0,1]])@GEOMA["T1"][str(0)]
    Oo = GEOMA["Oa"][str(0)]
    for i in [0,1]:
        strr = ["<i>x'</i>","<i>y'</i>"][i]
        xx = [Oo[0] + coordloc[i][0] * max_linesize / 2]
        yy=  [Oo[1] + coordloc[i][1] * max_linesize / 2]
        zz = [Oo[2] + coordloc[i][2] * max_linesize / 2]
        uu = [int(coordloc[i][0]) * max_conesize]
        vv = [int(coordloc[i][1]) * max_conesize]
        ww = [int(coordloc[i][2]) * max_conesize]
        fig.add_scatter3d(x=[Oo[0],Oo[0] + coordloc[i][0] * max_linesize / 2],
                            y=[Oo[1],Oo[1] + coordloc[i][1] * max_linesize / 2],
                            z=[Oo[2],Oo[2] + coordloc[i][2] * max_linesize / 2],
                            mode='lines',
                            line=dict(color='rgb(0,200,0)', width=1),
                            row=1, col=1,
                            showlegend=False)
        fig.add_cone(x=xx,
                        y=yy,
                        z=zz,
                        u=uu,
                        v=vv,
                        w=ww,
                        colorscale=[[0, 'rgb(0,200,0)'], [1, 'rgb(0,200,0)']], showscale=False,
                        showlegend=False)
        fig.add_scatter3d(x=xx,
                            y=yy,
                            z=zz,
                            mode="text",
                            textfont = dict(family = "times", color = 'rgb(0,200,0)'),
                            text=strr,
                            row=1, col=1,
                            showlegend=False)



def main_plt(mat_res_raw:pd.DataFrame, id:str, path: str, nn = False, boundaries = True, forces = True, node_nr = True, coord_sys = True):
    """
    Plots 8 variables from output of simulation, for the single simulation specified in mat_res_raw.
    Note (units): The units here correspond to the ones of post-processing for sig, and u; for eps_g they correspond
                    to the eps_g in the matrix mat_res_raw (see definition of colors). 
                    The units have been checked to correspond to the simulation units [N, mm] everywhere.

    Input: 
    mat_res_raw     (pd.DataFrame)      One line of the data set that shall be investigated in more detail
    id              (str)               Identifier: "sig", "eps", "u", "geom+load"

    Output: 
    plt             (plt)               Plot of the desired 8 variables       

    """
    
    # Getting data
    [nodes, elements, nx, ny, nz, Indicators1] = plot_nodes(mat_res_raw)
    [xmesh,ymesh,zmesh] = plot_meshes(mat_res_raw)
    colors = plot_colors(mat_res_raw)
    
    
    # 0 - Plotting Geometry and Loads
    if id == 'geom+load':
        fig, ax = plt.subplots(1,2, subplot_kw={"projection": "3d"})

        xmesh_cl_deformed = [xi for xi in xmesh[str(0)]['Deformed'] if xi is not None]
        xmesh_cl_undeformed = [xi for xi in xmesh[str(0)]['Undeformed'] if xi is not None]
        ymesh_cl_deformed = [yi for yi in ymesh[str(0)]['Deformed'] if yi is not None]
        ymesh_cl_undeformed = [yi for yi in ymesh[str(0)]['Undeformed'] if yi is not None]
        zmesh_cl_deformed = [zi for zi in zmesh[str(0)]['Deformed'] if zi is not None]
        zmesh_cl_undeformed = [zi for zi in zmesh[str(0)]['Undeformed'] if zi is not None]

        plt.style.use('_mpl-gallery')

        # ax[0].plot_wireframe(xmesh_cl_undeformed,
        #             ymesh_cl_undeformed,
        #             zmesh_cl_undeformed, 
        #             rstride=10, cstride=10
        #             )

        # ax[1].plot_wireframe(xmesh_cl_deformed,
        #             ymesh_cl_deformed,
        #             zmesh_cl_deformed,
        #             #    mode='lines',
        #             #    line = dict(color = 'rgb(100,100,100)',width = 0.5),
        #             #    row=1,col=1,
        #             #    showlegend = False
        #             rstride=10, cstride=10
        #             )
        

        ax[0].scatter(xmesh_cl_undeformed,
                    ymesh_cl_undeformed,
                    zmesh_cl_undeformed, 
                    )

        ax[1].scatter(xmesh_cl_deformed,
                    ymesh_cl_deformed,
                    zmesh_cl_deformed,
                    )
    
        if boundaries:
            plot_boundaries(mat_res_raw, ax[0])
        if forces: 
            plot_forces(mat_res_raw, ax[0])
        # if node_nr:
        #     plot_node_nr(mat_res_raw, ax)
        # if coord_sys:
        #     plot_coordsys(mat_res_raw, ax)


    if nn:
        label_add = np.array([['$_{,NN}$', '$_{,NN}$', '$_{,NN}$'],
                              ['$_{,NN}$', '$_{,NN}$', '$_{,NN}$'],
                              ['$_{,NN}$', '$_{,NN}$', '$_{,NN}$']])
    elif not nn:
        label_add = np.array([['$_{,NLFEA}$', '$_{,NLFEA}$', '$_{,NLFEA}$'],
                              ['$_{,NLFEA}$', '$_{,NLFEA}$', '$_{,NLFEA}$'],
                              ['$_{,NLFEA}$', '$_{,NLFEA}$', '$_{,NLFEA}$']])
   
    if id == 'sig':
        name = np.array([['Nx', 'Ny', 'Nxy'], 
                         ['Mx', 'My', 'Mxy'],
                         ['Qx', 'Qy', 'Qy']])
        labels_sig = np.array([['$n_x$', '$n_y$', '$n_{xy}$'], 
                            ['$m_x$', '$m_y$', '$m_{xy}$'], 
                            ['$v_{xz}$', '$v_{yz}$', '$v_{yz}$']])
        labels = np.char.add(labels_sig, label_add)
        # label_colorbar = np.array(['$[kN/m]$', '$[kNm/m]$', '$[kN/m]$'])
        label_colorbar = np.array(['$[N/mm]$', '$[Nmm/mm]$', '$[N/mm]$'])
        n_rows = 3
        img = [0, 0, 0]
    
    elif id == 'eps':
        name = np.array([['epsx', 'epsy', 'epsxy'], 
                        ['chix', 'chiy', 'chixy'],
                        [ 'gamx', 'gamy', 'gamy']])
        labels_eps= np.array([[r'$\varepsilon_x$', r'$\varepsilon_y$', r'$\varepsilon_{xy}$'], 
                            [r'$\chi_x$', r'$\chi_y$', r'$\chi_{xy}$'],
                            [r'$\gamma_{xz}$', r'$\gamma_{xz}$', r'$\gamma_{xz}$']])
        labels = np.char.add(labels_eps, label_add)
        label_colorbar = np.array([r'$[mm/mm]$', r'$[1/mm]$', r'$[mm/mm]$'])
        n_rows = 3
        img = [0, 0, 0]

        # # Convert [-] strains to [permille] strains
        # for i in range(3):
        #     for j in range(3):
        #         if i == 2 and j==2:
        #             pass
        #         else:
        #             colors[str(0)][name[i,j]][2] = colors[str(0)][name[i,j]][2]*10**(3)
    
    elif id == 'u':
        name = np.array([['ux','uy','uz'],
                          ['thx','thy','thz']])
        labels_u = np.array([['$u_x$','$u_y$','$u_z$'],
                         [r'$\vartheta_x$',r'$\vartheta_y$',r'$\vartheta_z$']])
        labels = np.char.add(labels_u, label_add[0:2, :])
        label_colorbar = np.array(['$[mm]$', '$[mrad]$'])
        n_rows = 2
        img = [0, 0]


    # Convert mm to m (for the x-/ y- axes)
    for i in range(n_rows):
        for j in range(3):
            if i == 2 and j==2:
                pass
            else:
                colors[str(0)][name[i,j]][0] = colors[str(0)][name[i,j]][0]*10**(-3)
                colors[str(0)][name[i,j]][1] = colors[str(0)][name[i,j]][1]*10**(-3)
    
    # Define max, min for colorbar
    min_temp, max_temp = np.zeros((name.shape)), np.zeros((name.shape))
    vmin, vmax = np.zeros((name.shape)), np.zeros((name.shape))
    for i in range(n_rows):
        for j in range(3):
            min_temp[i,j] = np.min(np.array(colors[str(0)][name[i,j]][2]))
            max_temp[i,j] = np.max(np.array(colors[str(0)][name[i,j]][2]))
        vmin[i,:] = np.min(min_temp[i,:])*np.ones((1,3))
        vmax[i,:] = np.max(max_temp[i,:])*np.ones((1,3))


    # Plot
    fig, axs = plt.subplots(n_rows,3, figsize = [8, n_rows*2.7])
    for i in range(n_rows):
        for j in range(3):
            if i == 2 and j==2:
                axs[i,j].set_title(' ')
            else:
                axs[i,j].contour(colors[str(0)][name[i,j]][0], colors[str(0)][name[i,j]][1], colors[str(0)][name[i,j]][2],
                                        colors = 'black', linewidths = 0.5,
                                        vmin=vmin[i,j], vmax=vmax[i,j], 
                                        # levels = np.linspace(vmin[i,j], vmax[i,j], 10)
                                        levels = get_good_labels(vmin[i,j], vmax[i,j])
                                        )
                img[i] = axs[i,j].contourf(colors[str(0)][name[i,j]][0], colors[str(0)][name[i,j]][1], colors[str(0)][name[i,j]][2],
                                        vmin=vmin[i,j], vmax=vmax[i,j],
                                        levels = get_good_labels(vmin[i,j], vmax[i,j])
                                        )
                axs[i,j].set_title(labels[i,j])

    # adjust ranges of colorbar to make them more intuitive (for specific examples)
    # steps = np.zeros((n_rows, 1))
    # if id == 'sig':
    #     vmin[0,:] = -4*np.ones((1,3))
    #     vmax[0,:] = 10*np.ones((1,3))
    #     vmin[1,:] = 0*np.ones((1,3))
    #     vmax[1,:] = 1200*np.ones((1,3))
    #     vmin[2,:] = -40*np.ones((1,3))
    #     vmax[2,:] = 40*np.ones((1,3))
    #     for i in range(n_rows):
    #         steps[i,0] = int(vmax[i,0]-vmin[i,0])
    #         if steps[i,0]>1000:
    #             steps[i,0] = int(steps[i,0]/100)
    #     steps = steps.astype(int)

    for i in range(n_rows):
        cbar = fig.colorbar(img[i], label = label_colorbar[i], ax = axs[i,0:3], location= 'right')
        # cbar.set_ticks(np.linspace(vmin[i,0], vmax[i,0], steps[i,0]))
        for label in cbar.ax.get_yticklabels()[::2]:
            label.set_visible(False)
    if n_rows == 3:
        axs[-1, -1].axis('off')
    for i in range(n_rows):
        for j in range(3):
            # axs[i,j] = plt.gca()
            axs[i,j].set_aspect('equal', 'box')
            axs[i,j].axis('square')
   
    
    # plt.tight_layout()
    plt.savefig(os.path.join(path, 'single_'+ id +'.png'), format = 'png')
    plt.show()
    plt.close()


    return



def get_good_labels(vmin, vmax, no_steps = 10):
    step = (vmax-vmin)/no_steps
    if step == 0:
        num_steps = 2
        vmin_f = -0.5
        vmax_f = 0.5
    else: 
        n_round = np.floor(np.log10(step))
        step_ = np.round(step/(10**n_round))*(10**n_round)
        vmin_f = np.floor(vmin/step_)*step_
        vmax_f = np.ceil(vmax/step_)*step_
        num_steps = int((vmax_f-vmin_f)/step_)+1
    
    custom_levels = np.linspace(vmin_f, vmax_f, num_steps)

    return custom_levels
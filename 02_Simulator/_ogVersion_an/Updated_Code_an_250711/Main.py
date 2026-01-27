# -------------------------------------------------------------------------------------------------------------------- #
# Main File for FEM_Q Analysis
# (C) Andreas Näsbom, ETH Zürich
# 22.08.2021
# -------------------------------------------------------------------------------------------------------------------- #

# -------------------------------------------------------------------------------------------------------------------- #
# Python Versions information
# - Compatible with numpy version 1.18.1 (pip install numpy==1.18.1), on later versions, commands must be adjusted
# - Compatible with Tensorflow: numpy version 1.19.3
# -------------------------------------------------------------------------------------------------------------------- #

print("-------------------------------------------------------")
print("Start Executing FEM_Q Script")
print("-------------------------------------------------------")
# -------------------------------------------------------------------------------------------------------------------- #
# 0 Import
# -------------------------------------------------------------------------------------------------------------------- #
from Mesh_gmsh import MATK,gauss_order,numls,NODESG,it_type,ms

from fem import f_assemble,f0_assemble, v_stat_con, c_dof,find_fi, find_ss,find_s0
from fem import solve_sys,solve_0,find_e,find_e0,find_s,find_eh,find_sh,find_b
import numpy as np
import time
import matplotlib.pyplot as plt
start_main = time.time()

global convrf
# -------------------------------------------------------------------------------------------------------------------- #
# 1 Auxiliary Functions for Convergence Check, Plot and Save
# -------------------------------------------------------------------------------------------------------------------- #

def un_thn(u):
    un = np.array([])
    thn = np.array([])
    count = 0
    for j in range(len(u)):
        if count < 2.5:
            un = np.append(un,u[j],axis=0)
        else:
            thn = np.append(thn,u[j],axis=0)
        count += 1
        if count > 5.5:
            count = 0
    return un,thn

def plot_convergence(i,e,rele,un,relun,thn,relthn,r):
    global it_steps
    global conve50
    global conve90
    global conve99
    global convun50
    global convun90
    global convun99
    global convthn50
    global convthn90
    global convthn99
    global convrf
    global convrm

    global eold
    global unold
    global thnold
    global plt
    global fig
    global ax1
    global ax2
    global ax3

    if i == 0:
        it_steps = [0]
        conve50 = [np.percentile(abs(rele),50)]
        conve90 = [np.percentile(abs(rele), 90)]
        conve99 = [np.percentile(abs(rele), 99)]
        convun50 = [np.percentile(abs(relun),50)]
        convun90 = [np.percentile(abs(relun), 90)]
        convun99 = [np.percentile(abs(relun), 99)]
        convthn50 = [np.percentile(abs(relthn),50)]
        convthn90 = [np.percentile(abs(relthn), 90)]
        convthn99 = [np.percentile(abs(relthn), 99)]
        convrf = sum(abs(r[0::6]))+sum(abs(r[1::6]))+sum(abs(r[2::6]))
        convrm = sum(abs(r[3::6])) + sum(abs(r[4::6])) + sum(abs(r[5::6]))
        # convr99 = np.percentile(abs(rcond),99)
    else:
        it_steps = np.append(it_steps, i)
        conve50 = np.append(conve50, np.percentile(abs(rele),50))
        conve90 = np.append(conve90, np.percentile(abs(rele), 90))
        conve99 = np.append(conve99, np.percentile(abs(rele), 99))
        convun50 = np.append(convun50, np.percentile(abs(relun),50))
        convun90 = np.append(convun90, np.percentile(abs(relun), 90))
        convun99 = np.append(convun99, np.percentile(abs(relun), 99))
        convthn50 = np.append(convthn50, np.percentile(abs(relthn),50))
        convthn90 = np.append(convthn90, np.percentile(abs(relthn), 90))
        convthn99 = np.append(convthn99, np.percentile(abs(relthn), 99))
        convrf = np.append(convrf, sum(abs(r[0::6]))+sum(abs(r[1::6]))+sum(abs(r[2::6])))
        convrm = np.append(convrm, sum(abs(r[3::6])) + sum(abs(r[4::6])) + sum(abs(r[5::6])))

    if  abs(min(min(conve50),0.001)) < 10**12 and max(conve50) > 0 and max(conve50) < 10**12:
        ax1.axis([0, numit, min(min(conve50),0.001), max(conve99)])
    ax1.set_yscale("log")
    ax1.set_title("$\delta$" + "$\epsilon$$_{klij}$" + "/" +"$\epsilon$$_{klij}$" )
    ax1.plot(it_steps, conve50,'k')
    ax1.plot(it_steps, conve90, 'b')
    ax1.plot(it_steps, conve99, 'g')
    ax1.legend(["$P$$_{50}$", "$P$$_{90}$","$P$$_{99}$"], loc=3)
    ax1.grid(True)

    if abs(min(min(convun50), 0.001)) < 10 ** 12 and max(convun50) > 0 and max(convun50) < 10 ** 12:
        ax2.axis([0, numit, min(min(convun50),0.001), max(convun99)])
    ax2.set_yscale("log")
    ax2.set_title("$\delta$" + "$u$$_{n}$" + "/" +"$u$$_{n}$" )
    ax2.plot(it_steps, convun50,'k')
    ax2.plot(it_steps, convthn50,'k--')
    ax2.plot(it_steps, convun90,'b')
    ax2.plot(it_steps, convthn90,'b--')
    ax2.plot(it_steps, convun99,'g')
    ax2.plot(it_steps, convthn99,'g--')
    ax2.legend(['Displacements','Rotations'],loc = 3)
    ax2.grid(True)

    rrel = r/max(abs(r))
    if abs(min(min(convrf), 0.001)) < 10 ** 12 and max(convrm) > 0 and max(convrm) < 10 ** 12:
        ax3.axis([0, numit, min(min(convrf),10000), max(abs(convrm))])
    ax3.set_yscale("log")
    ax3.set_title("$Residual$")
    ax3.plot(it_steps, convrf, 'r')
    ax3.plot(it_steps, convrm, 'm')
    ax3.legend(["$R$$_{F}$", "$R$$_{M}$"], loc=3)
    ax3.grid(True)

    plt.subplots_adjust(hspace=0.5)
    plt.pause(1)
    if i < numit-1:
        eold = e
        unold = un
        thnold = thn

    return conve90[-1],convun90[-1],convrf[-1]

def saveres(j,u,e,ex,ey,gxy,B,gauss_order,s,sh0,e0,MATK,fi,fe,f0,cDOF, relun, relthn):
    """--------------------------------------------------------------------------------------------------------------"""
    """------------------------ Extract results, create folder for load step and save results------------------------"""
    """--------------------------------------------------------------------------------------------------------------"""
    # 0 Import
    from Mesh_gmsh import ELS, COORD, GEOMA, MASK, GEOMK, na, BC
    ELEMENTS = ELS[0]
    from Postprocess import post
    import numpy as np
    from numpy import save
    import pickle
    import os
    import shutil

    global eold

    # 1 Solution Value Collection
    " 1   Output:   - ui,thi: Nodal displacements and rotations in global coordinates"
    "               - ehc, ec: Generalized strains and strains in element centers in local coordinates"
    "               - sh: Generalized stresses in element centers in local coordinates"
    "               - [s,sx,sy,txy]: Stresses in integration points in local coordinates"
    "               - [ssx,ssy]: Steel stresses in integration points in local coordinates"
    "               - [Nx,...Qy]: Stress Resultants (sectional forces) in element centers in local coordinates"
    ux = u[0::6]
    uy = u[1::6]
    uz = u[2::6]
    thx = u[3::6]
    thy = u[4::6]
    thz = u[5::6]
    sh = find_sh(s, 1) + sh0
    [ssx, ssy, spx, spy, e1, e3, thr, thc, cm] = find_ss(s, MATK["cm"])
    if max(MATK["cm"]) < 1.5:
        ssx = np.zeros_like(ex)
        ssy = np.zeros_like(ex)
        spx = np.zeros_like(ex)
        spy = np.zeros_like(ex)
    Nx = sh[:, :, :, 0]
    Ny = sh[:, :, :, 1]
    Nxy = sh[:, :, :, 2]
    Mx = sh[:, :, :, 3]
    My = sh[:, :, :, 4]
    Mxy = sh[:, :, :, 5]
    Qx = sh[:, :, :, 6]
    Qy = sh[:, :, :, 7]
    r = np.add(fi, -(fe + f0))
    diffe = eold - e
    rele = np.divide(diffe, e)

    # 2 Import POST
    " 2 Output:   -POST: post processed data from postprocess function"
    POST = post(ux, uy, uz, r, thx, thy, thz, Nx, Ny, Nxy, Mx, My, Mxy, Qx, Qy, ex, ey, gxy, e3, e1, thr, thc, cm, ssx, ssy, spx,
                spy, rele, relun, relthn)

    # 3 Export Data

    # 3.1 File Path
    # path = r"C:\Users\naesboma\00_an\FEM_Q\Calculations\24_04_25_bav"
    # path = "C:/Users/naesboma/00_an/FEM_Q/Calculations/24_05_31_Plattenstreifen"
    # path = r"C:\Users\naesboma\00_an\FEM_Q\Calculations\24_11_11_Prototyp"
    # path = r"C:\Users\naesboma\00_an\FEM_Q\Calculations\24_11_19_Querkraftträger"
    # path = r'C:\Users\naesboma\00_an\FEM_Q\Calculations\24_12_17_LUSET'
    # path = r"C:\Users\naesboma\00_an\FEM_Q\Calculations\25_01_08_CrackEl"
    # path = r"C:\Users\naesboma\00_an\FEM_Q\Calculations\25_07_07_Plate_Thick"
    path = r"C:\Users\naesboma\00_an\FEM_Q\Calculations\0000_Random"

    # 3.2 Join With Load Step Number
    a = [path, r"\LS", str(j)]
    path = ''.join(a)
    # a = [path, r"\ms", str(ms)]
    # path = ''.join(a)

    # 3.3 Create Folder
    try:
        os.makedirs(path)
    except FileExistsError:
        shutil.rmtree(path)
        os.makedirs(path)
    except PermissionError:
        print(f"Permission denied: Unable to create '{path}'. Load Step results not saved")
    except Exception as e:
        print(f"An error occurred: {e}. Load Step results not saved")

    # 3.4 Export
    # Boundary Conditions
    save(os.path.join(path, 'BC.npy'), BC)

    # COORD
    with open(os.path.join(path, 'COORD.pkl'), 'wb') as f:
        pickle.dump(COORD, f)

    # ELEMENTS
    with open(os.path.join(path, 'ELEMENTS.pkl'), 'wb') as f:
        pickle.dump(ELEMENTS, f)

    # Applied Loads
    save(os.path.join(path, 'fe.npy'), fe)

    # GEOMA
    with open(os.path.join(path, 'GEOMA.pkl'), 'wb') as f:
        pickle.dump(GEOMA, f)

    # GEOMK
    with open(os.path.join(path, 'GEOMK.pkl'), 'wb') as f:
        pickle.dump(GEOMK, f)

    # MASK
    with open(os.path.join(path, 'MASK.pkl'), 'wb') as f:
        pickle.dump(MASK, f)

    # NODESG
    with open(os.path.join(path, 'NODESG.pkl'), 'wb') as f:
        pickle.dump(NODESG, f)

    # POST
    with open(os.path.join(path, 'POST.pkl'), 'wb') as f:
        pickle.dump(POST, f)

    # gauss order
    with open(os.path.join(path, 'gauss_order.pkl'), 'wb') as f:
        pickle.dump(gauss_order, f)

    # Number of areas
    with open(os.path.join(path, 'na.pkl'), 'wb') as f:
        pickle.dump(na, f)

    # Deformations
    save(os.path.join(path, 'u.npy'), u)
    save(os.path.join(path, 'ux.npy'), ux)
    save(os.path.join(path, 'uy.npy'), uy)
    save(os.path.join(path, 'uz.npy'), uz)
    save(os.path.join(path, 'thx.npy'), thx)
    save(os.path.join(path, 'thy.npy'), thy)
    save(os.path.join(path, 'thz.npy'), thz)

    # Reactions
    reac = np.concatenate((np.ndarray.reshape(cDOF, len(cDOF), 1), r[cDOF]), axis=1)
    np.savetxt(os.path.join(path, 'reac.txt'), reac)

    # Reactions in z
    Rz = r[cDOF[divmod(cDOF, 6)[1] == 2]]
    cDOFz = cDOF[divmod(cDOF, 6)[1] == 2]
    nz = (cDOFz - 2) / 6
    cz = np.zeros((len(nz), 3))
    for i in range(len(nz)):
        cz[i, :] = NODESG[int(nz[i]), :]
    # reacz = np.concatenate((cz, Rz), axis=1)
    reacz = np.concatenate((np.ndarray.reshape(nz, len(nz), 1), np.concatenate((cz, Rz), axis=1)), axis=1)
    np.savetxt(os.path.join(path, 'reacz.txt'), reacz)

    # Convergence Plot
    plt.savefig(os.path.join(path, 'Convergenceplot'))

    return ux, uy, r

# -------------------------------------------------------------------------------------------------------------------- #
# 2 Initiation of Iteration and Convergence Plots
# -------------------------------------------------------------------------------------------------------------------- #
print("-------------------------------------------------------")
print("2 Solution")
print("-------------------------------------------------------")
# 2.1 Initiate Figure and Axes for Convergence Plot
fig = plt.figure()
ax1 = fig.add_subplot(3,1,1)
ax2 = fig.add_subplot(3,1,2)
ax3 = fig.add_subplot(3,1,3)

# 2.2 Vector of External Forces and Condensed DOFs
" 2.2 Output:   - fe: External Forces per DOF"
"               - cDOF: Condensed DOFs"
print("2.1 Assembly of force vector and condensed DOFs")
# 2.2.1 Vector of External Forces
fe = f_assemble(0)
B = find_b(gauss_order)

# 2.2.2 Residual Strains
[e0, ex0, ey0, gxy0] = find_e0(gauss_order)
[e0c, ex0c, ey0c, gxy0c] = find_e0(1)

# 2.2.3 Vector of External Forces Caused by Internal Stresses
s0 = find_s0()
[sh0,f0] = f0_assemble(B,s0,1)

# 2.2.4 Condensed DOFs
cDOF,cVAL = c_dof()

# 2.3 Iteration Initiation: Solve Equation for linear elasticity
" 2.3 Output:   - u: Deformation for linear elasticity"
"               - eh: Generalized strains for linear elasticity"
"               - [e,ex,ey,gxy,e1,e3,th]: Strains epsilon(u)-e0 "
"               - [un,thn]: Displacements and rotations separated"
"               - eold, unold, thnold: Initiations as i-1-Iteration Step values"
"               - Initiation of r (residual vector): size of u"
print("2.2 Solution for Linear Elasticity")

u = solve_0(B,fe+f0,np.zeros_like(e0),cDOF,cVAL)

uz = u[2::6]
start_main = time.time()
eh = find_eh(B,u,gauss_order)
[e,ex,ey,gxy] = find_e(e0,eh,gauss_order)

s_lin = find_s(e,0,gauss_order,True)
s_prev = s_lin
s = find_s(e,s_prev,gauss_order)

sh = find_sh(s, gauss_order)

eold = e
[un, thn] = un_thn(u)
unold = un
thnold = thn
r = u

# -------------------------------------------------------------------------------------------------------------------- #
# 3 Nonlinear Solution: Iteration with Secant/Tangent Stiffness
# -------------------------------------------------------------------------------------------------------------------- #

# 3.1 Number of Iteration Steps
" 3.1 Output:   - numit: Number of iteration steps"
numit = 6*numls

if numit == 0:
    fi = find_fi(B, sh)
    r = np.zeros_like(r)
    rcond = v_stat_con(r, cDOF, np.zeros_like(cVAL))

    [un, thn] = un_thn(u)
    diffun = np.zeros_like(un)
    relun = np.zeros_like(un)
    maxun = np.max(abs(un)) / 1000
    diffun[np.where(abs(un) > maxun)] = np.ndarray.flatten(unold[np.where(abs(un) > maxun)]) - np.ndarray.flatten(
        un[np.where(abs(un) > maxun)])
    relun[np.where(abs(un) > maxun)] = np.divide(diffun[np.where(abs(un) > maxun)],
                                                 np.ndarray.flatten(un[np.where(abs(un) > maxun)]))
    diffun[np.where(abs(un) < maxun)] = 0
    relun[np.where(abs(un) < maxun)] = 0


    diffthn = np.zeros_like(un)
    relthn = np.zeros_like(un)
    maxthn = np.max(abs(thn)) / 1000
    diffthn[np.where(abs(thn) > maxthn)] = np.ndarray.flatten(thnold[np.where(abs(thn) > maxthn)]) - np.ndarray.flatten(
        thn[np.where(abs(thn) > maxthn)])
    relthn[np.where(abs(thn) > maxthn)] = np.divide(diffthn[np.where(abs(thn) > maxthn)],
                                                    np.ndarray.flatten(thn[np.where(abs(thn) > maxthn)]))
    diffthn[np.where(abs(thn) < maxthn)] = 0
    relthn[np.where(abs(thn) < maxthn)] = 0

    ux, uy, r = saveres(0, u, e,ex,ey,gxy, B, gauss_order, s, sh0, e0, MATK, fi, fe, f0, cDOF, relun, relthn)

elif it_type == 1:
    # 3.2  Tangent Iteration
    print("2.3 Tangent Stiffness Iteration")
    fi = find_fi(B, sh)
    i = -1
    for j in range(numls):
        print("2.3." + str(j) + " Start solving load step " + str(j))
        fe = f_assemble(j)
        contit = True
        while contit:
            i += 1
            # 3.2.1 Solution with Tangent Stiffness for given Iteration Step
            " 3.2.1 Output: - s,sx,sy,txy: Stresses"
            "               - u: Deformations"
            "               - eh: Generalised strains"
            "               - [e, ex, ey, gxy, e1, e3, th]: Strains epsilon(u)-e0"

            du=solve_sys(B,fi-fe, cDOF,np.zeros_like(cVAL), MATK["cm"],e,s)
            u -= du
            eh = find_eh(B,u, gauss_order)
            [e, ex, ey, gxy] = find_e(e0, eh, gauss_order)
            s = find_s(e,s_prev,gauss_order)
            sh = find_sh(s, gauss_order)
            fi = find_fi(B, sh)

            # 3.2.2 Convergence Control: Residual Vector
            " 3.2.2 Output: - r: Residual = fi-fe"
            "               - rcond: Residual without condensed DOFs"
            r = fi-fe-f0
            rcond = v_stat_con(r, cDOF,np.zeros_like(cVAL))

            # 3.2.3 Convergence Control: Relative Change in Strains
            " 3.2.3 Output: - maxe: maximum absolute value of strain for which relative changes are tracked. "
            "                       Defined in order not to track changes in (close to) zero values"
            "               - diffe: e(i) - e(i-1), difference in strain between iteration steps, for strains > maxe"
            "               - rele: relative difference: diffe./e(i) at locations where abs(e(i)) > maxe"
            e[e<-99999] = 0
            maxe = np.max(abs(e))/1000
            diffe = np.ndarray.flatten(eold[np.where(abs(e)>maxe)])-np.ndarray.flatten(e[np.where(abs(e)>maxe)])
            rele = np.divide(diffe,np.ndarray.flatten(e[np.where(abs(e)>maxe)]))

            # 3.2.4 Convergence Control: Relative Change in Displacement
            " 3.2.4 Output: - [un ,thn]: node displacements and rotations separated"
            "               - maxun: maximum absolute value of displacement for which relative changes are tracked. "
            "                       Defined in order not to track changes in (close to) zero values"
            "               - diffun: un(i) - un(i-1), difference in displacement between iteration steps, for displ. > maxun"
            "               - relun: relative difference: diffun./un(i) at locations where abs(un(i)) > maxun"
            [un, thn] = un_thn(u)
            diffun = np.zeros_like(un)
            relun = np.zeros_like(un)
            maxun = np.max(abs(un))/1000
            diffun[np.where(abs(un)>maxun)] = np.ndarray.flatten(unold[np.where(abs(un)>maxun)])-np.ndarray.flatten(un[np.where(abs(un)>maxun)])
            relun[np.where(abs(un)>maxun)] = np.divide(diffun[np.where(abs(un)>maxun)],np.ndarray.flatten(un[np.where(abs(un)>maxun)]))
            diffun[np.where(abs(un) < maxun)] = 0
            relun[np.where(abs(un) < maxun)] = 0

            # 3.2.5 Convergence Control: Relative Change in Rotation
            " 3.2.5 Output: - maxthn: maximum absolute value of rotation for which relative changes are tracked. "
            "                       Defined in order not to track changes in (close to) zero values"
            "               - diffthn: thn(i) - thn(i-1), difference in displacement between iteration steps, for rot. > maxthn"
            "               - relthn: relative difference: diffthn./thn(i) at locations where abs(thn(i)) > maxthn"
            diffthn = np.zeros_like(un)
            relthn = np.zeros_like(un)
            maxthn = np.max(abs(thn))/1000
            diffthn[np.where(abs(thn)>maxthn)] = np.ndarray.flatten(thnold[np.where(abs(thn)>maxthn)])-np.ndarray.flatten(thn[np.where(abs(thn)>maxthn)])
            relthn[np.where(abs(thn)>maxthn)] = np.divide(diffthn[np.where(abs(thn)>maxthn)],np.ndarray.flatten(thn[np.where(abs(thn)>maxthn)]))
            diffthn[np.where(abs(thn) < maxthn)] = 0
            relthn[np.where(abs(thn) < maxthn)] = 0

            # 3.2.6 Manipulate Residual for Plotting
            " 3.2.6 Output:   - r: residual with 0 at condensed DOFs"
            r[cDOF.astype(int)] = 0


            # 3.2.7 Convergence Control: Plot Iteration Step and Display Sum of Residual Forces
            conve90_i,convun90_i,convrf_i = plot_convergence(i, e, rele,un,relun,thn,relthn,r)
            print(" - Iteration step " + str(i) + " complete, 90% quantile of relative strain error = " + str(np.round(abs(conve90_i), 4)))
            print(" - Iteration step " + str(i) + " complete, 90% quantile of relative displacement error = " + str(np.round(abs(convun90_i), 4)))
            print(" - Iteration step " + str(i) + " complete, sum of residual forces = " + str(np.round(abs(convrf_i), 1)))

            # 3.2.8 Convergence Control: Check criteria and continue
            epsc = 0.02
            if abs(conve90_i)<epsc and abs(convun90_i)<epsc*10000 and abs(convrf_i) < np.sum(abs(fe)*1):
                print("Load step " + str(j) + " converged")
                ux,uy,r = saveres(j, u, e,ex,ey,gxy, B, gauss_order, s, sh0, e0, MATK, fi, fe, f0, cDOF, relun, relthn)
                contit = False
                s_prev = s
            elif i == numit:
                print("No solution found for load step " + str(j))
                contit = False

# elif it_type == 2:
#     # 3.2  Secant Iteration
#     print("2.3 Secant Stiffness Iteration")
#     fi = find_fi(B, sh)
#     i = -1
#     for j in range(numls):
#         print("2.3." + str(j) + " Start solving load step " + str(j))
#         fe = f_assemble(j)
#         contit = True
#         while contit:
#             i+=1
#             # 3.2.1 Solution with Secant Stiffness for given Iteration Step
#             " 3.2.1 Output: - s,sx,sy,txy: Stresses"
#             "               - u: Deformations"
#             "               - eh: Generalized strains"
#             "               - [e, ex, ey, gxy, e1, e3, th]: Strains epsilon(u)-e0"
#
#             u = solve_sys(B, fe+f0, cDOF, cVAL, MATK["cm"], e, s)
#             eh = find_eh(B,u, gauss_order)
#             [e, ex, ey, gxy] = find_e(e0, eh, gauss_order)
#             s = find_s(e,s_prev,gauss_order)
#             sh = find_sh(s, gauss_order)
#             fi = find_fi(B, sh)
#             s_prev = s
#
#             # 3.2.2 Convergence Control: Residual Vector
#             " 3.2.2 Output: - r: Residual = fi-fe"
#             "               - rcond: Residual without condensed DOFs"
#             r = fi - fe - f0
#             rcond = v_stat_con(r, cDOF, np.zeros_like(cVAL))
#
#             # 3.2.3 Convergence Control: Relative Change in Strains
#             " 3.2.3 Output: - maxe: maximum absolute value of strain for which relative changes are tracked. "
#             "                       Defined in order not to track changes in (close to) zero values"
#             "               - diffe: e(i) - e(i-1), difference in strain between iteration steps, for strains > maxe"
#             "               - rele: relative difference: diffe./e(i) at locations where abs(e(i)) > maxe"
#             e[e<-99999] = 0
#             maxe = np.max(abs(e))/1000
#             diffe = np.ndarray.flatten(eold[np.where(abs(e)>maxe)])-np.ndarray.flatten(e[np.where(abs(e)>maxe)])
#             rele = np.divide(diffe,np.ndarray.flatten(e[np.where(abs(e)>maxe)]))
#
#             # 3.2.4 Convergence Control: Relative Change in Displacement
#             " 3.2.4 Output: - [un ,thn]: node displacements and rotations separated"
#             "               - maxun: maximum absolute value of displacement for which relative changes are tracked. "
#             "                       Defined in order not to track changes in (close to) zero values"
#             "               - diffun: un(i) - un(i-1), difference in displacement between iteration steps, for displ. > maxun"
#             "               - relun: relative difference: diffun./un(i) at locations where abs(un(i)) > maxun"
#             [un, thn] = un_thn(u)
#             diffun = np.zeros_like(un)
#             relun = np.zeros_like(un)
#             maxun = np.max(abs(un))/1000
#             diffun[np.where(abs(un)>maxun)] = np.ndarray.flatten(unold[np.where(abs(un)>maxun)])-np.ndarray.flatten(un[np.where(abs(un)>maxun)])
#             relun[np.where(abs(un)>maxun)] = np.divide(diffun[np.where(abs(un)>maxun)],np.ndarray.flatten(un[np.where(abs(un)>maxun)]))
#             diffun[np.where(abs(un) < maxun)] = 0
#             relun[np.where(abs(un) < maxun)] = 0
#
#             # 3.2.5 Convergence Control: Relative Change in Rotation
#             " 3.2.5 Output: - maxthn: maximum absolute value of rotation for which relative changes are tracked. "
#             "                       Defined in order not to track changes in (close to) zero values"
#             "               - diffthn: thn(i) - thn(i-1), difference in displacement between iteration steps, for rot. > maxthn"
#             "               - relthn: relative difference: diffthn./thn(i) at locations where abs(thn(i)) > maxthn"
#             diffthn = np.zeros_like(un)
#             relthn = np.zeros_like(un)
#             maxthn = np.max(abs(thn))/1000
#             diffthn[np.where(abs(thn)>maxthn)] = np.ndarray.flatten(thnold[np.where(abs(thn)>maxthn)])-np.ndarray.flatten(thn[np.where(abs(thn)>maxthn)])
#             relthn[np.where(abs(thn)>maxthn)] = np.divide(diffthn[np.where(abs(thn)>maxthn)],np.ndarray.flatten(thn[np.where(abs(thn)>maxthn)]))
#             diffthn[np.where(abs(thn) < maxthn)] = 0
#             relthn[np.where(abs(thn) < maxthn)] = 0
#
#             # 3.2.6 Manipulate Residual for Plotting
#             " 3.2.6 Output:   - r: residual with 0 at condensed DOFs"
#             r[cDOF.astype(int)] = 0
#
#             # 3.2.7 Convergence Control: Plot Iteration Step and Display Sum of Residual Forces
#             conve90_i, convun90_i, convrf_i = plot_convergence(i, e, rele, un, relun, thn, relthn, r)
#             print(" - Iteration step " + str(i) + " complete, 90% quantile of relative strain error = " + str(
#                 np.round(abs(conve90_i), 4)))
#             print(" - Iteration step " + str(i) + " complete, 90% quantile of relative displacement error = " + str(
#                 np.round(abs(convun90_i), 4)))
#             print(" - Iteration step " + str(i) + " complete, sum of residual forces = " + str(
#                 np.round(abs(convrf_i), 1)))
#
#             # 3.2.8 Convergence Control: Check criteria and continue
#             if abs(conve90_i) < 0.02 and abs(convun90_i) < 0.02 and abs(convrf_i) < np.sum(abs(fe) * 2):
#                 print("Load step " + str(j) + " converged")
#                 ux, uy, r = saveres(j, u, e,ex,ey,gxy, B, gauss_order, s, sh0, e0, MATK, fi, fe, f0, cDOF, relun, relthn)
#                 contit = False
#             elif i == numit:
#                 print("No solution found for load step " + str(j))
#                 contit = False
# 3.3 Collect Values for Convergence
" 3.3 Output:   - strain values in [k][l][i][j] format"
plt.show(block=False)
# diffe = eold - e
# rele = np.divide(diffe,e)

print(" - Maximum displacements:")
print("   - ux_max = " + str(np.round(float(max(np.abs(ux))), 3)))
print("   - uy_max = " + str(np.round(float(max(np.abs(uy))), 3)))
print("   - uz_max = " + str(np.round(float(max(np.abs(uz))), 3)))
print(" - Sum of applied forces:")
print("   - Fx = " + str(np.round(float(sum(fe[0::6])), 3)))
print("   - Fy = " + str(np.round(float(sum(fe[1::6])), 3)))
print("   - Fz = " + str(np.round(float(sum(fe[2::6])), 3)))
print(" - Sum of reaction forces:")
print("   - Rx = " + str(np.round(float(sum(r[cDOF[divmod(cDOF, 6)[1] == 0].astype(int)])), 1)))
print("   - Ry = " + str(np.round(float(sum(r[cDOF[divmod(cDOF, 6)[1] == 1].astype(int)])), 1)))
print("   - Rz = " + str(np.round(float(sum(r[cDOF[divmod(cDOF, 6)[1] == 2].astype(int)])), 1)))

# 4.1 Time Management
end_main = time.time()
print("total time used: ", end_main - start_main)
from fem import time_stress, time_strain, time_K, time_B, time_Kinv, time_sh, time_eh

print("time used in stress calculation", time_stress.time_spent)
print("time used in strain calculation", time_strain.time_spent)
print("time used in stiffness matrix calculation", time_K.time_spent)
print("time used in B-matrix calculation", time_B.time_spent)
print("time used in stiffness matrix inversion", time_Kinv.time_spent)
print("time used in calculation of sh", time_sh.time_spent)
print("time used in calculation of eh", time_eh.time_spent)
print('Finished :)')




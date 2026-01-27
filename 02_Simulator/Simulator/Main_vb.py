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
from Mesh_gmsh_vb import input_definition
from fem_vb import fem_func
# from fem_vb import f_assemble,f0_assemble, v_stat_con, c_dof,find_fi, find_ss,find_s0
# from fem_vb import solve_sys,solve_0,find_e,find_e0,find_s,find_eh,find_sh,find_b
import numpy as np
import time
import matplotlib.pyplot as plt

# global convrf
# -------------------------------------------------------------------------------------------------------------------- #
# 1 Auxiliary Functions for Convergence Check and Plot
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


def plot_convergence(i,e,rele,un,relun,thn,relthn,r, numit, convrf, convrm):
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
    # global convrf
    # global convrm
    # global convr99
    # global ann1
    # global ann11
    # global ann12
    # global ann13
    # global ann2
    # global ann22
    # global ann23
    # global ann24
    # global ann3
    # global ann33
    # global ann34
    # global ann35
    # global eold
    # global unold
    # global thnold
    global plt
    global fig
    global ax1
    global ax2
    global ax3

    fig = plt.figure()
    ax1 = fig.add_subplot(3,1,1)
    ax2 = fig.add_subplot(3,1,2)
    ax3 = fig.add_subplot(3,1,3)

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
        # convrf = sum(abs(r[0::6]))+sum(abs(r[1::6]))+sum(abs(r[2::6]))
        # convrm = sum(abs(r[3::6])) + sum(abs(r[4::6])) + sum(abs(r[5::6]))
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
        # convrf = np.append(convrf, sum(abs(r[0::6]))+sum(abs(r[1::6]))+sum(abs(r[2::6])))
        # convrm = np.append(convrm, sum(abs(r[3::6])) + sum(abs(r[4::6])) + sum(abs(r[5::6])))
        # convr99 = np.append(convr99, int(np.percentile(abs(rcond),99)))
        
    # if i > 0:
        # ann1.remove()
        # ann11.remove()
        # ann12.remove()
        # ann13.remove()
        # ann2.remove()
        # ann22.remove()
        # ann23.remove()
        # ann24.remove()
        # ann3.remove()
        # ann33.remove()
        # ann34.remove()
        # ann35.remove()
    if  abs(min(min(conve50),0.001)) < 10**12 and max(conve50) > 0 and max(conve50) < 10**12:
        ax1.axis([0, numit, min(min(conve50),0.001), max(conve99)])
    ax1.set_yscale("log")
    ax1.set_title("$\delta$" + "$\epsilon$$_{klij}$" + "/" +"$\epsilon$$_{klij}$" )
    ax1.plot(it_steps, conve50,'k')
    ax1.plot(it_steps, conve90, 'b')
    ax1.plot(it_steps, conve99, 'g')
    ax1.legend(["$P$$_{50}$", "$P$$_{90}$","$P$$_{99}$"], loc=3)
    ax1.grid(True)
    # ann1 = ax1.annotate("$f$$_{max}= $" + str(np.round(max(abs(rele)), 3)), xy=(numit*7/10, max(conve) / 10))
    # ann11 = ax1.annotate("$f$$_{<0.01}= $" + str(np.round(100*len(rele[abs(rele)<0.01])/len(rele), 3)) +  " %",
    #                      xy=(numit*7/10, max(conve) / 1000))
    # ann12 = ax1.annotate("$f$$_{99}= $" + str(np.round(np.percentile(abs(rele),99), 3)), xy=(numit*7/10, max(conve) / 100))
    # ann13 = ax1.annotate("$f$$_{ex<0.01}= $" + str(np.round(100*len(rele[0::3][abs(rele[0::3])<0.01])/len(rele[0::3]), 3)) + " %",
    #                      xy=(numit*7/10, max(conve) / 10000))
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
    # ann2 = ax2.annotate("$f$$_{max}= $" + str(np.round(max(abs(relun)), 3)) + ' / ' +str(np.round(max(abs(relthn)), 3)) , xy=(numit*7/10, max(convun) / 5))
    # ann22 = ax2.annotate("$f$$_{<0.01}= $" + str(np.round(100*len(relun[abs(relun)<0.01])/len(relun), 3)) + ' / ' + str(np.round(100*len(relthn[abs(relthn)<0.01])/len(relthn), 3)) + " %",
    #                      xy=(numit*7/10, max(convun) / 20))
    # ann23 = ax2.annotate("$f$$_{99}= $" + str(np.round(np.percentile(abs(relun),99), 3)) + ' / ' + str(np.round(np.percentile(abs(relthn),99), 3)) , xy=(numit*7/10, max(convun) / 10))
    # ann24 = ax2.annotate("$f$$_{uz<0.01}= $" + str(np.round(100*len(rcond[2::6][abs(relu[2::6])<0.01])/len(relu[2::6]), 3)) + " %",
    #                      xy=(numit*7/10, max(convu) / 10000))

    rrel = r/max(abs(r))
    if abs(min(min(convrf), 0.001)) < 10 ** 12 and max(convrm) > 0 and max(convrm) < 10 ** 12:
        ax3.axis([0, numit, min(min(convrf),0.001), max(abs(convrm))])
    ax3.set_yscale("log")
    ax3.set_title("$Residual$")
    ax3.plot(it_steps, convrf, 'r')
    ax3.plot(it_steps, convrm, 'm')
    ax3.legend(["$R$$_{F}$", "$R$$_{M}$"], loc=3)
    ax3.grid(True)
    # ax3.plot(it_steps, convr99, 'b')
    # ann3 = ax3.annotate("$R$$_{max}= $" + str(np.round(int(max(abs(rcond))), 3)), xy=(numit * 4 / 5, int(max(abs(convr))) / 100))
    # ann33 = ax3.annotate("$R$$_{<0.01}= $" + str(np.round(100*len(rrel[abs(rrel)<0.01])/len(rrel), 3)) + " %",xy=(numit * 4 / 5, int(max(abs(convr))) / 10000))
    # ann34 = ax3.annotate("$R$$_{99}= $" + str(np.round(int(np.percentile(abs(rcond),99)), 3)),
    #                     xy=(numit * 4 / 5, int(max(abs(convr))) / 1000))
    # ann34 = ax3.annotate("$R$$_{x,max}= $" + str(np.round(int(max(abs(rcond[0::8]))), 3)),
    #                     xy=(numit * 4 / 5, int(max(abs(convr))) / 10000))
    # ann35 = ax3.annotate("$R$$_{Rx<0.01}= $" + str(np.round(100*len(rrel[0::6][abs(rrel[0::8])<0.01])/len(rrel[0::8]), 3)) + " %",xy=(numit * 4 / 5, int(max(abs(convr))) / 100000))
    plt.subplots_adjust(hspace=0.5)
    plt.pause(1)
    # if i < numit-1:
    #     eold = e
    #     unold = un
    #     thnold = thn

    # plt.show()

def main_solver(mat: dict, conv_plt: bool):
    '''
    Function to start main solver, calls input definition function (with geometry and material definition)

    INPUTS:
    mat         (dict)      Containing all potential input features
    conv_plt    (bool)      To turn on or off plotting of convergence plots

    OUTPUTS:
    mat_res     (dict)      Contains all relevant outputs from simulation that are required for data set
    saved files to 06_LinElData\02_Simulator\Beispielrechnung\results
    
    '''

    start_main = time.time()
    # -------------------------------------------------------------------------------------------------------------------- #
    # 1 running mesh_gmsh_vb script as "input_definition" function
    # -------------------------------------------------------------------------------------------------------------------- #    

    MATK, NODESG, ELS, COORD, GEOMA, GEOMK, MASK, na, BC, gauss_order, it_type, Load_el, Load_n, copln = input_definition(mat)
    fem_func0 = fem_func(MATK, NODESG, ELS, COORD, GEOMA, GEOMK, MASK, na, BC, gauss_order, it_type, Load_el, Load_n, copln)
    # -------------------------------------------------------------------------------------------------------------------- #
    # 2 Initiation of Iteration and Convergence Plots
    # -------------------------------------------------------------------------------------------------------------------- #
    
    print("-------------------------------------------------------")
    print("2 Solution")
    print("-------------------------------------------------------")
    # 2.1 Initiate Figure and Axes for Convergence Plot
    # fig = plt.figure()
    # ax1 = fig.add_subplot(3,1,1)
    # ax2 = fig.add_subplot(3,1,2)
    # ax3 = fig.add_subplot(3,1,3)

    # 2.2 Vector of External Forces and Condensed DOFs
    " 2.2 Output:   - fe: External Forces per DOF"
    "               - cDOF: Condensed DOFs"
    print("2.1 Assembly of force vector and condensed DOFs")
    # 2.2.1 Vector of External Forces
    fe = fem_func0.f_assemble(Load_el, Load_n)
    B = fem_func0.find_b(gauss_order)

    # 2.2.2 Residual Strains
    [e0, ex0, ey0, gxy0, e10, e30, th0] = fem_func0.find_e0(gauss_order)
    [e0c, ex0c, ey0c, gxy0c, e10c, e30c, th0c] = fem_func0.find_e0(1)

    # 2.2.3 Vector of External Forces Caused by Internal Stresses (z.B. Schwinden)
    s0 = fem_func0.find_s0()
    [sh0,f0] = fem_func0.f0_assemble(B,s0,1)

    # 2.2.4 Constrained DOFs
    cDOF,cVAL = fem_func0.c_dof()

    # 2.3 Iteration Initiation: Solve Equation for linear elasticity
    " 2.3 Output:   - u: Deformation for linear elasticity"
    "               - eh: Generalized strains for linear elasticity"
    "               - [e,ex,ey,gxy,e1,e3,th]: Strains epsilon(u)-e0 "
    "               - [un,thn]: Displacements and rotations separated"
    "               - eold, unold, thnold: Initiations as i-1-Iteration Step values"
    "               - Initiation of r (residual vector): size of u"
    print("2.2 Solution for Linear Elasticity")

    u, De_tot = fem_func0.solve_0(B,fe+f0,np.zeros_like(e0),cDOF,cVAL)
    print('u', u[0:6])
    # u, Ke_tot = fem_func0.solve_0(B,fe+f0,np.zeros_like(e0),cDOF,cVAL)
    # print('u1', u)

    # path = "C:/Users/naesboma/00_an/FEM_Q/Calculations/23_07_27_Versuch/Q=1400"
    # from numpy import load
    # import os
    # u = load(os.path.join(path, 'u.npy'))


    uz = u[2::6]
    start_main = time.time()
    eh = fem_func0.find_eh(B,u,gauss_order)
    [e,ex,ey,gxy,e1,e3,th] = fem_func0.find_e(e0,eh,gauss_order)
    print('eh', eh[0,0,0,:])
    print('e', e[0,0,0,:])

    s_lin = fem_func0.find_s(e,0,gauss_order,True)
    s_prev = s_lin
    s = fem_func0.find_s(e,s_prev,gauss_order)
    st = s[0][10][0][0][0]
    print('s_klij', np.array([st.sx.real,st.sy.real,st.txy.real,st.txz.real,st.tyz.real]))
    sh = fem_func0.find_sh(s, gauss_order)
    print('sh', sh[0,0,0,:])

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
    if MATK['cm'][0] == 1 or MATK['cm'][0] == 10:
        numit = 0
    elif MATK['cm'][0] == 3:
        numit = 7

    # Create arrays for cumulative saving of eh, sh, De
    sh_cum = np.zeros((numit+1, *sh.shape))
    eh_cum = np.zeros((numit+1, *eh.shape))
    De_cum = np.zeros((numit+1, *De_tot.shape))
    sh_cum[0,:,:,:,:] = sh
    eh_cum[0,:,:,:,:] = eh
    De_cum[0,:,:,:,:,:] = De_tot

    if numit == 0:
        fi = fem_func0.find_fi(B, sh)
        r = np.zeros_like(r)
        rcond = fem_func0.v_stat_con(r, cDOF, np.zeros_like(cVAL))

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

    elif it_type == 2:
        # 3.2  Secant Iteration
        print("2.3 Secant Stiffness Iteration")
        for i in range(numit):

            # 3.2.1 Solution with Secant Stiffness for given Iteration Step
            " 3.2.1 Output: - s,sx,sy,txy: Stresses"
            "               - u: Deformations"
            "               - eh: Generalized strains"
            "               - [e, ex, ey, gxy, e1, e3, th]: Strains epsilon(u)-e0"

            # u, Ke_tot = fem_func0.solve_sys(B, fe+f0, cDOF, cVAL, MATK["cm"], e, s)
            u, De_tot = fem_func0.solve_sys(B, fe+f0, cDOF, cVAL, MATK["cm"], e, s)
            eh = fem_func0.find_eh(B,u, gauss_order)
            [e, ex, ey, gxy, e1, e3, th] = fem_func0.find_e(e0, eh, gauss_order)
            s = fem_func0.find_s(e,gauss_order)
            sh = fem_func0.find_sh(s, gauss_order)

            # 3.2.2 Convergence Control: Residual Vector
            " 3.2.2 Output: - r: Residual = fi-fe"
            "               - rcond: Residual without condensed DOFs"
            fi = fem_func0.find_fi(B,sh)
            r = np.add(fi, -(fe+f0))
            rcond = fem_func0.v_stat_con(r, cDOF,np.zeros_like(cVAL))
            print(" - Iteration step " + str(i) + " complete, maximum residual = " + str(np.round(np.max(abs(rcond)),1)))

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
            if i ==0: 
                convrf = sum(abs(r[0::6]))+sum(abs(r[1::6]))+sum(abs(r[2::6]))
                convrm = sum(abs(r[3::6])) + sum(abs(r[4::6])) + sum(abs(r[5::6]))   
            else: 
                convrf = np.append(convrf, sum(abs(r[0::6]))+sum(abs(r[1::6]))+sum(abs(r[2::6])))
                convrm = np.append(convrm, sum(abs(r[3::6])) + sum(abs(r[4::6])) + sum(abs(r[5::6])))
            
            if conv_plt:    
                plot_convergence(i, e, rele,un,relun,thn,relthn,r, numit, convrf, convrm)
        
            if i < numit-1:
                eold = e
                unold = un
                thnold = thn
            print(" - Iteration step " + str(i) + " complete, sum of residual forces = " + str(np.round(abs(convrf[-1]), 1)))

    elif it_type == 1:
        # 3.2  Tangent Iteration
        print("2.3 Tangent Stiffness Iteration")
        for i in range(numit):

            # 3.2.1 Solution with Secant Stiffness for given Iteration Step
            " 3.2.1 Output: - s,sx,sy,txy: Stresses"
            "               - u: Deformations"
            "               - eh: Generalized strains"
            "               - [e, ex, ey, gxy, e1, e3, th]: Strains epsilon(u)-e0"

            fi = fem_func0.find_fi(B, sh)
            print('fi', fi[0:6])
            # du, Ke_tot =fem_func0.solve_sys(B,fi-fe, cDOF,np.zeros_like(cVAL), MATK["cm"],e,s)
            du, De_tot=fem_func0.solve_sys(B,fi-fe, cDOF,np.zeros_like(cVAL), MATK["cm"],e,s)
            u -= du
            # print('u', u[0:6])
            eh = fem_func0.find_eh(B,u, gauss_order)
            # print('eh', eh[0,0,0,:])
            [e, ex, ey, gxy, e1, e3, th] = fem_func0.find_e(e0, eh, gauss_order)
            s = fem_func0.find_s(e,s_prev,gauss_order)
            sh = fem_func0.find_sh(s, gauss_order)
            # print('sh', sh[0,0,0,:])

            # 3.2.2 Convergence Control: Residual Vector
            " 3.2.2 Output: - r: Residual = fi-fe"
            "               - rcond: Residual without condensed DOFs"
            r = np.add(fi, -(fe+f0))
            rcond = fem_func0.v_stat_con(r, cDOF,np.zeros_like(cVAL))
            print(" - Iteration step " + str(i) + " complete, maximum residual = " + str(np.round(np.max(abs(rcond)),1)))

            # 3.2.3 Convergence Control: Relative Change in Strains
            " 3.2.3 Output: - maxe: maximum absolute value of strain for which relative changes are tracked. "
            "                       Defined in order not to track changes in (close to) zero values"
            "               - diffe: e(i) - e(i-1), difference in strain between iteration steps, for strains > maxe"
            "               - rele: relative difference: diffe./e(i) at locations where abs(e(i)) > maxe"
            e[e<-99999] = 0
            maxe = np.max(abs(e))/1000
            diffe = np.ndarray.flatten(eold[np.where(abs(e)>maxe)])-np.ndarray.flatten(e[np.where(abs(e)>maxe)])
            rele = np.divide(diffe,np.ndarray.flatten(e[np.where(abs(e)>maxe)]))
            # print('Iteration step', i, 'and corresponding rele value', rele)

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
            if i ==0: 
                convrf = sum(abs(r[0::6]))+sum(abs(r[1::6]))+sum(abs(r[2::6]))
                convrm = sum(abs(r[3::6])) + sum(abs(r[4::6])) + sum(abs(r[5::6]))   
            else: 
                convrf = np.append(convrf, sum(abs(r[0::6]))+sum(abs(r[1::6]))+sum(abs(r[2::6])))
                convrm = np.append(convrm, sum(abs(r[3::6])) + sum(abs(r[4::6])) + sum(abs(r[5::6])))
            
            # Collect cumulative data
            sh_cum[i+1,:,:,:,:] = sh
            eh_cum[i+1,:,:,:,:] = eh
            De_cum[i+1,:,:,:,:,:] = De_tot


            if conv_plt:  
                plot_convergence(i, e, rele,un,relun,thn,relthn,r, numit, convrf, convrm)
            if i < numit-1:
                eold = e
                unold = un
                thnold = thn
            print(" - Iteration step " + str(i) + " complete, sum of residual forces = " + str(np.round(abs(convrf[-1]), 1)))
            s_prev = s


    # 3.3 Collect Values for Convergence
    " 3.3 Output:   - strain values in [k][l][i][j] format"
    # plt.show(block=False)
    diffe = eold - e
    rele = np.divide(diffe,e)

    # cutoff the "irrelevant data" for cumulative vectors
    De_cum_ = De_cum[1:,:,:,:,:,:]
    sh_cum_ = sh_cum[:-1,:,:,:,:]
    eh_cum_ = eh_cum[:-1,:,:,:,:]

    De_cum_re = De_cum_.reshape((-1,8,8))
    sh_cum_re = sh_cum_.reshape((-1,8))
    eh_cum_re = eh_cum_.reshape((-1,8))

    De_coll = De_cum_re.reshape((-1,1,1,8,8))
    sh_coll = sh_cum_re.reshape((-1,1,1,8))
    eh_coll = eh_cum_re.reshape((-1,1,1,8))



    # 3.4 Solution Value Collection
    " 3.4 Output:   - ui,thi: Nodal displacements and rotations in global coordinates"
    "               - ehc, ec: Generalized strains and strains in element centers in local coordinates"
    "               - sh: Generalized stresses in element centers in local coordinates"
    "               - [s,sx,sy,txy]: Stresses in integration points in local coordinates"
    "               - [ssx,ssy]: Steel stresses in integration points in local coordinates"
    "               - [Nx,...Qy]: Stress Resultants (sectional forces) in element centers in local coordinates"
    # print('u2',u)
    ux = u[0::6]
    uy = u[1::6]
    uz = u[2::6]
    thx = u[3::6]
    thy = u[4::6]
    thz = u[5::6]
    eh = fem_func0.find_eh(B,u, gauss_order)
    # print('eh2', eh[:,0,0,0])


    Bc = fem_func0.find_b(1)
    ehc = fem_func0.find_eh(Bc,u,1)
    # print('ehc', ehc[:,0,0,0])
    ec = fem_func0.find_e(e0c,ehc,1)[0]
    sc = fem_func0.find_s(ec,s_prev,1)
    sh = fem_func0.find_sh(sc,1)+sh0
    if max(MATK["cm"]) == 3:
        [ssx, ssy, spx, spy] = fem_func0.find_ss(s,MATK["cm"])
    else:
        ssx = np.zeros_like(ex)
        ssy = np.zeros_like(ex)
        spx = np.zeros_like(ex)
        spy = np.zeros_like(ex)
    [e, ex, ey, gxy, e1, e3, th] = fem_func0.find_e(np.zeros_like(e0), eh, gauss_order)
    Nx = sh[:, :, :, 0]
    Ny = sh[:, :, :, 1]
    Nxy = sh[:, :, :, 2]
    Mx = sh[:, :, :, 3]
    My = sh[:, :, :, 4]
    Mxy = sh[:, :, :, 5]
    Qx = sh[:, :, :, 6]
    Qy = sh[:, :, :, 7]

    # 3.5 Print Information: Displacements, Applied Loads and Reactions
    print("2.4 Solution complete")
    r = np.add(fi, -(fe+f0))

    print(" - Maximum displacements:")
    print("   - ux_max = " + str(np.round(float(max(np.abs(ux))),3)))
    print("   - uy_max = " + str(np.round(float(max(np.abs(uy))),3)))
    print("   - uz_max = " + str(np.round(float(max(np.abs(uz))),3)))
    print(" - Sum of applied forces:")
    print("   - Fx = " + str(np.round(float(sum(fe[0::6])),3)))
    print("   - Fy = " + str(np.round(float(sum(fe[1::6])),3)))
    print("   - Fz = " + str(np.round(float(sum(fe[2::6])),3)))
    print(" - Sum of reaction forces:")
    print("   - Rx = " + str(np.round(float(sum(r[cDOF[divmod(cDOF,6)[1]==0].astype(int)])),1)))
    print("   - Ry = " + str(np.round(float(sum(r[cDOF[divmod(cDOF,6)[1]==1].astype(int)])),1)))
    print("   - Rz = " + str(np.round(float(sum(r[cDOF[divmod(cDOF,6)[1]==2].astype(int)])),1)))

    # -------------------------------------------------------------------------------------------------------------------- #
    # 4 Postprocessing and Time Management
    # -------------------------------------------------------------------------------------------------------------------- #

    # 4.0 Import
    # from Mesh_gmsh import ELS, COORD,GEOMA,MASK,GEOMK,na,BC
    # ELEMENTS = ELS[0]
    # from Postprocess import post
    # import numpy as np
    from Postprocess_vb import post_func
    from numpy import save
    import pickle
    import os

    # 4.1 Time Management
    end_main = time.time()
    print("total time used: ", end_main-start_main)
    from fem_vb import time_stress,time_strain,time_K,time_B,time_Kinv,time_sh,time_eh
    print("time used in stress calculation", time_stress.time_spent)
    print("time used in strain calculation", time_strain.time_spent)
    print("time used in stiffness matrix calculation", time_K.time_spent)
    print("time used in B-matrix calculation", time_B.time_spent)
    print("time used in stiffness matrix inversion", time_Kinv.time_spent)
    print("time used in calculation of sh", time_sh.time_spent)
    print("time used in calculation of eh", time_eh.time_spent)

    # 4.2 Postprocess Data
    print("-------------------------------------------------------")
    print("3 Postprocess Data")
    print("-------------------------------------------------------")


    # 4.2.1 Import POST
    " 4.2.1 Output:   -POST: post processed data from postprocess function"
    post_func0 = post_func(COORD, GEOMK, ELS, GEOMA)
    POST = post_func0.post(ux,uy,uz,r,thx,thy,thz,Nx,Ny,Nxy,Mx,My,Mxy,Qx,Qy,ex,ey,gxy,e3,e1,th,ssx,ssy,spx,spy,rele,relun,relthn)

    # 4.3 Export Data

    # 4.3.1 File Path
    # path = "C:/Users/naesboma/00_an/FEM_Q/Calculations/23_08_01_Dreieck"
    path = r"C:\Users\vbalmer\OneDrive\Dokumente\GitHub\research_vera\02_Computations\06_LinElData\02_Simulator\Beispielrechnung\results"

    ############################################################################
    # DO NOT USE THIS
    # this has created some confusion and rendered parts of the data set unusable. 
    # it transformed the moments to [kNm/m], the shear and normal strains to [permille] and the curvatures to [mili-degrees] 
    # --> completely wrong units for further calculations!
    # bring moments into correct units
    # sh[:, :, :, 3:6] = sh[:, :, :, 3:6]*10**(-3)
    # # bring normal and shear strains into correct units
    # ehc[:, :, :, 0:3] = ehc[:, :, :, 0:3]*10**(3)
    # ehc[:, :, :, 6:8] = ehc[:, :, :, 6:8]*10**(3)
    # #bring curvatures into correct units
    # ehc[:, :, :, 3:6] = ehc[:, :, :, 3:6]*180/np.pi
    # # bring rotations into correct curvatures
    # thx, thy, thz = thx*10**3, thy*10**3, thz*10**3
    ##############################################################################

    mat_res = {
        # 'L': mat['L'],
        # 'B': mat['B'],
        'BC': BC,
        'COORD_c': COORD["c"],
        'COORD': COORD,
        'ELEMENTS': ELS[0],
        'MATK': MATK,
        'fe': fe,
        'GEOMA': GEOMA,
        'GEOMK': GEOMK, 
        'MASK': MASK,
        'NODESG': NODESG,
        'POST': POST,
        'gauss_order': gauss_order,
        'na': na,
        # 'sig_g': sh,
        # 'eps_g': ehc,
        'sig_g': sh_coll,
        'eps_g': eh_coll,
        'ux': ux,
        'uy': uy,
        'uz': uz, 
        'thx': thx,
        'thy': thy,
        'thz': thz,
        # 'De_tot': De_tot
        'De_tot': De_coll,
        # 'Ke_tot': Ke_tot,
        }
    
    mat_res.update(mat)
    # mat_res = mat.update(mat_res)

    # # Boundary Conditions
    # save(os.path.join(path, 'BC.npy'),BC)

    # # COORD
    # with open(os.path.join(path,'COORD.pkl'), 'wb') as f:
    #     pickle.dump(COORD, f)

    # # ELEMENTS
    # with open(os.path.join(path,'ELEMENTS.pkl'), 'wb') as f:
    #     pickle.dump(ELS[0], f)

    # # Applied Loads
    # save(os.path.join(path, 'fe.npy'), fe)

    # # GEOMA
    # # with open(os.path.join(path,'GEOMA.pkl'), 'wb') as f:
    # #     pickle.dump(GEOMA, f)

    # # GEOMK
    # with open(os.path.join(path,'GEOMK.pkl'), 'wb') as f:
    #     pickle.dump(GEOMK, f)

    # # MASK
    # with open(os.path.join(path,'MASK.pkl'), 'wb') as f:
    #     pickle.dump(MASK, f)

    # # NODESG
    # with open(os.path.join(path,'NODESG.pkl'), 'wb') as f:
    #     pickle.dump(NODESG, f)

    # # POST
    # with open(os.path.join(path,'POST.pkl'), 'wb') as f:
    #     pickle.dump(POST, f)

    # # gauss order
    # with open(os.path.join(path,'gauss_order.pkl'), 'wb') as f:
    #     pickle.dump(gauss_order, f)

    # # Number of areas
    # with open(os.path.join(path,'na.pkl'), 'wb') as f:
    #     pickle.dump(na, f)

    # # Deformations
    # save(os.path.join(path, 'u.npy'),u)
    # save(os.path.join(path, 'ux.npy'),ux)
    # save(os.path.join(path, 'uy.npy'),uy)
    # save(os.path.join(path, 'uz.npy'),uz)
    # save(os.path.join(path, 'thx.npy'),thx)
    # save(os.path.join(path, 'thy.npy'),thy)
    # save(os.path.join(path, 'thz.npy'),thz)

    # # Reactions

    # reac=np.concatenate((np.ndarray.reshape(cDOF,len(cDOF),1),r[cDOF]),axis=1)
    # np.savetxt(os.path.join(path, 'reac.txt'),reac)

    # Rz = r[cDOF[divmod(cDOF,6)[1]==2]]
    # cDOFz = cDOF[divmod(cDOF,6)[1]==2]
    # nz = (cDOFz-2)/6
    # cz = np.zeros((len(nz),3))
    # for i in range(len(nz)):
    #     cz[i,:] = NODESG[int(nz[i]),:]
    # reacz = np.concatenate((cz,Rz),axis=1)
    # reacz = np.concatenate((np.ndarray.reshape(nz,len(nz),1),np.concatenate((cz,Rz),axis=1)),axis=1)
    # np.savetxt(os.path.join(path, 'reacz.txt'),reacz)

    # # Convergence Plot
    # plt.savefig(os.path.join(path, 'Convergenceplot'))

    print('Finished :)')
    return mat_res




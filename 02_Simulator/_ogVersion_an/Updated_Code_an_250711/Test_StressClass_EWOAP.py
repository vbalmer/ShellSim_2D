import numpy as np
from Stresses_mixreinf import stress,lbd, do_itkin
import matplotlib.pyplot as plt
from math import *
import os
import time


def Ableitungsfunktion(obj,e,numout):
    """ ------------------------------ Function to get stress and derivatives-------------------------------------------
        ----------------------------------------------    INPUT: -------------------------------------------------------
        - obj: object defining function which is to be derivated. Input vector "e", output vector "s".
               Requirements for "obj":
               - Pre-instanciated with all parameters except input vector "e"
               - Possesses method "obj.out(e)" returning the output vector "s" (stresses) for given input vector
                 "e" (strains).
        - e: Input vector: [ex_klij,ey_klij,gxy_klij,gxz_klij,gyz_klij] strain state in integration point in case of
             stress calculation
        - numout: length of output vector "s" (may differ from length of input vector e)
        ----------------------------------------------- OUTPUT:---------------------------------------------------------
        - addout: additional output information, e.g. [ex,ey,gxy,e1,e3]: Layer strains
        - s: [1 x numout] output vector, e.g. [sx,sy,txy]:  Layer stresses
        - Tanmat: [numout x numout] Tangent Stiffness Matrix ds/de
    -----------------------------------------------------------------------------------------------------------------"""
    Tanmat = np.zeros((numout,numout))
    for i in range(numout):
        obj.out(e + np.eye(1,len(e),i).flatten()*1e-17j)
        for j in range(numout):
            Tanmat[j,i] = obj.s[j].imag/1e-17

    return [[obj.ex.real,obj.ey.real,obj.gxy.real,obj.e1.real,obj.e3.real],
            [obj.sx.real,obj.sy.real,obj.txy.real],Tanmat]

# -------------------------------------------------------------------------------------------------------------------- #
# Test Stress Class
# -------------------------------------------------------------------------------------------------------------------- #
tb0 = 10.6
tb1 = 5.3
tbp0 = 6
tbp1 = 5.9
dx = 12
rhox = 0.0038
Esx = 200000
Eshx = 1000
fsyx = 520
fsux = 600

Ec = 40000
Gc = 40000/(2*(1+0.2))
vc = 0.0
t = 300
fcp = 75
fct = 5.3
ec0 = 2e-3
rhoy = 0.0157
dy  = 20
Esy = 200000
Eshy = 1000
fsyy = 520
fsuy = 600


Epx = 149000
Epy = 149000
ebp1 = 0.01
sp0 = 1000
fpux = 2650-sp0
fpuy = 2650-sp0
dpx = 8.2
dpy = 8.2
rhopx = 0
rhopy = 0

MAT = [Ec, vc, fcp, fct, ec0, 0*fct/Ec, Esx, Eshx, fsyx, fsux, Esy, Eshy, fsyy, fsuy, tb0, tb1, Epx, Epy, tbp0,
       tbp1, ebp1, fpux, fpuy]
GEOM = [rhox, rhoy, dx, dy, rhopx, rhopy, dpx, dpy]

# 1 Iteration and Results
# Output:
#   - ex1: Values of epsilon_x to be iterated through
#   - sx1: Pertaining values of stress (calculated in stress class)
#   - sxdev1: Pertaining values of (d sigma_x)/(d epsilon_x)


txy_model1 = np.linspace(0,6,200)
txy_model2 = 6*np.ones_like(txy_model1)
txy_model = np.concatenate((txy_model1,txy_model2),axis=0)
sx_model1 = -3.6*np.ones_like(txy_model1)
sx_model2 = np.linspace(-3.6,-8.8,200)
sx_model = np.concatenate((sx_model1,sx_model2),axis=0)
sy_model = np.zeros_like(txy_model)

ex_model = np.zeros_like(txy_model)
ey_model = np.zeros_like(txy_model)
gxy_model = np.zeros_like(txy_model)

e1 = np.zeros_like(ex_model)
e3 = np.zeros_like(ex_model)
dsx_dex = np.zeros_like(ex_model)
dsx_dey = np.zeros_like(ex_model)
dsx_dgxy = np.zeros_like(ex_model)
dsy_dex = np.zeros_like(ex_model)
dsy_dey = np.zeros_like(ex_model)
dsy_dgxy = np.zeros_like(ex_model)
dtxy_dex = np.zeros_like(ex_model)
dtxy_dey = np.zeros_like(ex_model)
dtxy_dgxy = np.zeros_like(ex_model)

ssrx = np.zeros_like(ex_model)
sprx = np.zeros_like(ex_model)
ssry = np.zeros_like(ex_model)
srm = np.zeros_like(ex_model)
wr = np.zeros_like(ex_model)
sc3 = np.zeros_like(ex_model)
submodel = np.zeros_like(ex_model)
th = np.zeros_like(ex_model)
thr = np.zeros_like(ex_model)
thc = np.zeros_like(ex_model)
fcs = np.zeros_like(ex_model)
sig_prev = stress(0,1, MAT, GEOM, 2, 2, 1)
[a,sigma_prev,b] = Ableitungsfunktion(sig_prev, np.array([0,0,0, 0.001, 0.001]), 3)
start = time.time()

target_stress = True
contit = True
for i in range(len(ex_model)):
    print('new_i')
    print(txy_model[i])
    contit_i = contit
    itcount = 0
    noconv = False
    sig_prev.thr = pi/4*18/18
    if i == 0:
        exi = a[0]
        eyi = a[1]
        gxyi = a[2]
    elif isnan(gxy_model[i-1]):
        contit = False
        noconv = True
    else:
        exi = ex_model[i-1]
        eyi = ey_model[i - 1]
        gxyi = gxy_model[i - 1]

    while contit_i:
        itcount += 1
        sig = stress(sig_prev, 4, MAT, GEOM, 2, 2, 1)
        [epsilon,sigma,Tanmat] = Ableitungsfunktion(sig, np.array([exi, eyi, gxyi, 0.001, 0.001]), 3)
        sxi = sigma[0]
        syi = sigma[1]
        txyi = sigma[2]
        dsx_dex[i] = Tanmat[0][0]
        dsx_dey[i] = Tanmat[0][1]
        dsx_dgxy[i] = Tanmat[0][2]
        dsy_dex[i] = Tanmat[1][0]
        dsy_dey[i] = Tanmat[1][1]
        dsy_dgxy[i] = Tanmat[1][2]
        dtxy_dex[i] = Tanmat[2][0]
        dtxy_dey[i] = Tanmat[2][1]
        dtxy_dgxy[i] = Tanmat[2][2]
        if target_stress:
            dsi = np.array([[sx_model[i]-sxi],[sy_model[i]-syi],[txy_model[i]-txyi]])/10
            if np.max(abs(dsi)) < 0.001:
                print('stress found')
                contit_i = False
            else:
                if abs(np.linalg.det(Tanmat)) > 0:
                    dei = np.linalg.inv(Tanmat)@dsi
                else:
                    dei = np.array([0,0,0])
                    contit_i = False
                    noconv = True
                exi+=float(dei[0]/1)
                eyi+=float(dei[1]/1)
                gxyi+=float(dei[2]/1)
                exi = min(0.02,exi)
                eyi = min(0.02,eyi)
                gxyi = min(0.02,gxyi)
        else:
            contit_i = False
        if itcount > 500:
            contit_i = False
            noconv = True
    if noconv:
        ex_model[i] = nan
        sx_model[i] = nan
        ey_model[i] = nan
        sy_model[i] = nan
        gxy_model[i] = nan
        txy_model[i] = nan
        e1[i] = nan
        e3[i] = nan
        ssrx[i] = nan
        sprx[i] = nan
        ssry[i] = nan
        sc3[i] = nan
        srm[i] = nan
        wr[i] = nan
        submodel[i] = nan
        th[i] = nan
        thr[i] = nan
        thc[i] = nan
        fcs[i] = nan
    else:
        ex_model[i] = exi
        ey_model[i] = eyi
        gxy_model[i] = gxyi
        sig_prev = sig
        e1[i] = epsilon[3]
        e3[i] = epsilon[4]

        ssrx[i] = sig.ssx.real
        sprx[i] = sig.spx.real
        ssry[i] = sig.ssy.real
        sc3[i] = sig.sc3.real
        srm[i] = sig.sr.real
        wr[i] = e1[i] * srm[i]
        submodel[i] = sig.submodel
        th[i] = sig.th.real
        thr[i] = sig.thr.real
        thc[i] = sig.thc.real
        fcs[i] = sig.fc_soft.real

    # Check if rupture in steel or CFRP
    isfail = 0
    if ssrx[i] >= fsux:
        print('steel failure')
        isfail = 0

    if sprx[i] >= fpux:
        print('CFRP failure')
        isfail = 0

    if isfail == 1:
        ex_model = np.delete(ex_model, range(i, len(wr)), axis=0)
        ey_model = np.delete(ey_model, range(i, len(wr)), axis=0)
        gxy_model = np.delete(gxy_model, range(i, len(wr)), axis=0)
        sx_model = np.delete(sx_model, range(i, len(wr)), axis=0)
        sy_model = np.delete(sy_model, range(i, len(wr)), axis=0)
        txy_model = np.delete(txy_model, range(i, len(wr)), axis=0)
        dsx_dex = np.delete(dsx_dex,range(i,len(wr)),axis=0)
        dsx_dey = np.delete(dsx_dey,range(i,len(wr)),axis=0)
        dsx_dgxy = np.delete(dsx_dgxy,range(i,len(wr)),axis=0)
        dsy_dex = np.delete(dsy_dex,range(i,len(wr)),axis=0)
        dsy_dey = np.delete(dsy_dey,range(i,len(wr)),axis=0)
        dsy_dgxy = np.delete(dsy_dgxy,range(i,len(wr)),axis=0)
        dtxy_dex = np.delete(dtxy_dex,range(i,len(wr)),axis=0)
        dtxy_dey = np.delete(dtxy_dey,range(i,len(wr)),axis=0)
        dtxy_dgxy = np.delete(dtxy_dgxy,range(i,len(wr)),axis=0)
        e1 = np.delete(e1,range(i,len(wr)),axis=0)
        e3 = np.delete(e3,range(i,len(wr)),axis=0)
        ssrx = np.delete(ssrx,range(i,len(wr)),axis=0)
        sprx = np.delete(sprx,range(i,len(wr)),axis=0)
        ssry = np.delete(ssry,range(i,len(wr)),axis=0)
        sc3 = np.delete(sc3,range(i,len(wr)),axis=0)
        srm = np.delete(srm,range(i,len(wr)),axis=0)
        th = np.delete(th, range(i, len(wr)), axis=0)
        thr = np.delete(thr, range(i, len(wr)), axis=0)
        thc = np.delete(thc, range(i, len(wr)), axis=0)
        fcs = np.delete(fcs, range(i, len(wr)), axis=0)
        submodel = np.delete(submodel, range(i, len(wr)), axis=0)
        wr = np.delete(wr,range(i,len(wr)),axis=0)
        break

    # if sig.cms_klij == 4:
    #     if i == ind_01:
    #         xcr_01 = [i.real for i in sig.xcr]
    #         es_all_01 = [i.real for i in sig.es_all]
    #         ss_all_01 = sig.ss_all
    #         ds_all_01 = sig.ds_all
    #         tbs_all_01 = sig.tbs_all
    #         ep_all_01 = [i.real for i in sig.ep_all]
    #         sp_all_01 = sig.sp_all
    #         dp_all_01 = sig.dp_all
    #         tbp_all_01 = sig.tbp_all
    #     elif i == ind_04:
    #         es_all_04 = [i.real for i in sig.es_all]
    #         ss_all_04 = sig.ss_all
    #         ds_all_04 = sig.ds_all
    #         tbs_all_04 = sig.tbs_all
    #         ep_all_04 = [i.real for i in sig.ep_all]
    #         sp_all_04 = sig.sp_all
    #         dp_all_04 = sig.dp_all
    #         tbp_all_04 = sig.tbp_all
end = time.time()
print('sr = ')
print(sig.sr)
print('Time Spent in calculation:')
print(end-start)
# 2 Control Calculations
# Output:
#   - sx2: stress values calculated by integrating sxdev1
#   - ex2: strain values between values of ex1: ex2[i]=(ex1[i]+ex1[i+1])/2
#   - sxdev2: Tangent stiffness derived via sxdev2[i] = (sx1[i+1]-sx1[i])/(ex1[i+1] - ex1[i])
#       --> "Less fancy" way to calculate derivative, but should be "equal" to sxdev1!
sx2 = np.zeros_like(ex_model)
sy2 = np.zeros_like(ex_model)
txy2 = np.zeros_like(ex_model)

for i in range(len(sx2)):
    if i == 0:
        sx2[i] = sx_model[0]
        sy2[i] = sy_model[0]
        txy2[i] = txy_model[0]
    else:
        sx2[i] = sx2[i - 1] + (dsx_dex[i - 1] + dsx_dex[i]) / 2 * (ex_model[i] - ex_model[i - 1]) + (dsx_dey[i - 1] + dsx_dey[i]) / 2 * (ey_model[i] - ey_model[i - 1]) + (dsx_dgxy[i - 1] + dsx_dgxy[i]) / 2 * (gxy_model[i] - gxy_model[i - 1])
        sy2[i] = sy2[i - 1] + (dsy_dex[i - 1] + dsy_dex[i]) / 2 * (ex_model[i] - ex_model[i - 1]) + (dsy_dey[i - 1] + dsy_dey[i]) / 2 * (ey_model[i] - ey_model[i - 1]) + (dsy_dgxy[i - 1] + dsy_dgxy[i]) / 2 * (gxy_model[i] - gxy_model[i - 1])
        txy2[i] = txy2[i - 1] + (dtxy_dex[i - 1] + dtxy_dex[i]) / 2 * (ex_model[i] - ex_model[i - 1]) + (dtxy_dey[i - 1] + dtxy_dey[i]) / 2 * (ey_model[i] - ey_model[i - 1]) + (dtxy_dgxy[i - 1] + dtxy_dgxy[i]) / 2 * (gxy_model[i] - gxy_model[i - 1])

ex2 = np.zeros(len(ex_model) - 1)
sxdev2 = np.zeros_like(ex2)
for i in range(len(ex2)):
    ex2[i] = (ex_model[i] + ex_model[i + 1]) / 2
    sxdev2[i] = (sx_model[i + 1] - sx_model[i]) / (ex_model[i + 1] - ex_model[i])

#-----------------------------------------------------------------------------------------------------------------------
# 3 Plot
#-----------------------------------------------------------------------------------------------------------------------
# 3.1 General Plot Paramters
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'cm'

# 3.2 Manipulate epxilon to be in microstrain
ex_model = ex_model*1000
ey_model = ey_model * 1000
gxy_model = gxy_model * 1000
e1 = e1*1000
e3 = e3*1000

# 3.3 First Plot: sigma_x
fig1, axs = plt.subplots(3, 3)
fig1.set_figheight(9)
fig1.set_figwidth(10)

axs[0, 0].plot(ex_model, sx_model, 'k')
axs[0, 0].set_title('$\epsilon_x-\sigma_x$')
axs[0, 0].set(ylabel="$\sigma_{x,Layer}$ [MPa]")

axs[1, 0].plot(ex_model, dsx_dex, 'b')
axs[1, 0].set(xlabel="$\epsilon$$_{x}$ [m$\epsilon$]")
axs[1, 0].set(ylabel="d $\sigma_{x}$ / d $\epsilon_{x}$ [MPa]")

axs[0, 1].plot(ey_model, sx_model, 'k')
axs[0, 1].set_title('$\epsilon_y-\sigma_x$')
axs[0, 1].set(ylabel="$\sigma_{x,Layer}$ [MPa]")

axs[1, 1].plot(ey_model, dsx_dey, 'b')
axs[1, 1].set(xlabel="$\epsilon$$_{y}$ [m$\epsilon$]")
axs[1, 1].set(ylabel="d $\sigma_{x}$ / d $\epsilon_{y}$ [MPa]")

axs[0, 2].plot(gxy_model, sx_model, 'k')
axs[0, 2].set_title('$\gamma_{xy}-\sigma_x$')
axs[0, 2].set(ylabel="$\sigma_{x,Layer}$ [MPa]")

axs[1, 2].plot(gxy_model, dsx_dgxy, 'b')
axs[1, 2].set(xlabel="$\gamma$$_{xy}$ [m$\epsilon$]")
axs[1, 2].set(ylabel="d $\sigma_{x}$ / d $\gamma_{xy}$ [MPa]")

axs[2, 0].plot(ex_model, ex_model, 'k')
axs[2, 0].plot(ex_model, ey_model, 'r')
axs[2, 0].plot(ex_model, gxy_model, 'b')
axs[2, 0].plot(ex_model, e1, 'm')
axs[2, 0].plot(ex_model, e3, 'c')
axs[2, 0].legend(['$\epsilon_x$','$\epsilon_y$','$\gamma_{xy}$','$\epsilon_1$','$\epsilon_3$'])
axs[2, 0].set_title('Strains')
axs[2, 0].set(xlabel="$\epsilon_x$ [m$\epsilon$]")
axs[2, 0].set(ylabel="$\epsilon$")

axs[2, 1].set_title('Principal Dir')
axs[2 ,1].plot(ex_model, thr * 180 / pi, 'k')
axs[2 ,1].plot(ex_model, thc * 180 / pi, 'r')
axs[2, 1].legend(['$\\theta_r$','$\\theta_c$'])
axs[2, 1].set(xlabel="$\epsilon_x$ [m$\epsilon$]")
axs[2, 1].set(ylabel="$\\theta$ [°]")

axs[2, 2].set_title('Submodel')
axs[2 ,2].plot(ex_model, submodel, 'k')
axs[2, 2].set(xlabel="$\epsilon_x$ [m$\epsilon$]")
axs[2, 2].set(ylabel="Model")

fig1.tight_layout(pad=2.0)

# 3.3 Second Plot: sigma_y
fig2, axs2 = plt.subplots(3, 3)
fig2.set_figheight(9)
fig2.set_figwidth(10)

axs2[0, 0].plot(ex_model, sy_model, 'k')
axs2[0, 0].set_title('$\epsilon_x-\sigma_y$')
axs2[0, 0].set(ylabel="$\sigma_{y,Layer}$ [MPa]")

axs2[1, 0].plot(ex_model, dsy_dex, 'b')
axs2[1, 0].set(xlabel="$\epsilon$$_{x}$ [m$\epsilon$]")
axs2[1, 0].set(ylabel="d $\sigma_{y}$ / d $\epsilon_{x}$ [MPa]")

axs2[0, 1].plot(ey_model, sy_model, 'k')
axs2[0, 1].set_title('$\epsilon_y-\sigma_y$')
axs2[0, 1].set(ylabel="$\sigma_{y,Layer}$ [MPa]")

axs2[1, 1].plot(ey_model, dsy_dey, 'b')
axs2[1, 1].set(xlabel="$\epsilon$$_{y}$ [m$\epsilon$]")
axs2[1, 1].set(ylabel="d $\sigma_{y}$ / d $\epsilon_{y}$ [MPa]")

axs2[0, 2].plot(gxy_model, sy_model, 'k')
axs2[0, 2].set_title('$\gamma_{xy}-\sigma_y$')
axs2[0, 2].set(ylabel="$\sigma_{y,Layer}$ [MPa]")

axs2[1, 2].plot(gxy_model, dsy_dgxy, 'b')
axs2[1, 2].set(xlabel="$\gamma$$_{xy}$ [m$\epsilon$]")
axs2[1, 2].set(ylabel="d $\sigma_{y}$ / d $\gamma_{xy}$ [MPa]")

axs2[2, 0].plot(ey_model, ex_model, 'k')
axs2[2, 0].plot(ey_model, ey_model, 'r')
axs2[2, 0].plot(ey_model, gxy_model, 'b')
axs2[2, 0].plot(ey_model, e1, 'm')
axs2[2, 0].plot(ey_model, e3, 'c')
axs2[2, 0].legend(['$\epsilon_x$','$\epsilon_y$','$\gamma_{xy}$','$\epsilon_1$','$\epsilon_3$'])
axs2[2, 0].set_title('Strains')
axs2[2, 0].set(xlabel="$\epsilon_y$ [m$\epsilon$]")
axs2[2, 0].set(ylabel="$\epsilon$")

axs2[2, 1].set_title('Principal Dir')
axs2[2 ,1].plot(ey_model, thr * 180 / pi, 'k')
axs2[2 ,1].plot(ey_model, thc * 180 / pi, 'r')
axs2[2, 1].legend(['$\\theta_r$','$\\theta_c$'])
axs2[2, 1].set(xlabel="$\epsilon_y$ [m$\epsilon$]")
axs2[2, 1].set(ylabel="$\\theta$ [°]")

axs2[2, 2].set_title('Submodel')
axs2[2 ,2].plot(ey_model, submodel, 'k')
axs2[2, 2].set(xlabel="$\epsilon_y$ [m$\epsilon$]")
axs2[2, 2].set(ylabel="Model")

fig2.tight_layout(pad=2.0)

# 3.3 Third Plot: tau_xy
fig3, axs3 = plt.subplots(3, 3)
fig3.set_figheight(9)
fig3.set_figwidth(10)

axs3[0, 0].plot(ex_model, txy_model, 'k')
axs3[0, 0].set_title('$\epsilon_x-\\tau_{xy}$')
axs3[0, 0].set(ylabel="$\\tau_{xy,Layer}$ [MPa]")

axs3[1, 0].plot(ex_model, dtxy_dex, 'b')
axs3[1, 0].set(xlabel="$\epsilon$$_{x}$ [m$\epsilon$]")
axs3[1, 0].set(ylabel="d $\\tau_{xy}$ / d $\epsilon_{x}$ [MPa]")

axs3[0, 1].plot(ey_model, txy_model, 'k')
axs3[0, 1].set_title('$\epsilon_y-\\tau_{xy}$')
axs3[0, 1].set(ylabel="$\\tau_{xy,Layer}$ [MPa]")

axs3[1, 1].plot(ey_model, dtxy_dey, 'b')
axs3[1, 1].set(xlabel="$\epsilon$$_{y}$ [m$\epsilon$]")
axs3[1, 1].set(ylabel="d $\\tau_{xy}$ / d $\epsilon_{y}$ [MPa]")

axs3[0, 2].plot(gxy_model, txy_model, 'k')
axs3[0, 2].set_title('$\gamma_{xy}-\\tau_{xy}$')
axs3[0, 2].set(ylabel="$\\tau_{xy,Layer}$ [MPa]")

axs3[1, 2].plot(gxy_model, dtxy_dgxy, 'b')
axs3[1, 2].set(xlabel="$\gamma$$_{xy}$ [m$\epsilon$]")
axs3[1, 2].set(ylabel="d $\\tau_{xy}$ / d $\gamma_{xy}$ [MPa]")

axs3[2, 0].plot(gxy_model, ex_model, 'k')
axs3[2, 0].plot(gxy_model, ey_model, 'r')
axs3[2, 0].plot(gxy_model, gxy_model, 'b')
axs3[2, 0].plot(gxy_model, e1, 'm')
axs3[2, 0].plot(gxy_model, e3, 'c')
axs3[2, 0].legend(['$\epsilon_x$','$\epsilon_y$','$\gamma_{xy}$','$\epsilon_1$','$\epsilon_3$'])
axs3[2, 0].set_title('Strains')
axs3[2, 0].set(xlabel="$\gamma_{xy}$ [m$\epsilon$]")
axs3[2, 0].set(ylabel="$\epsilon$")

axs3[2, 1].set_title('Principal Dir')
axs3[2 ,1].plot(gxy_model, thr * 180 / pi, 'k')
axs3[2 ,1].plot(gxy_model, thc * 180 / pi, 'r')
axs3[2, 1].legend(['$\\theta_r$','$\\theta_c$'])
axs3[2, 1].set(xlabel="$\gamma_{xy}$ [m$\epsilon$]")
axs3[2, 1].set(ylabel="$\\theta$ [°]")

axs3[2, 2].set_title('Submodel')
axs3[2 ,2].plot(gxy_model, submodel, 'k')
axs3[2, 2].set(xlabel="$\gamma_{xy}$ [m$\epsilon$]")
axs3[2, 2].set(ylabel="Model")

fig3.tight_layout(pad=2.0)


# 3.4 Fourth Plot: Material Stresses
fig4, axs4 = plt.subplots(3, 3)
fig4.set_figheight(9)
fig4.set_figwidth(10)

axs4[0, 0].plot(e3, sc3,'k')
axs4[0, 0].set_title('$\epsilon_3-\sigma_{c3}$')
axs4[0, 0].set(ylabel="$\sigma_{c3,Layer}$ [MPa]")
axs4[0, 0].set(xlabel="$\epsilon_{3,Layer}$ [m$\epsilon$]")

axs4[0, 1].plot(gxy_model, sc3,'k')
axs4[0, 1].set_title('$\\gamma_{xy}-\sigma_{c3}$')
axs4[0, 1].set(ylabel="$\sigma_{c3,Layer}$ [MPa]")
axs4[0, 1].set(xlabel="$\\gamma_{xy,Layer}$ [MPa]")

axs4[0, 2].plot(e1, fcs,'k')
axs4[0, 2].set_title('$\epsilon_1-f_{c,soft}$')
axs4[0, 2].set(ylabel="$f_{c,soft}$ [MPa]")
axs4[0, 2].set(xlabel="$\epsilon_{1,Layer}$ [MPa]")

axs4[1, 0].plot(ex_model, ssrx,'k')
axs4[1, 0].set_title('$\epsilon_x-\sigma_{sxr}$')
axs4[1, 0].set(ylabel="$\sigma_{sxr,Layer}$ [MPa]")
axs4[1, 0].set(xlabel="$\epsilon_{x,Layer}$ [MPa]")

axs4[1, 1].plot(ey_model, ssrx,'k')
axs4[1, 1].set_title('$\epsilon_y-\sigma_{sxr}$')
axs4[1, 1].set(ylabel="$\sigma_{sxr,Layer}$ [MPa]")
axs4[1, 1].set(xlabel="$\epsilon_{y,Layer}$ [MPa]")

axs4[1, 2].plot(gxy_model, ssrx,'k')
axs4[1, 2].set_title('$\\gamma_{xy}-\sigma_{sxr}$')
axs4[1, 2].set(ylabel="$\sigma_{sxr,Layer}$ [MPa]")
axs4[1, 2].set(xlabel="$\\gamma_{xy,Layer}$ [MPa]")

axs4[2, 0].plot(ex_model, ssry,'k')
axs4[2, 0].set_title('$\epsilon_x-\sigma_{syr}$')
axs4[2, 0].set(ylabel="$\sigma_{syr,Layer}$ [MPa]")
axs4[2, 0].set(xlabel="$\epsilon_{x,Layer}$ [MPa]")

axs4[2, 1].plot(ey_model, ssry,'k')
axs4[2, 1].set_title('$\epsilon_y-\sigma_{syr}$')
axs4[2, 1].set(ylabel="$\sigma_{syr,Layer}$ [MPa]")
axs4[2, 1].set(xlabel="$\epsilon_{y,Layer}$ [MPa]")

axs4[2, 2].plot(gxy_model, ssry,'k')
axs4[2, 2].set_title('$\\gamma_{xy}-\sigma_{syr}$')
axs4[2, 2].set(ylabel="$\sigma_{syr,Layer}$ [MPa]")
axs4[2, 2].set(xlabel="$\\gamma_{xy,Layer}$ [MPa]")

fig4.tight_layout(pad=2.0)

if sig.cms_klij == 4:
    plt.figure(figsize=(6,6))
    plt.subplot(4,1,1)
    plt.plot(xcr_01,[i*1000 for i in es_all_01])
    plt.plot(xcr_01,[i*1000 for i in ep_all_01])
    plt.ylabel("$\epsilon$$_{x} [m\epsilon]$")

    plt.subplot(4,1,2)
    plt.plot(xcr_01,ss_all_01)
    plt.plot(xcr_01,sp_all_01)
    plt.ylabel("$\sigma$$_{x} [MPa]$")

    plt.subplot(4,1,3)
    plt.plot(xcr_01,tbs_all_01)
    plt.plot(xcr_01,tbp_all_01)
    plt.ylabel("$\u03C4$$_{b} [MPa]$")

    plt.subplot(4,1,4)
    plt.plot(xcr_01,ds_all_01)
    plt.plot(xcr_01,dp_all_01)
    plt.ylabel("$\delta$$_{x} [mm]$")
    plt.xlabel("$x [mm]$")
    plt.subplots_adjust(hspace=1)


# ----------------------------------------------------------------------------------------------------------------------
# Save results
# ----------------------------------------------------------------------------------------------------------------------
import shutil
# path = r"C:\Users\naesboma\00_an\04_UHBB\02_Materialmodelle\01_TCM"

path = r"C:\Users\naesboma\00_an\04_UHBB\06_Papers\06_Numerical_Modelling\02_Modell\01_LUSET_abe\Calculations"

# 3.2 Join With Load Step Number
a = [path, r"/", str(do_itkin)]
path = ''.join(a)

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

path = r'C:\Users\naesboma\00_an\00_Admin\EWOAP'

np.savetxt(os.path.join(path, 'wr_TC.txt'),wr)
np.savetxt(os.path.join(path, 'ssrx.txt'),ssrx)
np.savetxt(os.path.join(path, 'ssry.txt'),ssry)

np.savetxt(os.path.join(path, 'gxy.txt'), gxy_model)
np.savetxt(os.path.join(path, 'txy_model.txt'), txy_model)
np.savetxt(os.path.join(path, 'sx_model.txt'), sx_model)

plt.show()
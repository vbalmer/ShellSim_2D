import numpy as np
from Stresses_mixreinf import stress,lbd
import matplotlib.pyplot as plt
from math import *
import os


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
dtest = 12
# 0 Material Parameter Assignment
if dtest == 8:
    tb0 = 5.9
    tb1 = tb0/2.36
    tbp0 = 4.6
    tbp1 = tbp0-.1
    dx = 8
    rhox = dx**2*pi/150**2
    Esx = 193000
    Eshx = 1580
    fsyx = 537 + 0.3e-3 * Esx
    fsux = 599 + 0.3e-3 * Esx
elif dtest == 10:
    tb0 = 7.7
    tb1 = 3.0
    tbp0 = 4.6
    tbp1 = 4.5
    dx = 10
    rhox = dx**2*pi/150**2
    Esx = 198000
    Eshx = 1340
    fsyx = 521 + 0.3e-3 * Esx
    fsux = 603 + 0.3e-3 * Esx
elif dtest == 12:
    tb0 = 3.8
    tb1 = 2.2
    tbp0 = 4.6
    tbp1 = 4.5
    dx = 12
    rhox = dx**2*pi/150**2
    Esx = 195000
    Eshx = 1150
    fsyx = 521 + 0.3e-3 * Esx
    fsux = 581 + 0.3e-3 * Esx
elif dtest == 10.2:
    tb0 = 4.6
    tb1 = 2.3
    tbp0 = 4.6
    tbp1 = 4.5
    dx = 10.2
    rhox = dx**2*pi/150**2
    Esx = 194000
    Eshx = 1280
    fsyx = 526 + 0.15e-3 * Esx
    fsux = 587 + 0.15e-3 * Esx

Ec = 40000
Gc = 40000/(2*(1+0.2))
vc = 0.0
t = 150
fcp = 100
fct = 3
ec0 = 2.5e-3
rhoy = 0.008*0
dy  = 10
Esy = 193000
Eshy = 1580
fsyy = 537+0.15e-3*Esx
fsuy = 599+0.15e-3*Esx


Epx = 149000
Epy = 149000
ebp1 = 0.01
sp0 = 1000
fpux = 2650-sp0
fpuy = 2650-sp0
dpx = 8.2
dpy = 8.2
# xi = 8
# omega = (4*4**2*pi*fsux+2*4.1**2*pi*fpux)/150**2
# rhopx = omega/(xi*fsux+fpux)
# rhox = (omega-rhopx*fpux)/fsux
# print(rhopx)
# print(rhox)
rhopx = 2*dpx**2*pi/4/150**2
rhopy = 0

MAT = [Ec, vc, fcp, fct, ec0, 0*fct/Ec, Esx, Eshx, fsyx, fsux, Esy, Eshy, fsyy, fsuy, tb0, tb1, Epx, Epy, tbp0,
       tbp1, ebp1, fpux, fpuy]
GEOM = [rhox, rhoy, dx, dy, rhopx, rhopy, dpx, dpy]

# 1 Iteration and Results
# Output:
#   - ex1: Values of epsilon_x to be iterated through
#   - sx1: Pertaining values of stress (calculated in stress class)
#   - sxdev1: Pertaining values of (d sigma_x)/(d epsilon_x)
# exmin = .01e-3
exmin = 0.00075*0.1
exmax = 0.00075*0.2
# exmax = 0.03e-3
# exmin = 5e-3
# exmax = 5e-3
steps = 500
ex1 = np.linspace(exmin, exmax, steps)
# ind_01 = np.where(np.abs(ex1-0.0015) == np.min(np.abs(ex1-0.0015)))
# print(ind_01[0])
# ind_01 = int(ind_01[0])
# ind_04 = np.where(np.abs(ex1-0.004) == np.min(np.abs(ex1-0.004)))
# ind_04 = int(ind_04[0])

eymin = .1e-3
eymax = .2e-3
ey1 = np.linspace(eymin,eymax,steps)

gxymin = 2e-3
gxymax = 3e-3
gxy1 = np.linspace(gxymin,gxymax,steps)

e1 = np.zeros_like(ex1)
e3 = np.zeros_like(ex1)
sx1 = np.zeros_like(ex1)
sy1 = np.zeros_like(ex1)
txy1 = np.zeros_like(ex1)
dsx_dex = np.zeros_like(ex1)
dsx_dey = np.zeros_like(ex1)
dsx_dgxy = np.zeros_like(ex1)
dsy_dex = np.zeros_like(ex1)
dsy_dey = np.zeros_like(ex1)
dsy_dgxy = np.zeros_like(ex1)
dtxy_dex = np.zeros_like(ex1)
dtxy_dey = np.zeros_like(ex1)
dtxy_dgxy = np.zeros_like(ex1)
Q_TC = np.zeros_like(ex1)

ssrx = np.zeros_like(ex1)
sprx = np.zeros_like(ex1)
ssry = np.zeros_like(ex1)
srm = np.zeros_like(ex1)
wr = np.zeros_like(ex1)
sc3 = np.zeros_like(ex1)
submodel = np.zeros_like(ex1)
th = np.zeros_like(ex1)
fcs = np.zeros_like(ex1)
sig_prev = stress(0,1, MAT, GEOM, 2,2,1)
[a,sigma_prev,b] = Ableitungsfunktion(sig_prev,np.array([ex1[0], ey1[0], gxy1[0], 0.001, 0.001]),3)
# print(sig_prev.thr)
for i in range(len(ex1)):
    sig = stress(sig_prev,4, MAT, GEOM, 2,2,1)
    [epsilon,sigma,Tanmat] = Ableitungsfunktion(sig,np.array([ex1[i], ey1[i], gxy1[i], 0.001, 0.001]),3)
    sig_prev = sig
    sx1[i] = sigma[0]
    Q_TC[i] = sigma[0]*150**2+sp0*4.1**2*pi*2-.3e-3*Esx*4*dx**2*pi/4
    sy1[i] = sigma[1]
    txy1[i] = sigma[2]
    dsx_dex[i] = Tanmat[0][0]
    dsx_dey[i] = Tanmat[0][1]
    dsx_dgxy[i] = Tanmat[0][2]
    dsy_dex[i] = Tanmat[1][0]
    dsy_dey[i] = Tanmat[1][1]
    dsy_dgxy[i] = Tanmat[1][2]
    dtxy_dex[i] = Tanmat[2][0]
    dtxy_dey[i] = Tanmat[2][1]
    dtxy_dgxy[i] = Tanmat[2][2]
    e1[i] = epsilon[3]
    e3[i] = epsilon[4]

    ssrx[i] = sig.ssx.real
    sprx[i] = sig.spx.real
    ssry[i] = sig.ssy.real
    sc3[i] = sig.sc3.real
    srm[i] = sig.sr.real
    wr[i] = (ex1[i].real-lbd*fct/Ec/2)*srm[i]
    submodel[i] = sig.submodel
    th[i] = sig.thc.real
    fcs[i] = sig.fc_soft.real

    # Check if rupture in steel or CFRP
    isfail = 0
    if ssrx[i] >= fsux:
        print('steel failure')
        isfail = 1

    if sprx[i] >= fpux:
        print('CFRP failure')
        isfail = 1

    if isfail == 1:
        ex1 = np.delete(ex1, range(i, len(wr)), axis=0)
        ey1 = np.delete(ey1, range(i, len(wr)), axis=0)
        gxy1 = np.delete(gxy1, range(i, len(wr)), axis=0)
        sx1 = np.delete(sx1,range(i,len(wr)),axis=0)
        Q_TC = np.delete(Q_TC,range(i,len(wr)),axis=0)
        sy1 = np.delete(sy1,range(i,len(wr)),axis=0)
        txy1 = np.delete(txy1,range(i,len(wr)),axis=0)
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

print('sr = ')
print(sig.sr)
# 2 Control Calculations
# Output:
#   - sx2: stress values calculated by integrating sxdev1
#   - ex2: strain values between values of ex1: ex2[i]=(ex1[i]+ex1[i+1])/2
#   - sxdev2: Tangent stiffness derived via sxdev2[i] = (sx1[i+1]-sx1[i])/(ex1[i+1] - ex1[i])
#       --> "Less fancy" way to calculate derivative, but should be "equal" to sxdev1!
sx2 = np.zeros_like(ex1)
sy2 = np.zeros_like(ex1)
txy2 = np.zeros_like(ex1)
for i in range(len(sx2)):
    if i == 0:
        sx2[i] = sx1[0]
        sy2[i] = sy1[0]
        txy2[i] = txy1[0]
    else:
        sx2[i] = sx2[i - 1] + (dsx_dex[i - 1] + dsx_dex[i]) / 2 * (ex1[i] - ex1[i - 1]) + (dsx_dey[i - 1] + dsx_dey[i]) / 2 * (ey1[i] - ey1[i - 1]) + (dsx_dgxy[i - 1] + dsx_dgxy[i]) / 2 * (gxy1[i] - gxy1[i - 1])
        sy2[i] = sy2[i - 1] + (dsy_dex[i - 1] + dsy_dex[i]) / 2 * (ex1[i] - ex1[i - 1]) + (dsy_dey[i - 1] + dsy_dey[i]) / 2 * (ey1[i] - ey1[i - 1]) + (dsy_dgxy[i - 1] + dsy_dgxy[i]) / 2 * (gxy1[i] - gxy1[i - 1])
        txy2[i] = txy2[i - 1] + (dtxy_dex[i - 1] + dtxy_dex[i]) / 2 * (ex1[i] - ex1[i - 1]) + (dtxy_dey[i - 1] + dtxy_dey[i]) / 2 * (ey1[i] - ey1[i - 1]) + (dtxy_dgxy[i - 1] + dtxy_dgxy[i]) / 2 * (gxy1[i] - gxy1[i - 1])

ex2 = np.zeros(len(ex1)-1)
sxdev2 = np.zeros_like(ex2)
for i in range(len(ex2)):
    ex2[i] = (ex1[i]+ex1[i+1])/2
    sxdev2[i] = (sx1[i+1]-sx1[i])/(ex1[i+1] - ex1[i])

#-----------------------------------------------------------------------------------------------------------------------
# 3 Plot
#-----------------------------------------------------------------------------------------------------------------------
# 3.1 General Plot Paramters
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'cm'

# 3.2 Manipulate epxilon to be in microstrain
ex1 = ex1*1000
ey1 = ey1*1000
gxy1 = gxy1*1000
e1 = e1*1000
e3 = e3*1000

# 3.3 First Plot: sigma_x
fig1, axs = plt.subplots(3, 3)
fig1.set_figheight(9)
fig1.set_figwidth(10)

axs[0, 0].plot(ex1, sx1,'k')
axs[0, 0].plot(ex1, sx2,'r', linestyle='dashed')
axs[0, 0].set_title('$\epsilon_x-\sigma_x$')
axs[0, 0].set(ylabel="$\sigma_{x,Layer}$ [MPa]")

axs[1, 0].plot(ex1, dsx_dex, 'b')
axs[1, 0].set(xlabel="$\epsilon$$_{x}$ [m$\epsilon$]")
axs[1, 0].set(ylabel="d $\sigma_{x}$ / d $\epsilon_{x}$ [MPa]")

axs[0, 1].plot(ey1, sx1,'k')
axs[0, 1].plot(ey1, sx2,'r', linestyle='dashed')
axs[0, 1].set_title('$\epsilon_y-\sigma_x$')
axs[0, 1].set(ylabel="$\sigma_{x,Layer}$ [MPa]")

axs[1, 1].plot(ey1, dsx_dey, 'b')
axs[1, 1].set(xlabel="$\epsilon$$_{y}$ [m$\epsilon$]")
axs[1, 1].set(ylabel="d $\sigma_{x}$ / d $\epsilon_{y}$ [MPa]")

axs[0, 2].plot(gxy1, sx1,'k')
axs[0, 2].plot(gxy1, sx2,'r', linestyle='dashed')
axs[0, 2].set_title('$\gamma_{xy}-\sigma_x$')
axs[0, 2].set(ylabel="$\sigma_{x,Layer}$ [MPa]")

axs[1, 2].plot(gxy1, dsx_dgxy, 'b')
axs[1, 2].set(xlabel="$\gamma$$_{xy}$ [m$\epsilon$]")
axs[1, 2].set(ylabel="d $\sigma_{x}$ / d $\gamma_{xy}$ [MPa]")

axs[2, 0].plot(ex1, ex1,'k')
axs[2, 0].plot(ex1, ey1,'r')
axs[2, 0].plot(ex1, gxy1,'b')
axs[2, 0].plot(ex1, e1,'m')
axs[2, 0].plot(ex1, e3,'c')
axs[2, 0].legend(['$\epsilon_x$','$\epsilon_y$','$\gamma_{xy}$','$\epsilon_1$','$\epsilon_3$'])
axs[2, 0].set_title('Strains')
axs[2, 0].set(xlabel="$\epsilon_x$ [m$\epsilon$]")
axs[2, 0].set(ylabel="$\epsilon$")

axs[2, 1].set_title('Principal Dir')
axs[2 ,1].plot(ex1, th*180/pi, 'k')
axs[2, 1].set(xlabel="$\epsilon_x$ [m$\epsilon$]")
axs[2, 1].set(ylabel="$\\theta$ [°]")

axs[2, 2].set_title('Submodel')
axs[2 ,2].plot(ex1, submodel, 'k')
axs[2, 2].set(xlabel="$\epsilon_x$ [m$\epsilon$]")
axs[2, 2].set(ylabel="Model")

fig1.tight_layout(pad=2.0)

# 3.3 Second Plot: sigma_y
fig2, axs2 = plt.subplots(3, 3)
fig2.set_figheight(9)
fig2.set_figwidth(10)

axs2[0, 0].plot(ex1, sy1,'k')
axs2[0, 0].plot(ex1, sy2,'r', linestyle='dashed')
axs2[0, 0].set_title('$\epsilon_x-\sigma_y$')
axs2[0, 0].set(ylabel="$\sigma_{y,Layer}$ [MPa]")

axs2[1, 0].plot(ex1, dsy_dex, 'b')
axs2[1, 0].set(xlabel="$\epsilon$$_{x}$ [m$\epsilon$]")
axs2[1, 0].set(ylabel="d $\sigma_{y}$ / d $\epsilon_{x}$ [MPa]")

axs2[0, 1].plot(ey1, sy1,'k')
axs2[0, 1].plot(ey1, sy2,'r', linestyle='dashed')
axs2[0, 1].set_title('$\epsilon_y-\sigma_y$')
axs2[0, 1].set(ylabel="$\sigma_{y,Layer}$ [MPa]")

axs2[1, 1].plot(ey1, dsy_dey, 'b')
axs2[1, 1].set(xlabel="$\epsilon$$_{y}$ [m$\epsilon$]")
axs2[1, 1].set(ylabel="d $\sigma_{y}$ / d $\epsilon_{y}$ [MPa]")

axs2[0, 2].plot(gxy1, sy1,'k')
axs2[0, 2].plot(gxy1, sy2,'r', linestyle='dashed')
axs2[0, 2].set_title('$\gamma_{xy}-\sigma_y$')
axs2[0, 2].set(ylabel="$\sigma_{y,Layer}$ [MPa]")

axs2[1, 2].plot(gxy1, dsy_dgxy, 'b')
axs2[1, 2].set(xlabel="$\gamma$$_{xy}$ [m$\epsilon$]")
axs2[1, 2].set(ylabel="d $\sigma_{y}$ / d $\gamma_{xy}$ [MPa]")

axs2[2, 0].plot(ey1, ex1,'k')
axs2[2, 0].plot(ey1, ey1,'r')
axs2[2, 0].plot(ey1, gxy1,'b')
axs2[2, 0].plot(ey1, e1,'m')
axs2[2, 0].plot(ey1, e3,'c')
axs2[2, 0].legend(['$\epsilon_x$','$\epsilon_y$','$\gamma_{xy}$','$\epsilon_1$','$\epsilon_3$'])
axs2[2, 0].set_title('Strains')
axs2[2, 0].set(xlabel="$\epsilon_y$ [m$\epsilon$]")
axs2[2, 0].set(ylabel="$\epsilon$")

axs2[2, 1].set_title('Principal Dir')
axs2[2 ,1].plot(ey1, th*180/pi, 'k')
axs2[2, 1].set(xlabel="$\epsilon_y$ [m$\epsilon$]")
axs2[2, 1].set(ylabel="$\\theta$ [°]")

axs2[2, 2].set_title('Submodel')
axs2[2 ,2].plot(ey1, submodel, 'k')
axs2[2, 2].set(xlabel="$\epsilon_y$ [m$\epsilon$]")
axs2[2, 2].set(ylabel="Model")

fig2.tight_layout(pad=2.0)

# 3.3 Third Plot: tau_xy
fig3, axs3 = plt.subplots(3, 3)
fig3.set_figheight(9)
fig3.set_figwidth(10)

axs3[0, 0].plot(ex1, txy1,'k')
axs3[0, 0].plot(ex1, txy2,'r', linestyle='dashed')
axs3[0, 0].set_title('$\epsilon_x-\\tau_{xy}$')
axs3[0, 0].set(ylabel="$\\tau_{xy,Layer}$ [MPa]")

axs3[1, 0].plot(ex1, dtxy_dex, 'b')
axs3[1, 0].set(xlabel="$\epsilon$$_{x}$ [m$\epsilon$]")
axs3[1, 0].set(ylabel="d $\\tau_{xy}$ / d $\epsilon_{x}$ [MPa]")

axs3[0, 1].plot(ey1, txy1,'k')
axs3[0, 1].plot(ey1, txy2,'r', linestyle='dashed')
axs3[0, 1].set_title('$\epsilon_y-\\tau_{xy}$')
axs3[0, 1].set(ylabel="$\\tau_{xy,Layer}$ [MPa]")

axs3[1, 1].plot(ey1, dtxy_dey, 'b')
axs3[1, 1].set(xlabel="$\epsilon$$_{y}$ [m$\epsilon$]")
axs3[1, 1].set(ylabel="d $\\tau_{xy}$ / d $\epsilon_{y}$ [MPa]")

axs3[0, 2].plot(gxy1, txy1,'k')
axs3[0, 2].plot(gxy1, txy2,'r', linestyle='dashed')
axs3[0, 2].set_title('$\gamma_{xy}-\\tau_{xy}$')
axs3[0, 2].set(ylabel="$\\tau_{xy,Layer}$ [MPa]")

axs3[1, 2].plot(gxy1, dtxy_dgxy, 'b')
axs3[1, 2].set(xlabel="$\gamma$$_{xy}$ [m$\epsilon$]")
axs3[1, 2].set(ylabel="d $\\tau_{xy}$ / d $\gamma_{xy}$ [MPa]")

axs3[2, 0].plot(gxy1, ex1,'k')
axs3[2, 0].plot(gxy1, ey1,'r')
axs3[2, 0].plot(gxy1, gxy1,'b')
axs3[2, 0].plot(gxy1, e1,'m')
axs3[2, 0].plot(gxy1, e3,'c')
axs3[2, 0].legend(['$\epsilon_x$','$\epsilon_y$','$\gamma_{xy}$','$\epsilon_1$','$\epsilon_3$'])
axs3[2, 0].set_title('Strains')
axs3[2, 0].set(xlabel="$\gamma_{xy}$ [m$\epsilon$]")
axs3[2, 0].set(ylabel="$\epsilon$")

axs3[2, 1].set_title('Principal Dir')
axs3[2 ,1].plot(gxy1, th*180/pi, 'k')
axs3[2, 1].set(xlabel="$\gamma_{xy}$ [m$\epsilon$]")
axs3[2, 1].set(ylabel="$\\theta$ [°]")

axs3[2, 2].set_title('Submodel')
axs3[2 ,2].plot(gxy1, submodel, 'k')
axs3[2, 2].set(xlabel="$\gamma_{xy}$ [m$\epsilon$]")
axs3[2, 2].set(ylabel="Model")

fig3.tight_layout(pad=2.0)


plt.figure(4)
plt.subplot(3,1,1)
plt.plot(ex1, ssrx)
plt.xlabel("$\epsilon$$_{x} [-]$")
plt.ylabel("$\sigma_{sr} [MPa]$")

plt.subplot(3,1,2)
plt.plot(ex1, sc3)
plt.xlabel("$\epsilon$$_{x} [-]$")
plt.ylabel("$\sigma_{c3r}  [MPa]$")

plt.subplot(3,1,3)
plt.plot(e1, fcs)
plt.xlabel("$\epsilon$$_{1} [-]$")
plt.ylabel("$f_{cs}  [MPa]$")

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

path = r"C:\Users\naesboma\00_an\04_UHBB\02_Materialmodelle\01_TCM"
if sig.cms_klij == 4:
    np.savetxt(os.path.join(path, 'xcr.txt'),xcr_01)
    np.savetxt(os.path.join(path, 'es_all.txt'),es_all_01)
    np.savetxt(os.path.join(path, 'ep_all.txt'),ep_all_01)
    np.savetxt(os.path.join(path, 'ds_all.txt'),ds_all_01)
    np.savetxt(os.path.join(path, 'dp_all.txt'),dp_all_01)
    np.savetxt(os.path.join(path, 'ss_all.txt'),ss_all_01)
    np.savetxt(os.path.join(path, 'sp_all.txt'),sp_all_01)
    np.savetxt(os.path.join(path, 'tbs_all.txt'),tbs_all_01)
    np.savetxt(os.path.join(path, 'tbp_all.txt'),tbp_all_01)
np.savetxt(os.path.join(path, 'Q_TC.txt'),Q_TC)
np.savetxt(os.path.join(path, 'em_TC.txt'),ex1)
np.savetxt(os.path.join(path, 'sx_TC.txt'),sx1)
np.savetxt(os.path.join(path, 'wr_TC.txt'),wr)
np.savetxt(os.path.join(path, 'ssrx.txt'),ssrx)
np.savetxt(os.path.join(path, 'sprx.txt'),sprx)
np.savetxt(os.path.join(path, 'srm.txt'),srm)
plt.show()
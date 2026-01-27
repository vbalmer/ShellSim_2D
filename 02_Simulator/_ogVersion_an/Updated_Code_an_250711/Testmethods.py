""" --------------------------------------------------------------------------------------------------------------------
    ---------------------------------- Script to test methods from stress class ----------------------------------------
    -----------------------------------------------------------------------------------------------------------------"""

from Stresses_mixreinf import stress
import matplotlib.pyplot as plt
import numpy as np
import os

# ----------------------------------------------------------------------------------------------------------------------
# 0 Initiate Stress instance
# ----------------------------------------------------------------------------------------------------------------------

tb0 = 6.4
tb1 = 3.2
tbp0 = 6
tbp1 = 5.9
dx = 20
sx = 100
rhox = 0.0233
Esx = 199000
Eshx = 841*10
fsyx = 505
fsux = 608

Ec = 30100
Gc = 30100/(2*(1+0.2))
vc = 0.0
t = 270
fcp = 33.5
fct = 3.2
ec0 = 2e-3
Dmax = 16
rhoy = 0.0086
dy  = 14
sy  = 200
Esy = 201000
Eshy = 717
fsyy = 520
fsuy = 614


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

MAT = [Ec, vc, fcp, fct, ec0, 0*fct/Ec, Dmax, Esx, Eshx, fsyx, fsux, Esy, Eshy, fsyy, fsuy, tb0, tb1, Epx, Epy, tbp0,
       tbp1, ebp1, fpux, fpuy]
GEOM = [rhox, rhoy, dx, dy, sx, sy, rhopx, rhopy, dpx, dpy]

cm_klij = 4
cmcc_klij = 3
cmcs_klij = 3
cms_klij = 1
cmtn_klij = 3
s = stress(0,3, MAT, GEOM, cmcc_klij, cmcs_klij, cms_klij, cmtn_klij)

# ----------------------------------------------------------------------------------------------------------------------
# 1 Test aggregate interlock laws
# ----------------------------------------------------------------------------------------------------------------------

dn,dt,tau,sigma = s.crack_kin(True)

fig1, axs = plt.subplots(1, 1)
fig1.set_figheight(6)
fig1.set_figwidth(6)
axs.plot(dt, tau, 'm')
axs.plot(dt, sigma, 'c')
plt.show()

path = r'C:\Users\naesboma\00_an\FEM_Q\Verifications\1_Interlock'
np.savetxt(os.path.join(path, 'dn.txt'), dn)
np.savetxt(os.path.join(path, 'dt.txt'), dt)
np.savetxt(os.path.join(path, 'tctnr.txt'), tau)
np.savetxt(os.path.join(path, 'scnr.txt'), sigma)

# ----------------------------------------------------------------------------------------------------------------------
# 2 Concrete in compression
# ----------------------------------------------------------------------------------------------------------------------

# steps = 100
# e3_all = np.linspace(0, -2.5e-3, steps)
# e1_all = np.linspace(0.001, 0.001, steps)
# sc3_all = np.zeros_like(e3_all)
#
# for j in range(len(e3_all)):
#        s.e1 = e1_all[j]
#        sc3_all[j] = s.sc(e3_all[j])
#
# fig1, axs = plt.subplots(1, 1)
# fig1.set_figheight(6)
# fig1.set_figwidth(6)
# axs.plot(e3_all, sc3_all, 'k')
# plt.show()
#
# path = r'C:\Users\naesboma\00_an\FEM_Q\Verifications\2_Concrete'
# np.savetxt(os.path.join(path, 'e3.txt'), e3_all)
# np.savetxt(os.path.join(path, 'sc3.txt'), sc3_all)

# ----------------------------------------------------------------------------------------------------------------------
# 3 Softening law
# ----------------------------------------------------------------------------------------------------------------------

# steps = 200
# e3_all = np.linspace(-2.5e-3, -2.5e-3, steps)
# e1_all = np.linspace(0, 0.025, steps)
# kc_all = np.zeros_like(e3_all)
#
# for j in range(len(e3_all)):
#        s.e1 = e1_all[j]
#        kc_all[j] = -s.sc(e3_all[j])/fcp
#
# fig1, axs = plt.subplots(1, 1)
# fig1.set_figheight(6)
# fig1.set_figwidth(6)
# axs.plot(e1_all, kc_all, 'k')
# plt.show()
#
# path = r'C:\Users\naesboma\00_an\FEM_Q\Verifications\2_Concrete'
# np.savetxt(os.path.join(path, 'e1.txt'), e1_all)
# np.savetxt(os.path.join(path, 'kc.txt'), kc_all)

# ----------------------------------------------------------------------------------------------------------------------
# 4 Tension chord model
# ----------------------------------------------------------------------------------------------------------------------

# steps = 300
# e_all = np.linspace(-.0005, 0.015, steps)
# ssr_all = np.zeros_like(e_all)
# ssbare_all = np.zeros_like(e_all)
#
# for j in range(len(e_all)):
#        ssr,_ = s.ssr(e_all[j], -0.0005, 150*2, rhox, 0, dx, fsyx, fsux, Esx, Eshx, tb0, tb1, Ec)
#        if ssr < fsux:
#               ssr_all[j] = ssr.real
#        else:
#               ssr_all[j] = 'nan' \
#                            ''
#        ssbare = s.ss_bilin(e_all[j], fsyx, Esx, Eshx)
#        if ssbare < fsux:
#               ssbare_all[j] = ssbare.real
#        else:
#               ssbare_all[j] = 'nan'
#
# fig1, axs = plt.subplots(1, 1)
# fig1.set_figheight(6)
# fig1.set_figwidth(6)
# axs.plot(e_all, ssr_all, 'b')
# axs.plot(e_all, ssbare_all, 'b--')
# plt.show()
#
# path = r'C:\Users\naesboma\00_an\FEM_Q\Verifications\3_Steel'
# np.savetxt(os.path.join(path, 'e.txt'), e_all)
# np.savetxt(os.path.join(path, 'ssr.txt'), ssr_all)
# np.savetxt(os.path.join(path, 'ssb.txt'), ssbare_all)

# ----------------------------------------------------------------------------------------------------------------------
# 4 Seelhofer
# ----------------------------------------------------------------------------------------------------------------------

# steps = 300
# e_all = np.linspace(-.0004, 0.015, steps)
# ssr_all = np.zeros_like(e_all)
# ssbare_all = np.zeros_like(e_all)
#
# for j in range(len(e_all)):
#        ssr,_ = s.ssr_seelhofer(e_all[j], -0.0005, rhox, 600, dx, fsyx, Esx, Eshx, tb0, tb1)
#        if ssr < fsux:
#               ssr_all[j] = ssr.real
#        else:
#               ssr_all[j] = 'nan' \
#                            ''
#        ssbare = s.ss_bilin(e_all[j], fsyx, Esx, Eshx)
#        if ssbare < fsux:
#               ssbare_all[j] = ssbare.real
#        else:
#               ssbare_all[j] = 'nan'
#
# fig1, axs = plt.subplots(1, 1)
# fig1.set_figheight(6)
# fig1.set_figwidth(6)
# axs.plot(e_all, ssr_all, 'b')
# axs.plot(e_all, ssbare_all, 'b--')
# plt.show()
#
# path = r'C:\Users\naesboma\00_an\FEM_Q\Verifications\3_Steel'
# np.savetxt(os.path.join(path, 'e_seel.txt'), e_all)
# np.savetxt(os.path.join(path, 'ssr_seel.txt'), ssr_all)
# np.savetxt(os.path.join(path, 'ssb_seel.txt'), ssbare_all)
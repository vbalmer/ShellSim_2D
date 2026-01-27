import numpy as np
from Stresses_mixreinf import stress
import matplotlib.pyplot as plt


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

# 0 Material Parameter Assignment
Ec = 40000
Gc = 40000/(2*(1+0.2))
vc = 0.0
t = 100
fcp = 100
fct = 6.5
tb0 = 13
tb1 = 6.5
ec0 = 2.5e-3
rhox = 0.006*10**-6
dx = 12
Esx = 205000
Eshx = 1000
fsyx = 600
fsux = 700
rhoy = 0.005*10**-6
dy  = 10
Esy = 205000
Eshy = 1000
fsyy = 600
fsuy = 700

Epx = 140000
Epy = 140000
tbp0 = 9
tbp1 = 7
ebp1 = 0.01
fpux = 1600
fpuy = 1600
rhopx = 0
rhopy = 0
dpx = 8.2
dpy = 8.2

MAT = [Ec, vc, fcp, fct, ec0,0, Esx, Eshx, fsyx, fsux, Esy, Eshy, fsyy, fsuy, tb0, tb1, Epx, Epy, tbp0,
       tbp1, ebp1, fpux, fpuy]
GEOM = [rhox, rhoy, dx, dy, rhopx, rhopy, dpx, dpy]

# 1 Iteration and Results
# Output:
#   - ex1: Values of epsilon_x to be iterated through
#   - sx1: Pertaining values of stress (calculated in stress class)
#   - sxdev1: Pertaining values of (d sigma_x)/(d epsilon_x)
exmin = -5e-3
exmax = 0e-3
steps = 5000
ex1 = np.linspace(exmin, exmax, steps)

eymin = 3e-3
eymax = 3e-3
ey1 = np.linspace(eymin,eymax,steps)

gxymin = 1e-3
gxymax = 1e-3
gxy1 = np.linspace(gxymin,gxymax,steps)

e1 = np.zeros_like(ex1)
e3 = np.zeros_like(ex1)
sx1 = np.zeros_like(ex1)
sy1 = np.zeros_like(ex1)
txy1 = np.zeros_like(ex1)
dsx_dex = np.zeros_like(ex1)
dsx_dey = np.zeros_like(ex1)
dsx_dgxy = np.zeros_like(ex1)
for i in range(len(ex1)):
    sig = stress(3, MAT, GEOM)
    [epsilon,sigma,Tanmat] = Ableitungsfunktion(sig,np.array([ex1[i], ey1[i], gxy1[i], 0.001, 0.001]),3)
    sx1[i] = sigma[0]
    sy1[i] = sigma[1]
    txy1[i] = sigma[2]
    dsx_dex[i] = Tanmat[0][0]
    dsx_dey[i] = Tanmat[0][1]
    dsx_dgxy[i] = Tanmat[0][2]
    e1[i] = epsilon[3]
    e3[i] = epsilon[4]

# 2 Control Calculations
# Output:
#   - sx2: stress values calculated by integrating sxdev1
#   - ex2: strain values between values of ex1: ex2[i]=(ex1[i]+ex1[i+1])/2
#   - sxdev2: Tangent stiffness derived via sxdev2[i] = (sx1[i+1]-sx1[i])/(ex1[i+1] - ex1[i])
#       --> "Less fancy" way to calculate derivative, but should be "equal" to sxdev1!
sx2 = np.zeros_like(ex1)
for i in range(len(sx2)):
    if i == 0:
        sx2[i] = sx1[0]
    else:
        sx2[i] = sx2[i - 1] + (dsx_dex[i - 1] + dsx_dex[i]) / 2 * (ex1[i] - ex1[i - 1]) + (dsx_dey[i - 1] + dsx_dey[i]) / 2 * (ey1[i] - ey1[i - 1]) + (dsx_dgxy[i - 1] + dsx_dgxy[i]) / 2 * (gxy1[i] - gxy1[i - 1])


ex2 = np.zeros(len(ex1)-1)
sxdev2 = np.zeros_like(ex2)
for i in range(len(ex2)):
    ex2[i] = (ex1[i]+ex1[i+1])/2
    sxdev2[i] = (sx1[i+1]-sx1[i])/(ex1[i+1] - ex1[i])

# 3 Plot
plt.subplot(2,1,1)
plt.plot(ex1, sx1)
plt.plot(ex1, sx2, linestyle='dashed')
plt.legend(["$\sigma_{x}$","$\sigma_{x,int}$"])
plt.xlabel("$\epsilon$$_{x} [-]$")
plt.ylabel("$\sigma_{x,Layer} [MPa]$")

plt.subplot(2,1,2)
plt.plot(ex1, dsx_dex)
# plt.plot(ex2, sxdev2, linestyle='dashed')
plt.xlabel("$\epsilon$$_{x} [-]$")
plt.ylabel("d $\sigma_{x}$ / d $\epsilon_{x}$ [MPa]")


plt.subplots_adjust(hspace=0.5)
plt.show()
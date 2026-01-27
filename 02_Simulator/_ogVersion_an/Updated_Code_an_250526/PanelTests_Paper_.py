import numpy as np
from Stresses_mixreinf import stress,lbd, do_itkin, do_char
from scipy import interpolate
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

def movmean(x,wind):
    x_smooth = np.zeros_like(x)
    l = len(x)
    for i in range(l):
        if i == 0:
            x_smooth[i] = x[i]
        elif i == l-1:
            x_smooth[i] = x[i]
        else:
            windi = np.min([wind,i,l-1-i])
            if windi == 0:
                x_smooth[i] = x[i]
            else:
                x_smooth[i] = np.mean(x[i-windi:i+windi])
    return x_smooth
""" ---------------------------------------------------------------------------------------------------------------- 
 ---------------------------------------------------- Initiate -----------------------------------------------------
 ------------------------------------------------------------------------------------------------------------------- """
path = r'C:\Users\naesboma\00_an\04_UHBB\06_Papers\06_Numerical_Modelling\02_Modell\02_Nachrechnungen_Paper'

tb0 = {}
tb1 = {}
tbp0 = {}
tbp1 = {}
dx = {}
sx = {}
rhox = {}
Esx = {}
Eshx = {}
fsyx = {}
fsux = {}

Ec = {}
Gc = {}
vc = {}
t = {}
fcp = {}
fct = {}
ec0 = {}
Dmax = {}
rhoy = {}
dy  = {}
sy  = {}
Esy = {}
Eshy = {}
fsyy = {}
fsuy = {}

Epx = {}
Epy = {}
ebp1 = {}
sp0 = {}
fpux = {}
fpuy = {}
dpx = {}
dpy = {}
rhopx = {}
rhopy = {}


""" ---------------------------------------------------------------------------------------------------------------- 
 ------------------------------- Introduce Tests and their parameters ----------------------------------------------
 ------------------------------------------------------------------------------------------------------------------- """
# Tests = ['VA0','VA1','VA2','VA3','VA4'] # Houston, high strength, show
# Tests = ['PV10', 'PV19', 'PV20', 'PV21', 'PV22'] # Toronto, normal strength, no show
# Tests = ['PP1','PP2','PP3'] # Toronto, normal strength, prestressed (PP2 and PP3) show
# Tests = ['SE1','SE6'] # Toronto, normal strength, show
Tests = ['HB3','HB4'] # Houston, high strength, show
# Tests = ['VB1','VB2','VB3'] # Houston, high strength, show
# Tests = ['PHS2','PHS3','PHS4','PHS5','PHS6','PHS7'] # Toronto, no show
# Tests = ['SL1','SL3'] # Zürich, show
# Tests = ['SE1','SE6','PP1','PT1','PP2','PP3','SL1','SL3']
# Tests = ['VA0','VA2','VA4','VB1','VB2','VB3','HB3','HB4']
# Tests = ['VB1','VB2','VB3','HB3','HB4']
" ---------------------------------------------------- VA0 ------------------------------------------------------------"
fcp['VA0'] = 98.8
Ec['VA0'] = 10000*fcp['VA0']**(1/3)
Gc['VA0'] = Ec['VA0']/2
vc['VA0'] = 0
t['VA0'] = 178
fct['VA0'] = 0.3*fcp['VA0']**(2/3)
ec0['VA0'] = 2.4e-3
Dmax['VA0'] = 10 # Guess

fsyx['VA0'] = 445
fsux['VA0'] = 579
Esx['VA0'] = 200000
esyx = fsyx['VA0']/Esx['VA0']
esux = 0.1
Eshx['VA0'] = (fsux['VA0']-fsyx['VA0'])/(esux-esyx)
dx['VA0'] = 11.3
sx['VA0'] = 189
rhox['VA0'] = 0.00571

fsyy['VA0'] = 445
fsuy['VA0'] = 579
Esy['VA0'] = 200000
esyy = fsyy['VA0']/Esy['VA0']
esuy = 0.1
Eshy['VA0'] = (fsuy['VA0']-fsyy['VA0'])/(esuy-esyy)
dy['VA0'] = 11.3
sy['VA0'] = 189
rhoy['VA0'] = 0.00571

tb0['VA0'] = 2*fct['VA0']
tb1['VA0'] = fct['VA0']
tbp0['VA0'] = 6
tbp1['VA0'] = 3

Epx['VA0'] = 1
Epy['VA0'] = 1
ebp1['VA0'] = .01
sp0['VA0'] = 0
fpux['VA0'] = 1
fpuy['VA0'] = 1
dpx['VA0'] = 8
dpy['VA0'] = 8
rhopx['VA0'] = 0
rhopy['VA0'] = 0

" ---------------------------------------------------- VA1 ------------------------------------------------------------"
fcp['VA1'] = 95.1
Ec['VA1'] = 10000*fcp['VA1']**(1/3)
Gc['VA1'] = Ec['VA1']/2
vc['VA1'] = 0
t['VA1'] = 178
fct['VA1'] = 0.3*fcp['VA1']**(2/3)
ec0['VA1'] = 2.45e-3
Dmax['VA1'] = 10 # Guess

fsyx['VA1'] = 445
fsux['VA1'] = 579
Esx['VA1'] = 200000
esyx = fsyx['VA1']/Esx['VA1']
esux = 0.1
Eshx['VA1'] = (fsux['VA1']-fsyx['VA1'])/(esux-esyx)
dx['VA1'] = 11.3
sx['VA1'] = 189
rhox['VA1'] = 0.01143

fsyy['VA1'] = 445
fsuy['VA1'] = 579
Esy['VA1'] = 200000
esyy = fsyy['VA1']/Esy['VA1']
esuy = 0.1
Eshy['VA1'] = (fsuy['VA1']-fsyy['VA1'])/(esuy-esyy)
dy['VA1'] = 11.3
sy['VA1'] = 189
rhoy['VA1'] = 0.01143

tb0['VA1'] = 2*fct['VA1']
tb1['VA1'] = fct['VA1']
tbp0['VA1'] = 6
tbp1['VA1'] = 3

Epx['VA1'] = 1
Epy['VA1'] = 1
ebp1['VA1'] = .01
sp0['VA1'] = 0
fpux['VA1'] = 1
fpuy['VA1'] = 1
dpx['VA1'] = 8
dpy['VA1'] = 8
rhopx['VA1'] = 0
rhopy['VA1'] = 0

" ---------------------------------------------------- VA2 ------------------------------------------------------------"
fcp['VA2'] = 98.2
Ec['VA2'] = 10000*fcp['VA2']**(1/3)
Gc['VA2'] = Ec['VA2']/2
vc['VA2'] = 0
t['VA2'] = 178
fct['VA2'] = 0.3*fcp['VA2']**(2/3)
ec0['VA2'] = 2.5e-3
Dmax['VA2'] = 10 # Guess

fsyx['VA2'] = 409
fsux['VA2'] = 534
Esx['VA2'] = 200000
esyx = fsyx['VA2']/Esx['VA2']
esux = 0.1
Eshx['VA2'] = (fsux['VA2']-fsyx['VA2'])/(esux-esyx)
dx['VA2'] = 16
sx['VA2'] = 189
rhox['VA2'] = 0.02276

fsyy['VA2'] = 409
fsuy['VA2'] = 534
Esy['VA2'] = 200000
esyy = fsyy['VA2']/Esy['VA2']
esuy = 0.1
Eshy['VA2'] = (fsuy['VA2']-fsyy['VA2'])/(esuy-esyy)
dy['VA2'] = 16
sy['VA2'] = 189
rhoy['VA2'] = 0.02276

tb0['VA2'] = 2*fct['VA2']
tb1['VA2'] = fct['VA2']
tbp0['VA2'] = 6
tbp1['VA2'] = 3

Epx['VA2'] = 1
Epy['VA2'] = 1
ebp1['VA2'] = .01
sp0['VA2'] = 0
fpux['VA2'] = 1
fpuy['VA2'] = 1
dpx['VA2'] = 8
dpy['VA2'] = 8
rhopx['VA2'] = 0
rhopy['VA2'] = 0

" ---------------------------------------------------- VA3 ------------------------------------------------------------"
fcp['VA3'] = 94.6
Ec['VA3'] = 10000*fcp['VA3']**(1/3)
Gc['VA3'] = Ec['VA3']/2
vc['VA3'] = 0
t['VA3'] = 178
fct['VA3'] = 0.3*fcp['VA3']**(2/3)
ec0['VA3'] = 2.45e-3
Dmax['VA3'] = 10 # Guess

fsyx['VA3'] = 455
fsux['VA3'] = 608
Esx['VA3'] = 200000
esyx = fsyx['VA3']/Esx['VA3']
esux = 0.1
Eshx['VA3'] = (fsux['VA3']-fsyx['VA3'])/(esux-esyx)
dx['VA3'] = 19.5
sx['VA3'] = 189
rhox['VA3'] = 0.03419

fsyy['VA3'] = 455
fsuy['VA3'] = 608
Esy['VA3'] = 200000
esyy = fsyy['VA3']/Esy['VA3']
esuy = 0.1
Eshy['VA3'] = (fsuy['VA3']-fsyy['VA3'])/(esuy-esyy)
dy['VA3'] = 19.5
sy['VA3'] = 189
rhoy['VA3'] = 0.03419

tb0['VA3'] = 2*fct['VA3']
tb1['VA3'] = fct['VA3']
tbp0['VA3'] = 6
tbp1['VA3'] = 3

Epx['VA3'] = 1
Epy['VA3'] = 1
ebp1['VA3'] = .01
sp0['VA3'] = 0
fpux['VA3'] = 1
fpuy['VA3'] = 1
dpx['VA3'] = 8
dpy['VA3'] = 8
rhopx['VA3'] = 0
rhopy['VA3'] = 0

" ---------------------------------------------------- VA4 ------------------------------------------------------------"
fcp['VA4'] = 103.1
Ec['VA4'] = 10000*fcp['VA4']**(1/3)
Gc['VA4'] = Ec['VA4']/2
vc['VA4'] = 0
t['VA4'] = 178
fct['VA4'] = 0.3*fcp['VA4']**(2/3)
ec0['VA4'] = 2.35e-3
Dmax['VA4'] = 10 # Guess

fsyx['VA4'] = 470
fsux['VA4'] = 606
Esx['VA4'] = 200000
esyx = fsyx['VA4']/Esx['VA4']
esux = 0.1
Eshx['VA4'] = (fsux['VA4']-fsyx['VA4'])/(esux-esyx)
dx['VA4'] = 25.2
sx['VA4'] = 189
rhox['VA4'] = 0.0499

fsyy['VA4'] = 470
fsuy['VA4'] = 606
Esy['VA4'] = 200000
esyy = fsyy['VA4']/Esy['VA4']
esuy = 0.1
Eshy['VA4'] = (fsuy['VA4']-fsyy['VA4'])/(esuy-esyy)
dy['VA4'] = 25.2
sy['VA4'] = 189
rhoy['VA4'] = 0.0499

tb0['VA4'] = 2*fct['VA4']
tb1['VA4'] = fct['VA4']
tbp0['VA4'] = 6
tbp1['VA4'] = 3

Epx['VA4'] = 1
Epy['VA4'] = 1
ebp1['VA4'] = .01
sp0['VA4'] = 0
fpux['VA4'] = 1
fpuy['VA4'] = 1
dpx['VA4'] = 8
dpy['VA4'] = 8
rhopx['VA4'] = 0
rhopy['VA4'] = 0

" ---------------------------------------------------- PV27 ------------------------------------------------------------"
fcp['PV27'] = 20.5
Ec['PV27'] = 10000*fcp['PV27']**(1/3)
Gc['PV27'] = Ec['PV27']/2
vc['PV27'] = 0
t['PV27'] = 70
fct['PV27'] = 0.3*fcp['PV27']**(2/3)
ec0['PV27'] = 1.9e-3
Dmax['PP27'] = 13 # Guess

fsyx['PV27'] = 442
fsux['PV27'] = 508
Esx['PV27'] = 200000
esyx = fsyx['PV27']/Esx['PV27']
esux = 0.1
Eshx['PV27'] = (fsux['PV27']-fsyx['PV27'])/(esux-esyx)
dx['PV27'] = 6.35
sx['PV27'] = 50
rhox['PV27'] = 0.01785

fsyy['PV27'] = 442
fsuy['PV27'] = 508
Esy['PV27'] = 200000
esyy = fsyy['PV27']/Esy['PV27']
esuy = 0.1
Eshy['PV27'] = (fsuy['PV27']-fsyy['PV27'])/(esuy-esyy)
dy['PV27'] = 6.35
sy['PV27'] = 50
rhoy['PV27'] = 0.01785

tb0['PV27'] = 2*fct['PV27']
tb1['PV27'] = fct['PV27']
tbp0['PV27'] = 6
tbp1['PV27'] = 3

Epx['PV27'] = 1
Epy['PV27'] = 1
ebp1['PV27'] = .01
sp0['PV27'] = 0
fpux['PV27'] = 1
fpuy['PV27'] = 1
dpx['PV27'] = 8
dpy['PV27'] = 8
rhopx['PV27'] = 0
rhopy['PV27'] = 0

" ---------------------------------------------------- PV10 ------------------------------------------------------------"
fcp['PV10'] = 14.5
Ec['PV10'] = 10000*fcp['PV10']**(1/3)
Gc['PV10'] = Ec['PV10']/2
vc['PV10'] = 0
t['PV10'] = 70
fct['PV10'] = 0.3*fcp['PV10']**(2/3)
ec0['PV10'] = 2.7e-3
Dmax['PP10'] = 13 # Guess

fsyx['PV10'] = 276
fsux['PV10'] = 317
Esx['PV10'] = 200000
esyx = fsyx['PV10']/Esx['PV10']
esux = 0.1
Eshx['PV10'] = (fsux['PV10']-fsyx['PV10'])/(esux-esyx)
dx['PV10'] = 6.35
sx['PV10'] = 50
rhox['PV10'] = 0.01785

fsyy['PV10'] = 276
fsuy['PV10'] = 317
Esy['PV10'] = 200000
esyy = fsyy['PV10']/Esy['PV10']
esuy = 0.1
Eshy['PV10'] = (fsuy['PV10']-fsyy['PV10'])/(esuy-esyy)
dy['PV10'] = 4.7
sy['PV10'] = 50
rhoy['PV10'] = 0.00999

tb0['PV10'] = 2*fct['PV10']
tb1['PV10'] = fct['PV10']
tbp0['PV10'] = 6
tbp1['PV10'] = 3

Epx['PV10'] = 1
Epy['PV10'] = 1
ebp1['PV10'] = .01
sp0['PV10'] = 0
fpux['PV10'] = 1
fpuy['PV10'] = 1
dpx['PV10'] = 8
dpy['PV10'] = 8
rhopx['PV10'] = 0
rhopy['PV10'] = 0

" ---------------------------------------------------- PV19 ------------------------------------------------------------"
fcp['PV19'] = 19
Ec['PV19'] = 10000*fcp['PV19']**(1/3)
Gc['PV19'] = Ec['PV19']/2
vc['PV19'] = 0
t['PV19'] = 70
fct['PV19'] = 0.3*fcp['PV19']**(2/3)
ec0['PV19'] = 2.15e-3
Dmax['PP19'] = 13 # Guess

fsyx['PV19'] = 458
fsux['PV19'] = 527
Esx['PV19'] = 200000
esyx = fsyx['PV19']/Esx['PV19']
esux = 0.1
Eshx['PV19'] = (fsux['PV19']-fsyx['PV19'])/(esux-esyx)
dx['PV19'] = 6.35
sx['PV19'] = 50
rhox['PV19'] = 0.01785

fsyy['PV19'] = 299
fsuy['PV19'] = 344
Esy['PV19'] = 200000
esyy = fsyy['PV19']/Esy['PV19']
esuy = 0.1
Eshy['PV19'] = (fsuy['PV19']-fsyy['PV19'])/(esuy-esyy)
dy['PV19'] = 4.01
sy['PV19'] = 50
rhoy['PV19'] = 0.00713

tb0['PV19'] = 2*fct['PV19']
tb1['PV19'] = fct['PV19']
tbp0['PV19'] = 6
tbp1['PV19'] = 3

Epx['PV19'] = 1
Epy['PV19'] = 1
ebp1['PV19'] = .01
sp0['PV19'] = 0
fpux['PV19'] = 1
fpuy['PV19'] = 1
dpx['PV19'] = 8
dpy['PV19'] = 8
rhopx['PV19'] = 0
rhopy['PV19'] = 0

" ---------------------------------------------------- PV20 ------------------------------------------------------------"
fcp['PV20'] = 19.6
Ec['PV20'] = 10000*fcp['PV20']**(1/3)
Gc['PV20'] = Ec['PV20']/2
vc['PV20'] = 0
t['PV20'] = 70
fct['PV20'] = 0.3*fcp['PV20']**(2/3)
ec0['PV20'] = 1.8e-3
Dmax['PP20'] = 13 # Guess

fsyx['PV20'] = 460
fsux['PV20'] = 529
Esx['PV20'] = 200000
esyx = fsyx['PV20']/Esx['PV20']
esux = 0.1
Eshx['PV20'] = (fsux['PV20']-fsyx['PV20'])/(esux-esyx)
dx['PV20'] = 6.35
sx['PV20'] = 50
rhox['PV20'] = 0.01785

fsyy['PV20'] = 297
fsuy['PV20'] = 341
Esy['PV20'] = 200000
esyy = fsyy['PV20']/Esy['PV20']
esuy = 0.1
Eshy['PV20'] = (fsuy['PV20']-fsyy['PV20'])/(esuy-esyy)
dy['PV20'] = 4.47
sy['PV20'] = 50
rhoy['PV20'] = 0.00885

tb0['PV20'] = 2*fct['PV20']
tb1['PV20'] = fct['PV20']
tbp0['PV20'] = 6
tbp1['PV20'] = 3

Epx['PV20'] = 1
Epy['PV20'] = 1
ebp1['PV20'] = .01
sp0['PV20'] = 0
fpux['PV20'] = 1
fpuy['PV20'] = 1
dpx['PV20'] = 8
dpy['PV20'] = 8
rhopx['PV20'] = 0
rhopy['PV20'] = 0

" ---------------------------------------------------- PV21 ------------------------------------------------------------"
fcp['PV21'] = 19.5
Ec['PV21'] = 10000*fcp['PV21']**(1/3)
Gc['PV21'] = Ec['PV21']/2
vc['PV21'] = 0
t['PV21'] = 70
fct['PV21'] = 0.3*fcp['PV21']**(2/3)
ec0['PV21'] = 1.8e-3
Dmax['PP21'] = 13 # Guess

fsyx['PV21'] = 458
fsux['PV21'] = 527
Esx['PV21'] = 200000
esyx = fsyx['PV21']/Esx['PV21']
esux = 0.1
Eshx['PV21'] = (fsux['PV21']-fsyx['PV21'])/(esux-esyx)
dx['PV21'] = 6.35
sx['PV21'] = 50
rhox['PV21'] = 0.01785

fsyy['PV21'] = 302
fsuy['PV21'] = 347
Esy['PV21'] = 200000
esyy = fsyy['PV21']/Esy['PV21']
esuy = 0.1
Eshy['PV21'] = (fsuy['PV21']-fsyy['PV21'])/(esuy-esyy)
dy['PV21'] = 5.41
sy['PV21'] = 50
rhoy['PV21'] = 0.01296

tb0['PV21'] = 2*fct['PV21']
tb1['PV21'] = fct['PV21']
tbp0['PV21'] = 6
tbp1['PV21'] = 3

Epx['PV21'] = 1
Epy['PV21'] = 1
ebp1['PV21'] = .01
sp0['PV21'] = 0
fpux['PV21'] = 1
fpuy['PV21'] = 1
dpx['PV21'] = 8
dpy['PV21'] = 8
rhopx['PV21'] = 0
rhopy['PV21'] = 0

" ---------------------------------------------------- PV22 ------------------------------------------------------------"
fcp['PV22'] = 19.6
Ec['PV22'] = 10000*fcp['PV22']**(1/3)
Gc['PV22'] = Ec['PV22']/2
vc['PV22'] = 0
t['PV22'] = 70
fct['PV22'] = 0.3*fcp['PV22']**(2/3)
ec0['PV22'] = 2e-3
Dmax['PP22'] = 13 # Guess

fsyx['PV22'] = 458
fsux['PV22'] = 527
Esx['PV22'] = 200000
esyx = fsyx['PV22']/Esx['PV22']
esux = 0.1
Eshx['PV22'] = (fsux['PV22']-fsyx['PV22'])/(esux-esyx)
dx['PV22'] = 6.35
sx['PV22'] = 50
rhox['PV22'] = 0.01785

fsyy['PV22'] = 420
fsuy['PV22'] = 483
Esy['PV22'] = 200000
esyy = fsyy['PV22']/Esy['PV22']
esuy = 0.1
Eshy['PV22'] = (fsuy['PV22']-fsyy['PV22'])/(esuy-esyy)
dy['PV22'] = 5.87
sy['PV22'] = 50
rhoy['PV22'] = 0.01524

tb0['PV22'] = 2*fct['PV22']
tb1['PV22'] = fct['PV22']
tbp0['PV22'] = 6
tbp1['PV22'] = 3

Epx['PV22'] = 1
Epy['PV22'] = 1
ebp1['PV22'] = .01
sp0['PV22'] = 0
fpux['PV22'] = 1
fpuy['PV22'] = 1
dpx['PV22'] = 8
dpy['PV22'] = 8
rhopx['PV22'] = 0
rhopy['PV22'] = 0

" ---------------------------------------------------- PP1 ------------------------------------------------------------"
fcp['PP1'] = 27
Ec['PP1'] = 10000*fcp['PP1']**(1/3)
Gc['PP1'] = Ec['PP1']/2
vc['PP1'] = 0
t['PP1'] = 287
fct['PP1'] = 0.3*fcp['PP1']**(2/3)
ec0['PP1'] = 2.12e-3
Dmax['PP1'] = 13

fsyx['PP1'] = 479
fsux['PP1'] = 667
Esx['PP1'] = 200000
esyx = fsyx['PP1']/Esx['PP1']
esux = 0.09
Eshx['PP1'] = (fsux['PP1']-fsyx['PP1'])/(esux-esyx)
dx['PP1'] = 19.5
sx['PP1'] = 108
rhox['PP1'] = 0.01942

fsyy['PP1'] = 480
fsuy['PP1'] = 640
Esy['PP1'] = 200000
esyy = fsyy['PP1']/Esy['PP1']
esuy = 0.091
Eshy['PP1'] = (fsuy['PP1']-fsyy['PP1'])/(esuy-esyy)
dy['PP1'] = 11.3
sy['PP1'] = 108
rhoy['PP1'] = 0.00647

tb0['PP1'] = 2*fct['PP1']
tb1['PP1'] = fct['PP1']
tbp0['PP1'] = 6
tbp1['PP1'] = 3

Epx['PP1'] = 1
Epy['PP1'] = 1
ebp1['PP1'] = .01
sp0['PP1'] = 0
fpux['PP1'] = 1
fpuy['PP1'] = 1
dpx['PP1'] = 8
dpy['PP1'] = 8
rhopx['PP1'] = 0
rhopy['PP1'] = 0


" ---------------------------------------------------- PP2 ------------------------------------------------------------"
fcp['PP2'] = 28.1
Ec['PP2'] = 10000*fcp['PP2']**(1/3)
Gc['PP2'] = Ec['PP2']/2
vc['PP2'] = 0
t['PP2'] = 287
fct['PP2'] = 0.3*fcp['PP2']**(2/3)
ec0['PP2'] = 2.38e-3
Dmax['PP2'] = 13

fsyx['PP2'] = 486
fsux['PP2'] = 630
Esx['PP2'] = 200000
esyx = fsyx['PP2']/Esx['PP2']
esux = 0.1
Eshx['PP2'] = (fsux['PP2']-fsyx['PP2'])/(esux-esyx)
dx['PP2'] = 16
sx['PP2'] = 108
rhox['PP2'] = 0.01295

fsyy['PP2'] = 480
fsuy['PP2'] = 640
Esy['PP2'] = 200000
esyy = fsyy['PP2']/Esy['PP2']
esuy = 0.091
Eshy['PP2'] = (fsuy['PP2']-fsyy['PP2'])/(esuy-esyy)
dy['PP2'] = 11.3
sy['PP2'] = 108
rhoy['PP2'] = 0.00647

tb0['PP2'] = 2*fct['PP2']
tb1['PP2'] = fct['PP2']
tbp0['PP2'] = 2*fct['PP2']/10
tbp1['PP2'] = fct['PP2']/10

Epx['PP2'] = 200000
Epy['PP2'] = 200000
ebp1['PP2'] = 0.00455
sp0['PP2'] = 707
fpux['PP2'] = 910
fpuy['PP2'] = 910
dpx['PP2'] = 8
dpy['PP2'] = 8
rhopx['PP2'] = 0.00293
rhopy['PP2'] = 0

" ---------------------------------------------------- PP3 ------------------------------------------------------------"
fcp['PP3'] = 27.7
Ec['PP3'] = 10000*fcp['PP3']**(1/3)
Gc['PP3'] = Ec['PP3']/2
vc['PP3'] = 0
t['PP3'] = 287
fct['PP3'] = 0.3*fcp['PP3']**(2/3)
ec0['PP3'] = 1.92e-3
Dmax['PP3'] = 13

fsyx['PP3'] = 480
fsux['PP3'] = 640
Esx['PP3'] = 200000
esyx = fsyx['PP3']/Esx['PP3']
esux = 0.091
Eshx['PP3'] = (fsux['PP3']-fsyx['PP3'])/(esux-esyx)
dx['PP3'] = 11.3
sx['PP3'] = 108
rhox['PP3'] = 0.00647

fsyy['PP3'] = 480
fsuy['PP3'] = 640
Esy['PP3'] = 200000
esyy = fsyy['PP3']/Esy['PP3']
esuy = 0.091
Eshy['PP3'] = (fsuy['PP3']-fsyy['PP3'])/(esuy-esyy)
dy['PP3'] = 11.3
sy['PP3'] = 108
rhoy['PP3'] = 0.00647

tb0['PP3'] = 2*fct['PP3']
tb1['PP3'] = fct['PP3']
tbp0['PP3'] = 2*fct['PP3']/100
tbp1['PP3'] = fct['PP3']/100

Epx['PP3'] = 200000
Epy['PP3'] = 200000
ebp1['PP3'] = 0.00455
sp0['PP3'] = 750
fpux['PP3'] = 910
fpuy['PP3'] = 910
dpx['PP3'] = 30
dpy['PP3'] = 30
rhopx['PP3'] = 0.00586
rhopy['PP3'] = 0

" ---------------------------------------------------- SE1 ------------------------------------------------------------"
fcp['SE1'] = 42.5
Ec['SE1'] = 10000*fcp['SE1']**(1/3)
Gc['SE1'] = Ec['SE1']/2
vc['SE1'] = 0
t['SE1'] = 287
fct['SE1'] = 0.3*fcp['SE1']**(2/3)
ec0['SE1'] = 2.54e-3
Dmax['SE1'] = 10

fsyx['SE1'] = 492
fsux['SE1'] = 640
Esx['SE1'] = 200000
esyx = fsyx['SE1']/Esx['SE1']
esux = 0.1
Eshx['SE1'] = (fsux['SE1']-fsyx['SE1'])/(esux-esyx)
dx['SE1'] = 19.5
sx['SE1'] = 72
rhox['SE1'] = 0.0293

fsyy['SE1'] = 479
fsuy['SE1'] = 623
Esy['SE1'] = 200000
esyy = fsyy['SE1']/Esy['SE1']
esuy = 0.1
Eshy['SE1'] = (fsuy['SE1']-fsyy['SE1'])/(esuy-esyy)
dy['SE1'] = 11.3
sy['SE1'] = 72
rhoy['SE1'] = 0.00978

tb0['SE1'] = 2*fct['SE1']
tb1['SE1'] = fct['SE1']
tbp0['SE1'] = 2*fct['SE1']/100
tbp1['SE1'] = fct['SE1']/100

Epx['SE1'] = 200000
Epy['SE1'] = 200000
ebp1['SE1'] = 0.00455
sp0['SE1'] = 750
fpux['SE1'] = 910
fpuy['SE1'] = 910
dpx['SE1'] = 30
dpy['SE1'] = 30
rhopx['SE1'] = 0
rhopy['SE1'] = 0

" ---------------------------------------------------- SE6 ------------------------------------------------------------"
fcp['SE6'] = 40
Ec['SE6'] = 10000*fcp['SE6']**(1/3)
Gc['SE6'] = Ec['SE6']/2
vc['SE6'] = 0
t['SE6'] = 287
fct['SE6'] = 0.3*fcp['SE6']**(2/3)
ec0['SE6'] = 2.5e-3
Dmax['SE6'] = 10

fsyx['SE6'] = 492
fsux['SE6'] = 640
Esx['SE6'] = 200000
esyx = fsyx['SE6']/Esx['SE6']
esux = 0.1
Eshx['SE6'] = (fsux['SE6']-fsyx['SE6'])/(esux-esyx)
dx['SE6'] = 19.5
sx['SE6'] = 72
rhox['SE6'] = 0.0293

fsyy['SE6'] = 479
fsuy['SE6'] = 623
Esy['SE6'] = 200000
esyy = fsyy['SE6']/Esy['SE6']
esuy = 0.1
Eshy['SE6'] = (fsuy['SE6']-fsyy['SE6'])/(esuy-esyy)
dy['SE6'] = 11.3
sy['SE6'] = 72
rhoy['SE6'] = 0.00326

tb0['SE6'] = 2*fct['SE6']
tb1['SE6'] = fct['SE6']
tbp0['SE6'] = 2*fct['SE6']/100
tbp1['SE6'] = fct['SE6']/100

Epx['SE6'] = 200000
Epy['SE6'] = 200000
ebp1['SE6'] = 0.00455
sp0['SE6'] = 750
fpux['SE6'] = 910
fpuy['SE6'] = 910
dpx['SE6'] = 30
dpy['SE6'] = 30
rhopx['SE6'] = 0
rhopy['SE6'] = 0

" ---------------------------------------------------- HB3 ------------------------------------------------------------"
fcp['HB3'] = 66.8
Ec['HB3'] = 10000*fcp['HB3']**(1/3)
Gc['HB3'] = Ec['HB3']/2
vc['HB3'] = 0
t['HB3'] = 178
fct['HB3'] = 0.3*fcp['HB3']**(2/3)
ec0['HB3'] = 2.8e-3
Dmax['HB3'] = 10 # Guess

fsyx['HB3'] = 446
fsux['HB3'] = 583
Esx['HB3'] = 200000
esyx = fsyx['HB3']/Esx['HB3']
esux = 0.1
Eshx['HB3'] = (fsux['HB3']-fsyx['HB3'])/(esux-esyx)
dx['HB3'] = 19.5
sx['HB3'] = 200
rhox['HB3'] = 0.0171

fsyy['HB3'] = 450
fsuy['HB3'] = 579
Esy['HB3'] = 200000
esyy = fsyy['HB3']/Esy['HB3']
esuy = 0.1
Eshy['HB3'] = (fsuy['HB3']-fsyy['HB3'])/(esuy-esyy)
dy['HB3'] = 11.3
sy['HB3'] = 200
rhoy['HB3'] = 0.0057

tb0['HB3'] = 2*fct['HB3']
tb1['HB3'] = fct['HB3']
tbp0['HB3'] = 2*fct['HB3']/100
tbp1['HB3'] = fct['HB3']/100

Epx['HB3'] = 200000
Epy['HB3'] = 200000
ebp1['HB3'] = 0.00455
sp0['HB3'] = 750
fpux['HB3'] = 910
fpuy['HB3'] = 910
dpx['HB3'] = 30
dpy['HB3'] = 30
rhopx['HB3'] = 0
rhopy['HB3'] = 0

" ---------------------------------------------------- HB4 ------------------------------------------------------------"
fcp['HB4'] = 62.9
Ec['HB4'] = 10000*fcp['HB4']**(1/3)
Gc['HB4'] = Ec['HB4']/2
vc['HB4'] = 0
t['HB4'] = 178
fct['HB4'] = 0.3*fcp['HB4']**(2/3)
ec0['HB4'] = 2.7e-3
Dmax['HB4'] = 10 # Guess

fsyx['HB4'] = 470
fsux['HB4'] = 629
Esx['HB4'] = 200000
esyx = fsyx['HB4']/Esx['HB4']
esux = 0.1
Eshx['HB4'] = (fsux['HB4']-fsyx['HB4'])/(esux-esyx)
dx['HB4'] = 25.2
sx['HB4'] = 200
rhox['HB4'] = 0.0284

fsyy['HB4'] = 450
fsuy['HB4'] = 579
Esy['HB4'] = 200000
esyy = fsyy['HB4']/Esy['HB4']
esuy = 0.1
Eshy['HB4'] = (fsuy['HB4']-fsyy['HB4'])/(esuy-esyy)
dy['HB4'] = 11.3
sy['HB4'] = 200
rhoy['HB4'] = 0.0057

tb0['HB4'] = 2*fct['HB4']
tb1['HB4'] = fct['HB4']
tbp0['HB4'] = 2*fct['HB4']/100
tbp1['HB4'] = fct['HB4']/100

Epx['HB4'] = 200000
Epy['HB4'] = 200000
ebp1['HB4'] = 0.00455
sp0['HB4'] = 750
fpux['HB4'] = 910
fpuy['HB4'] = 910
dpx['HB4'] = 30
dpy['HB4'] = 30
rhopx['HB4'] = 0
rhopy['HB4'] = 0

" ---------------------------------------------------- VB1 ------------------------------------------------------------"
fcp['VB1'] = 98.2
Ec['VB1'] = 10000*fcp['VB1']**(1/3)
Gc['VB1'] = Ec['VB1']/2
vc['VB1'] = 0
t['VB1'] = 178
fct['VB1'] = 0.3*fcp['VB1']**(2/3)
ec0['VB1'] = 2.5e-3
Dmax['VB1'] = 10 # Guess

fsyx['VB1'] = 409
fsux['VB1'] = 534
Esx['VB1'] = 200000
esyx = fsyx['VB1']/Esx['VB1']
esux = 0.1
Eshx['VB1'] = (fsux['VB1']-fsyx['VB1'])/(esux-esyx)
dx['VB1'] = 16
sx['VB1'] = 189
rhox['VB1'] = 0.0228

fsyy['VB1'] = 445
fsuy['VB1'] = 579
Esy['VB1'] = 200000
esyy = fsyy['VB1']/Esy['VB1']
esuy = 0.1
Eshy['VB1'] = (fsuy['VB1']-fsyy['VB1'])/(esuy-esyy)
dy['VB1'] = 11.3
sy['VB1'] = 189
rhoy['VB1'] = 0.0114

tb0['VB1'] = 2*fct['VB1']
tb1['VB1'] = fct['VB1']
tbp0['VB1'] = 2*fct['VB1']/100
tbp1['VB1'] = fct['VB1']/100

Epx['VB1'] = 200000
Epy['VB1'] = 200000
ebp1['VB1'] = 0.00455
sp0['VB1'] = 750
fpux['VB1'] = 910
fpuy['VB1'] = 910
dpx['VB1'] = 30
dpy['VB1'] = 30
rhopx['VB1'] = 0
rhopy['VB1'] = 0

" ---------------------------------------------------- VB2 ------------------------------------------------------------"
fcp['VB2'] = 97.6
Ec['VB2'] = 10000*fcp['VB2']**(1/3)
Gc['VB2'] = Ec['VB2']/2
vc['VB2'] = 0
t['VB2'] = 178
fct['VB2'] = 0.3*fcp['VB2']**(2/3)
ec0['VB2'] = 2.45e-3
Dmax['VB2'] = 10 # Guess

fsyx['VB2'] = 455
fsux['VB2'] = 608
Esx['VB2'] = 200000
esyx = fsyx['VB2']/Esx['VB2']
esux = 0.1
Eshx['VB2'] = (fsux['VB2']-fsyx['VB2'])/(esux-esyx)
dx['VB2'] = 19.5
sx['VB2'] = 189
rhox['VB2'] = 0.0342

fsyy['VB2'] = 445
fsuy['VB2'] = 579
Esy['VB2'] = 200000
esyy = fsyy['VB2']/Esy['VB2']
esuy = 0.1
Eshy['VB2'] = (fsuy['VB2']-fsyy['VB2'])/(esuy-esyy)
dy['VB2'] = 11.3
sy['VB2'] = 189
rhoy['VB2'] = 0.0114

tb0['VB2'] = 2*fct['VB2']
tb1['VB2'] = fct['VB2']
tbp0['VB2'] = 2*fct['VB2']/100
tbp1['VB2'] = fct['VB2']/100

Epx['VB2'] = 200000
Epy['VB2'] = 200000
ebp1['VB2'] = 0.00455
sp0['VB2'] = 750
fpux['VB2'] = 910
fpuy['VB2'] = 910
dpx['VB2'] = 30
dpy['VB2'] = 30
rhopx['VB2'] = 0
rhopy['VB2'] = 0

" ---------------------------------------------------- VB3 ------------------------------------------------------------"
fcp['VB3'] = 102.3
Ec['VB3'] = 10000*fcp['VB3']**(1/3)
Gc['VB3'] = Ec['VB3']/2
vc['VB3'] = 0
t['VB3'] = 178
fct['VB3'] = 0.3*fcp['VB3']**(2/3)
ec0['VB3'] = 2.35e-3
Dmax['VB3'] = 10 # Guess

fsyx['VB3'] = 470
fsux['VB3'] = 606
Esx['VB3'] = 200000
esyx = fsyx['VB3']/Esx['VB3']
esux = 0.1
Eshx['VB3'] = (fsux['VB3']-fsyx['VB3'])/(esux-esyx)
dx['VB3'] = 25.2
sx['VB3'] = 189
rhox['VB3'] = 0.057

fsyy['VB3'] = 445
fsuy['VB3'] = 579
Esy['VB3'] = 200000
esyy = fsyy['VB3']/Esy['VB3']
esuy = 0.1
Eshy['VB3'] = (fsuy['VB3']-fsyy['VB3'])/(esuy-esyy)
dy['VB3'] = 11.3
sy['VB3'] = 189
rhoy['VB3'] = 0.0114

tb0['VB3'] = 2*fct['VB3']
tb1['VB3'] = fct['VB3']
tbp0['VB3'] = 2*fct['VB3']/100
tbp1['VB3'] = fct['VB3']/100

Epx['VB3'] = 200000
Epy['VB3'] = 200000
ebp1['VB3'] = 0.00455
sp0['VB3'] = 750
fpux['VB3'] = 910
fpuy['VB3'] = 910
dpx['VB3'] = 30
dpy['VB3'] = 30
rhopx['VB3'] = 0
rhopy['VB3'] = 0

" ---------------------------------------------------- PHS2 ------------------------------------------------------------"
fcp['PHS2'] = 66.1
Ec['PHS2'] = 10000*fcp['PHS2']**(1/3)
Gc['PHS2'] = Ec['PHS2']/2
vc['PHS2'] = 0
t['PHS2'] = 70
fct['PHS2'] = 0.3*fcp['PHS2']**(2/3)
ec0['PHS2'] = 2.48e-3
Dmax['PHS2'] = 10 # Guess

fsyx['PHS2'] = 606
fsux['PHS2'] = 606*1.2
Esx['PHS2'] = 200000
esyx = fsyx['PHS2']/Esx['PHS2']
esux = 0.05
Eshx['PHS2'] = (fsux['PHS2']-fsyx['PHS2'])/(esux-esyx)
dx['PHS2'] = 8
sx['PHS2'] = 45
rhox['PHS2'] = 0.0323

fsyy['PHS2'] = 521
fsuy['PHS2'] = 521*1.2
Esy['PHS2'] = 200000
esyy = fsyy['PHS2']/Esy['PHS2']
esuy = 0.05
Eshy['PHS2'] = (fsuy['PHS2']-fsyy['PHS2'])/(esuy-esyy)
dy['PHS2'] = 5.72
sy['PHS2'] = 45
rhoy['PHS2'] = 0.0041

tb0['PHS2'] = 2*fct['PHS2']
tb1['PHS2'] = fct['PHS2']
tbp0['PHS2'] = 2*fct['PHS2']/100
tbp1['PHS2'] = fct['PHS2']/100

Epx['PHS2'] = 200000
Epy['PHS2'] = 200000
ebp1['PHS2'] = 0.00455
sp0['PHS2'] = 750
fpux['PHS2'] = 910
fpuy['PHS2'] = 910
dpx['PHS2'] = 30
dpy['PHS2'] = 30
rhopx['PHS2'] = 0
rhopy['PHS2'] = 0

" ---------------------------------------------------- PHS3 ------------------------------------------------------------"
fcp['PHS3'] = 58.4
Ec['PHS3'] = 10000*fcp['PHS3']**(1/3)
Gc['PHS3'] = Ec['PHS3']/2
vc['PHS3'] = 0
t['PHS3'] = 70
fct['PHS3'] = 0.3*fcp['PHS3']**(2/3)
ec0['PHS3'] = 2.44e-3
Dmax['PHS3'] = 10 # Guess

fsyx['PHS3'] = 606
fsux['PHS3'] = 606*1.2
Esx['PHS3'] = 200000
esyx = fsyx['PHS3']/Esx['PHS3']
esux = 0.05
Eshx['PHS3'] = (fsux['PHS3']-fsyx['PHS3'])/(esux-esyx)
dx['PHS3'] = 8
sx['PHS3'] = 45
rhox['PHS3'] = 0.0323

fsyy['PHS3'] = 521
fsuy['PHS3'] = 521*1.2
Esy['PHS3'] = 200000
esyy = fsyy['PHS3']/Esy['PHS3']
esuy = 0.05
Eshy['PHS3'] = (fsuy['PHS3']-fsyy['PHS3'])/(esuy-esyy)
dy['PHS3'] = 5.72
sy['PHS3'] = 45
rhoy['PHS3'] = 0.0082

tb0['PHS3'] = 2*fct['PHS3']
tb1['PHS3'] = fct['PHS3']
tbp0['PHS3'] = 2*fct['PHS3']/100
tbp1['PHS3'] = fct['PHS3']/100

Epx['PHS3'] = 200000
Epy['PHS3'] = 200000
ebp1['PHS3'] = 0.00455
sp0['PHS3'] = 750
fpux['PHS3'] = 910
fpuy['PHS3'] = 910
dpx['PHS3'] = 30
dpy['PHS3'] = 30
rhopx['PHS3'] = 0
rhopy['PHS3'] = 0

" ---------------------------------------------------- PHS4 ------------------------------------------------------------"
fcp['PHS4'] = 68.5
Ec['PHS4'] = 10000*fcp['PHS4']**(1/3)
Gc['PHS4'] = Ec['PHS4']/2
vc['PHS4'] = 0
t['PHS4'] = 70
fct['PHS4'] = 0.3*fcp['PHS4']**(2/3)
ec0['PHS4'] = 2.60e-3
Dmax['PHS4'] = 10 # Guess

fsyx['PHS4'] = 606
fsux['PHS4'] = 606*1.2
Esx['PHS4'] = 200000
esyx = fsyx['PHS4']/Esx['PHS4']
esux = 0.05
Eshx['PHS4'] = (fsux['PHS4']-fsyx['PHS4'])/(esux-esyx)
dx['PHS4'] = 8
sx['PHS4'] = 45
rhox['PHS4'] = 0.0323

fsyy['PHS4'] = 521
fsuy['PHS4'] = 521*1.2
Esy['PHS4'] = 200000
esyy = fsyy['PHS4']/Esy['PHS4']
esuy = 0.05
Eshy['PHS4'] = (fsuy['PHS4']-fsyy['PHS4'])/(esuy-esyy)
dy['PHS4'] = 5.72
sy['PHS4'] = 45
rhoy['PHS4'] = 0.0082

tb0['PHS4'] = 2*fct['PHS4']
tb1['PHS4'] = fct['PHS4']
tbp0['PHS4'] = 2*fct['PHS4']/100
tbp1['PHS4'] = fct['PHS4']/100

Epx['PHS4'] = 200000
Epy['PHS4'] = 200000
ebp1['PHS4'] = 0.00455
sp0['PHS4'] = 750
fpux['PHS4'] = 910
fpuy['PHS4'] = 910
dpx['PHS4'] = 30
dpy['PHS4'] = 30
rhopx['PHS4'] = 0
rhopy['PHS4'] = 0

" ---------------------------------------------------- PHS5 ------------------------------------------------------------"
fcp['PHS5'] = 52.1
Ec['PHS5'] = 10000*fcp['PHS5']**(1/3)
Gc['PHS5'] = Ec['PHS5']/2
vc['PHS5'] = 0
t['PHS5'] = 70
fct['PHS5'] = 0.3*fcp['PHS5']**(2/3)
ec0['PHS5'] = 2.58e-3
Dmax['PHS5'] = 10 # Guess

fsyx['PHS5'] = 606
fsux['PHS5'] = 606*1.2
Esx['PHS5'] = 200000
esyx = fsyx['PHS5']/Esx['PHS5']
esux = 0.05
Eshx['PHS5'] = (fsux['PHS5']-fsyx['PHS5'])/(esux-esyx)
dx['PHS5'] = 8
sx['PHS5'] = 45
rhox['PHS5'] = 0.0323

fsyy['PHS5'] = 521
fsuy['PHS5'] = 521*1.2
Esy['PHS5'] = 200000
esyy = fsyy['PHS5']/Esy['PHS5']
esuy = 0.05
Eshy['PHS5'] = (fsuy['PHS5']-fsyy['PHS5'])/(esuy-esyy)
dy['PHS5'] = 5.72
sy['PHS5'] = 45
rhoy['PHS5'] = 0.0041

tb0['PHS5'] = 2*fct['PHS5']
tb1['PHS5'] = fct['PHS5']
tbp0['PHS5'] = 2*fct['PHS5']/100
tbp1['PHS5'] = fct['PHS5']/100

Epx['PHS5'] = 200000
Epy['PHS5'] = 200000
ebp1['PHS5'] = 0.00455
sp0['PHS5'] = 750
fpux['PHS5'] = 910
fpuy['PHS5'] = 910
dpx['PHS5'] = 30
dpy['PHS5'] = 30
rhopx['PHS5'] = 0
rhopy['PHS5'] = 0

" ---------------------------------------------------- PHS6 ------------------------------------------------------------"
fcp['PHS6'] = 49.7
Ec['PHS6'] = 10000*fcp['PHS6']**(1/3)
Gc['PHS6'] = Ec['PHS6']/2
vc['PHS6'] = 0
t['PHS6'] = 70
fct['PHS6'] = 0.3*fcp['PHS6']**(2/3)
ec0['PHS6'] = 2.25e-3
Dmax['PHS6'] = 10 # Guess

fsyx['PHS6'] = 606
fsux['PHS6'] = 606*1.2
Esx['PHS6'] = 200000
esyx = fsyx['PHS6']/Esx['PHS6']
esux = 0.05
Eshx['PHS6'] = (fsux['PHS6']-fsyx['PHS6'])/(esux-esyx)
dx['PHS6'] = 8
sx['PHS6'] = 45
rhox['PHS6'] = 0.0323

fsyy['PHS6'] = 521
fsuy['PHS6'] = 521*1.2
Esy['PHS6'] = 200000
esyy = fsyy['PHS6']/Esy['PHS6']
esuy = 0.05
Eshy['PHS6'] = (fsuy['PHS6']-fsyy['PHS6'])/(esuy-esyy)
dy['PHS6'] = 5.72
sy['PHS6'] = 45
rhoy['PHS6'] = 0.0041

tb0['PHS6'] = 2*fct['PHS6']
tb1['PHS6'] = fct['PHS6']
tbp0['PHS6'] = 2*fct['PHS6']/100
tbp1['PHS6'] = fct['PHS6']/100

Epx['PHS6'] = 200000
Epy['PHS6'] = 200000
ebp1['PHS6'] = 0.00455
sp0['PHS6'] = 750
fpux['PHS6'] = 910
fpuy['PHS6'] = 910
dpx['PHS6'] = 30
dpy['PHS6'] = 30
rhopx['PHS6'] = 0
rhopy['PHS6'] = 0

" ---------------------------------------------------- PHS7 ------------------------------------------------------------"
fcp['PHS7'] = 53.6
Ec['PHS7'] = 10000*fcp['PHS7']**(1/3)
Gc['PHS7'] = Ec['PHS7']/2
vc['PHS7'] = 0
t['PHS7'] = 70
fct['PHS7'] = 0.3*fcp['PHS7']**(2/3)
ec0['PHS7'] = 2.10e-3
Dmax['PHS7'] = 10 # Guess

fsyx['PHS7'] = 606
fsux['PHS7'] = 606*1.2
Esx['PHS7'] = 200000
esyx = fsyx['PHS7']/Esx['PHS7']
esux = 0.05
Eshx['PHS7'] = (fsux['PHS7']-fsyx['PHS7'])/(esux-esyx)
dx['PHS7'] = 8
sx['PHS7'] = 45
rhox['PHS7'] = 0.0323

fsyy['PHS7'] = 521
fsuy['PHS7'] = 521*1.2
Esy['PHS7'] = 200000
esyy = fsyy['PHS7']/Esy['PHS7']
esuy = 0.05
Eshy['PHS7'] = (fsuy['PHS7']-fsyy['PHS7'])/(esuy-esyy)
dy['PHS7'] = 5.72
sy['PHS7'] = 45
rhoy['PHS7'] = 0.0082

tb0['PHS7'] = 2*fct['PHS7']
tb1['PHS7'] = fct['PHS7']
tbp0['PHS7'] = 2*fct['PHS7']/100
tbp1['PHS7'] = fct['PHS7']/100

Epx['PHS7'] = 200000
Epy['PHS7'] = 200000
ebp1['PHS7'] = 0.00455
sp0['PHS7'] = 750
fpux['PHS7'] = 910
fpuy['PHS7'] = 910
dpx['PHS7'] = 30
dpy['PHS7'] = 30
rhopx['PHS7'] = 0
rhopy['PHS7'] = 0

" ---------------------------------------------------- SL1 ------------------------------------------------------------"
fcp['SL1'] = 30.1
Ec['SL1'] = 10000*fcp['SL1']**(1/3)
Gc['SL1'] = Ec['SL1']/2
vc['SL1'] = 0
t['SL1'] = 350
fct['SL1'] = 0.3*fcp['SL1']**(2/3)
ec0['SL1'] = 2e-3
Dmax['SL1'] = 16

fsyx['SL1'] = 523.6
fsux['SL1'] = 626.7
Esx['SL1'] = 201600
esyx = fsyx['SL1']/Esx['SL1']
esux = 0.112
Eshx['SL1'] = (fsux['SL1']-fsyx['SL1'])/(esux-esyx)
dx['SL1'] = 20
sx['SL1'] = 200
rhox['SL1'] = 0.009

fsyy['SL1'] = 518.5
fsuy['SL1'] = 624.6
Esy['SL1'] = 192100
esyy = fsyy['SL1']/Esy['SL1']
esuy = 0.0997
Eshy['SL1'] = (fsuy['SL1']-fsyy['SL1'])/(esuy-esyy)
dy['SL1'] = 8
sy['SL1'] = 200
rhoy['SL1'] = 0.0022

tb0['SL1'] = 2*fct['SL1']
tb1['SL1'] = fct['SL1']
tbp0['SL1'] = 2*fct['SL1']/100
tbp1['SL1'] = fct['SL1']/100

Epx['SL1'] = 200000
Epy['SL1'] = 200000
ebp1['SL1'] = 0.00455
sp0['SL1'] = 750
fpux['SL1'] = 910
fpuy['SL1'] = 910
dpx['SL1'] = 30
dpy['SL1'] = 30
rhopx['SL1'] = 0
rhopy['SL1'] = 0

" ---------------------------------------------------- SL3 ------------------------------------------------------------"
fcp['SL3'] = 37
Ec['SL3'] = 10000*fcp['SL3']**(1/3)
Gc['SL3'] = Ec['SL3']/2
vc['SL3'] = 0
t['SL3'] = 350
fct['SL3'] = 0.3*fcp['SL3']**(2/3)
ec0['SL3'] = 2e-3
Dmax['SL3'] = 16

fsyx['SL3'] = 523.6
fsux['SL3'] = 626.7
Esx['SL3'] = 201600
esyx = fsyx['SL3']/Esx['SL3']
esux = 0.112
Eshx['SL3'] = (fsux['SL3']-fsyx['SL3'])/(esux-esyx)
dx['SL3'] = 20
sx['SL3'] = 200
rhox['SL3'] = 0.009

fsyy['SL3'] = 518.5
fsuy['SL3'] = 624.6
Esy['SL3'] = 192100
esyy = fsyy['SL3']/Esy['SL3']
esuy = 0.0997
Eshy['SL3'] = (fsuy['SL3']-fsyy['SL3'])/(esuy-esyy)
dy['SL3'] = 8
sy['SL3'] = 200
rhoy['SL3'] = 0.0022

tb0['SL3'] = 2*fct['SL3']
tb1['SL3'] = fct['SL3']
tbp0['SL3'] = 2*fct['SL3']/100
tbp1['SL3'] = fct['SL3']/100

Epx['SL3'] = 200000
Epy['SL3'] = 200000
ebp1['SL3'] = 0.00455
sp0['SL3'] = 750
fpux['SL3'] = 910
fpuy['SL3'] = 910
dpx['SL3'] = 30
dpy['SL3'] = 30
rhopx['SL3'] = 0
rhopy['SL3'] = 0

" ---------------------------------------------------- SL5 ------------------------------------------------------------"
fcp['SL5'] = 38.1
Ec['SL5'] = 10000*fcp['SL5']**(1/3)
Gc['SL5'] = Ec['SL5']/2
vc['SL5'] = 0
t['SL5'] = 350
fct['SL5'] = 0.3*fcp['SL5']**(2/3)
ec0['SL5'] = 2e-3
Dmax['SL5'] = 16

fsyx['SL5'] = 523.6
fsux['SL5'] = 626.7
Esx['SL5'] = 201600
esyx = fsyx['SL5']/Esx['SL5']
esux = 0.112
Eshx['SL5'] = (fsux['SL5']-fsyx['SL5'])/(esux-esyx)
dx['SL5'] = 20
sx['SL5'] = 200
rhox['SL5'] = 0.009

fsyy['SL5'] = 518.5
fsuy['SL5'] = 624.6
Esy['SL5'] = 192100
esyy = fsyy['SL5']/Esy['SL5']
esuy = 0.0997
Eshy['SL5'] = (fsuy['SL5']-fsyy['SL5'])/(esuy-esyy)
dy['SL5'] = 8
sy['SL5'] = 200
rhoy['SL5'] = 0.0022

tb0['SL5'] = 2*fct['SL5']
tb1['SL5'] = fct['SL5']
tbp0['SL5'] = 2*fct['SL5']/100
tbp1['SL5'] = fct['SL5']/100

Epx['SL5'] = 200000
Epy['SL5'] = 200000
ebp1['SL5'] = 0.00455
sp0['SL5'] = 750
fpux['SL5'] = 910
fpuy['SL5'] = 910
dpx['SL5'] = 30
dpy['SL5'] = 30
rhopx['SL5'] = 0
rhopy['SL5'] = 0

" ---------------------------------------------------- PT1 ------------------------------------------------------------"
fcp['PT1'] = 33.5
Ec['PT1'] = 10000*fcp['PT1']**(1/3)
Gc['PT1'] = Ec['PT1']/2
vc['PT1'] = 0
t['PT1'] = 350
fct['PT1'] = 0.3*fcp['PT1']**(2/3)
ec0['PT1'] = 2e-3
Dmax['PT1'] = 16

fsyx['PT1'] = 505
fsux['PT1'] = 608
Esx['PT1'] = 199000
esyx = fsyx['PT1']/Esx['PT1']
esux = 0.125
Eshx['PT1'] = (fsux['PT1']-fsyx['PT1'])/(esux-esyx)
dx['PT1'] = 20
sx['PT1'] = 100
rhox['PT1'] = 0.0233

fsyy['PT1'] = 520
fsuy['PT1'] = 614
Esy['PT1'] = 201000
esyy = fsyy['PT1']/Esy['PT1']
esuy = 0.128
Eshy['PT1'] = (fsuy['PT1']-fsyy['PT1'])/(esuy-esyy)
dy['PT1'] = 14
sy['PT1'] = 100
rhoy['PT1'] = 0.0086

tb0['PT1'] = 2*fct['PT1']
tb1['PT1'] = fct['PT1']
tbp0['PT1'] = 2*fct['PT1']/100
tbp1['PT1'] = fct['PT1']/100

Epx['PT1'] = 200000
Epy['PT1'] = 200000
ebp1['PT1'] = 0.00455
sp0['PT1'] = 750
fpux['PT1'] = 910
fpuy['PT1'] = 910
dpx['PT1'] = 30
dpy['PT1'] = 30
rhopx['PT1'] = 0
rhopy['PT1'] = 0
"----------------------------------------------------------------------------------------------------------------------"

"----------------------------------------------------------------------------------------------------------------------"

" Model Parameters"
target_stress = True
cm_klij = 4
cmcc_klij = 2
cmcs_klij = 2
cms_klij = 1
cmtn_klij = 1

doplot_main = True
"----------------------------------------------------------------------------------------------------------------------"

for j in range(len(Tests)):
    # 1 Material and Geometry

    MAT = [Ec[Tests[j]], vc[Tests[j]], fcp[Tests[j]], fct[Tests[j]], ec0[Tests[j]], 0*fct[Tests[j]]/Ec[Tests[j]],
           Dmax[Tests[j]], Esx[Tests[j]], Eshx[Tests[j]], fsyx[Tests[j]], fsux[Tests[j]], Esy[Tests[j]], Eshy[Tests[j]],
           fsyy[Tests[j]], fsuy[Tests[j]], tb0[Tests[j]], tb1[Tests[j]], Epx[Tests[j]], Epy[Tests[j]], tbp0[Tests[j]],
           tbp1[Tests[j]], ebp1[Tests[j]], fpux[Tests[j]], fpuy[Tests[j]]]
    GEOM = [rhox[Tests[j]], rhoy[Tests[j]], dx[Tests[j]], dy[Tests[j]], sx[Tests[j]], sy[Tests[j]], rhopx[Tests[j]],
            rhopy[Tests[j]], dpx[Tests[j]], dpy[Tests[j]]]

    # 2 Load Test Results
    from numpy import loadtxt
    if Tests[j] == 'PT1':
        pathi = r'C:\Users\naesboma\00_an\04_UHBB\06_Papers\06_Numerical_Modelling\02_Modell\01_LUSET_abe'
        from numpy import loadtxt
        data = loadtxt(os.path.join(pathi, 'Data_PT1.txt'), comments="#", delimiter=",", unpack=False)
        istart = 0
        iend = 151
        gxy_test = data[:, 5] / 1000
        txy_test = data[:, 2]
    else:
        import csv
        path_data = path +'/Exp_Data' + '/' +'Data_' + Tests[j] + '.csv'
        gxy_test = []
        txy_test = []
        with open(path_data, mode='r') as file:
            csvFile = csv.reader(file)
            for row in csvFile:
                gxy_test.append(float(row[0])/1000)
                txy_test.append(float(row[1]))
    # print(txy_test)
    # 3 Get Vectors of Results and Stresses for calculating
    steps = 100
    if Tests[j] == 'VA0':
        txy_model = np.linspace(.1, 3.5, steps)
        sx_model = np.linspace( 0, 0, steps)
        sy_model = np.linspace(0, 0, steps)
    elif Tests[j] == 'VA1':
        txy_model = np.linspace(.1, 7, steps)
        sx_model = np.linspace(0, 0, steps)
        sy_model = np.linspace(0, 0, steps)
    elif Tests[j] == 'VA2':
        txy_model = np.linspace(.1, 12, steps)
        sx_model = np.linspace(0, 0, steps)
        sy_model = np.linspace(0, 0, steps)
    elif Tests[j] == 'VA3':
        txy_model = np.linspace(.1, 18, steps)
        sx_model = np.linspace(0, 0, steps)
        sy_model = np.linspace(0, 0, steps)
    elif Tests[j] == 'VA4':
        txy_model = np.linspace(.1, 26, steps)
        sx_model = np.linspace(0, 0, steps)
        sy_model = np.linspace(0, 0, steps)
    elif Tests[j] == 'PV27' or Tests[j] == 'PV22':
        txy_model = np.linspace(.1, 8, steps)
        sx_model = np.linspace(0, 0, steps)
        sy_model = np.linspace(0, 0, steps)
    elif Tests[j] == 'PV10' or Tests[j] == 'PV19' or Tests[j] == 'PV20' or Tests[j] == 'PV21' or Tests[j] == 'PP1':
        txy_model = np.linspace(.1, 6, steps)
        sx_model = np.linspace(0, 0, steps)
        sy_model = np.linspace(0, 0, steps)
    elif Tests[j] == 'PP2':
        txy_model = np.linspace(.1, 6, steps)
        sx_model = np.linspace(-2.072, -2.072, steps)
        sy_model = np.linspace(0, 0, steps)
    elif Tests[j] == 'PP3':
        txy_model = np.linspace(.1, 6, steps)
        sx_model = np.linspace(-4.395, -4.395, steps)
        sy_model = np.linspace(0, 0, steps)
    elif Tests[j] == 'SE1':
        txy_model = np.linspace(.1, 8, steps)
        sx_model = np.linspace(0, 0, steps)
        sy_model = np.linspace(0, 0, steps)
    elif Tests[j] == 'SE6':
        txy_model = np.linspace(.1, 5, steps)
        sx_model = np.linspace(0, 0, steps)
        sy_model = np.linspace(0, 0, steps)
    elif Tests[j] == 'HB3' or Tests[j] == 'HB4':
        txy_model = np.linspace(.1, 7, steps)
        sx_model = np.linspace(0, 0, steps)
        sy_model = np.linspace(0, 0, steps)
    elif Tests[j] == 'VB1':
        txy_model = np.linspace(.1, 10, steps)
        sx_model = np.linspace(0, 0, steps)
        sy_model = np.linspace(0, 0, steps)
    elif Tests[j] == 'VB2':
        txy_model = np.linspace(.1, 12, steps)
        sx_model = np.linspace(0, 0, steps)
        sy_model = np.linspace(0, 0, steps)
    elif Tests[j] == 'VB3':
        txy_model = np.linspace(.1, 14, steps)
        sx_model = np.linspace(0, 0, steps)
        sy_model = np.linspace(0, 0, steps)
    elif Tests[j] == 'PHS2':
        txy_model = np.linspace(.1, 8, steps)
        sx_model = np.linspace(0, 0, steps)
        sy_model = np.linspace(0, 0, steps)
    elif Tests[j] == 'PHS3':
        txy_model = np.linspace(.1, 10, steps)
        sx_model = np.linspace(0, 0, steps)
        sy_model = np.linspace(0, 0, steps)
    elif Tests[j] == 'PHS4':
        txy_model = np.linspace(.1, 8, steps)
        sx_model = np.linspace(0.025, 2, steps)
        sy_model = np.linspace(0.025, 2, steps)
    elif Tests[j] == 'PHS5':
        txy_model = np.linspace(.1, 6, steps)
        sx_model = np.linspace(0.025, 1.5, steps)
        sy_model = np.linspace(0.025, 1.5, steps)
    elif Tests[j] == 'PHS6':
        txy_model = np.linspace(.1, 12, steps)
        sx_model = np.linspace(-0.025, -3, steps)
        sy_model = np.linspace(-0.025, -3, steps)
    elif Tests[j] == 'PHS7':
        txy_model = np.linspace(.1, 12, steps)
        sx_model = np.linspace(-0.025, -3, steps)
        sy_model = np.linspace(-0.025, -3, steps)
    elif Tests[j] == 'PT1':
        txy_model = np.linspace(.1, 6.6, steps)
        sx_model = np.linspace(0, 0, steps)
        sy_model = np.linspace(0, 0, steps)
    elif Tests[j] == 'SL1':
        txy_model = np.linspace(.1, 1.25, steps)*(fcp['SL1']**(1/2)/3)
        sx_model = np.linspace(0, 0, steps)
        sy_model = np.linspace(0, 0, steps)
        txy_test = [i*(fcp['SL1']**(1/2)/3) for i in txy_test]
    elif Tests[j] == 'SL3':
        txy_model = np.linspace(.1, 1.25, steps)*(fcp['SL3']**(1/2)/3)
        sy_model = np.linspace(0, 0, steps)
        txy_test = [i*(fcp['SL3']**(1/2)/3) for i in txy_test]
        path_data_sx = path + '/Exp_Data' + '/' + 'Data_' + Tests[j] + '_Nx' +'.csv'
        sx_model1 = [0]
        t_model1 = [0]
        with open(path_data_sx, mode='r') as file:
            csvFile = csv.reader(file)
            for row in csvFile:
                t_model1.append(float(row[0]))
                sx_model1.append(float(row[1])/(350))
        tmax = t_model1[-1]
        t_txy = np.linspace(0,tmax,steps)
        fsx = interpolate.interp1d(t_model1,sx_model1)
        sx_model = fsx(t_txy)


    ex_test = np.ones_like(gxy_test)/1000
    ey_test = np.zeros_like(gxy_test)/1000

    sx_test = np.ones_like(gxy_test)*sx_model[0]
    sy_test = np.ones_like(gxy_test)*sy_model[0]

    ex_model = np.zeros_like(txy_model)
    ey_model = np.zeros_like(txy_model)
    gxy_model = np.zeros_like(txy_model)

    # 4 Initiate Result Vectors
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
    model = np.zeros_like(ex_model)
    th = np.zeros_like(ex_model)
    thr = np.zeros_like(ex_model)
    thc = np.zeros_like(ex_model)
    fcs = np.zeros_like(ex_model)
    scnr = np.zeros_like(ex_model)
    tctnr = np.zeros_like(ex_model)
    tctndow = np.zeros_like(ex_model)
    dn = np.zeros_like(ex_model)
    dt = np.zeros_like(ex_model)
    srx = np.zeros_like(ex_model)
    sry = np.zeros_like(ex_model)

    # 5 Initiate Sig_prev
    sig_prev = stress(0, 3, MAT, GEOM, cmcc_klij, cmcs_klij, cms_klij, cmtn_klij)
    if Tests[j] == 'PT1':
        refind = 60
    else:
        refind = 2
    [a, sigma_prev, b] = Ableitungsfunktion(sig_prev, np.array([ex_test[refind], ey_test[refind], gxy_test[refind], 0.001, 0.001]),3)
    print(sigma_prev)
    start = time.time()

    # 6 Iterate over load vectors and get results from stress class
    contit = True
    for i in range(len(ex_model)):
        print('new_i')
        print(txy_model[i])
        contit_i = contit
        itcount = 0
        noconv = False
        # sig_prev.thr = pi / 4
        if i == 0:
            exi = a[0]
            eyi = a[1]
            gxyi = a[2]
        elif isnan(gxy_model[i - 1]):
            contit = False
            noconv = True
        else:
            exi = ex_model[i - 1]
            eyi = ey_model[i - 1]
            gxyi = gxy_model[i - 1]

        while contit_i:
            itcount += 1
            if i < 3:
                sig = stress(sig_prev, 3, MAT, GEOM, cmcc_klij, cmcs_klij, cms_klij, cmtn_klij)
            else:
                sig = stress(sig_prev, cm_klij, MAT, GEOM, cmcc_klij, cmcs_klij, cms_klij, cmtn_klij)
            [epsilon, sigma, Tanmat] = Ableitungsfunktion(sig, np.array([exi, eyi, gxyi, 0.001, 0.001]), 3)
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
                rrmin = 1
                if Tests[j] == 'PP3':
                    rrmax = 2
                else:
                    rrmax = 5
                rr = max(rrmax - i, rrmin)
                dsi = np.array([[sx_model[i] - sxi], [sy_model[i] - syi], [txy_model[i] - txyi]]) / rr
                # print(dsi)
                if np.max(abs(dsi)) < 0.001:
                    print('stress found')
                    contit_i = False
                else:
                    if abs(np.linalg.det(Tanmat)) > 0:
                        dei = np.linalg.inv(Tanmat) @ dsi
                    else:
                        dei = np.array([0, 0, 0])
                        contit_i = False
                        noconv = True
                    exi += float(dei[0])
                    eyi += float(dei[1])
                    gxyi += float(dei[2])

                    if abs(exi) > 0.1:
                        exi = exi/abs(exi)*0.1
                    if abs(eyi) > 0.1:
                        eyi = eyi/abs(eyi)*0.1
                    if abs(gxyi) > 0.1:
                        gxyi = gxyi/abs(gxyi)*0.1
            else:
                txy_model[i] = txyi
                sx_model[i] = sxi
                sy_model[i] = syi
                contit_i = False
            if itcount > 100:
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
            srx[i] = nan
            sry[i] = nan
            sc3[i] = nan
            srm[i] = nan
            wr[i] = nan
            submodel[i] = nan
            model[i] = nan
            th[i] = nan
            thr[i] = nan
            thc[i] = nan
            fcs[i] = nan
            scnr[i] = nan
            tctnr[i] = nan
            tctndow[i] = nan
            dn[i] = nan
            dt[i] = nan
        else:
            ex_model[i] = exi
            ey_model[i] = eyi
            gxy_model[i] = gxyi
            print('gxy = ', gxyi)
            sig_prev = sig
            sigma_prev = sig
            e1[i] = epsilon[3]
            e3[i] = epsilon[4]

            ssrx[i] = sig.ssx.real
            sprx[i] = sig.spx.real
            ssry[i] = sig.ssy.real
            srx[i] = sig.srx.real
            sry[i] = sig.sry.real
            sc3[i] = sig.sc3.real
            srm[i] = sig.sr.real
            wr[i] = (ex_model[i].real - lbd * fct[Tests[j]] / Ec[Tests[j]] / 2) * srm[i]
            submodel[i] = sig.submodel
            model[i] = sig.cm_klij
            th[i] = sig.th.real
            thr[i] = sig.thr.real
            thc[i] = sig.thc.real
            fcs[i] = sig.fc_soft.real
            if sig.cm_klij == 4:
                scnr[i] = sig.scnr.real
                tctnr[i] = sig.tctnr.real
                tctndow[i] = sig.tctndow.real
                dn[i] = sig.dn.real
                dt[i] = sig.dt.real

        # Check if rupture in steel or CFRP
        isfail = 0
        if ssrx[i] >= fsux[Tests[j]]:
            print('steel failure')
            isfail = 1

        if sprx[i] >= fpux[Tests[j]]:
            print('CFRP failure')
            isfail = 1
        if abs(e3[i]) >= ec0[Tests[j]]:
            print('Concrete failure')
            isfail = 1

        if isfail == 1:
            ex_model = np.delete(ex_model, range(i, len(wr)), axis=0)
            ey_model = np.delete(ey_model, range(i, len(wr)), axis=0)
            gxy_model = np.delete(gxy_model, range(i, len(wr)), axis=0)
            sx_model = np.delete(sx_model, range(i, len(wr)), axis=0)
            sy_model = np.delete(sy_model, range(i, len(wr)), axis=0)
            txy_model = np.delete(txy_model, range(i, len(wr)), axis=0)
            dsx_dex = np.delete(dsx_dex, range(i, len(wr)), axis=0)
            dsx_dey = np.delete(dsx_dey, range(i, len(wr)), axis=0)
            dsx_dgxy = np.delete(dsx_dgxy, range(i, len(wr)), axis=0)
            dsy_dex = np.delete(dsy_dex, range(i, len(wr)), axis=0)
            dsy_dey = np.delete(dsy_dey, range(i, len(wr)), axis=0)
            dsy_dgxy = np.delete(dsy_dgxy, range(i, len(wr)), axis=0)
            dtxy_dex = np.delete(dtxy_dex, range(i, len(wr)), axis=0)
            dtxy_dey = np.delete(dtxy_dey, range(i, len(wr)), axis=0)
            dtxy_dgxy = np.delete(dtxy_dgxy, range(i, len(wr)), axis=0)
            e1 = np.delete(e1, range(i, len(wr)), axis=0)
            e3 = np.delete(e3, range(i, len(wr)), axis=0)
            ssrx = np.delete(ssrx, range(i, len(wr)), axis=0)
            sprx = np.delete(sprx, range(i, len(wr)), axis=0)
            ssry = np.delete(ssry, range(i, len(wr)), axis=0)
            srx = np.delete(srx, range(i, len(wr)), axis=0)
            sry = np.delete(sry, range(i, len(wr)), axis=0)
            sc3 = np.delete(sc3, range(i, len(wr)), axis=0)
            srm = np.delete(srm, range(i, len(wr)), axis=0)
            th = np.delete(th, range(i, len(wr)), axis=0)
            thr = np.delete(thr, range(i, len(wr)), axis=0)
            thc = np.delete(thc, range(i, len(wr)), axis=0)
            fcs = np.delete(fcs, range(i, len(wr)), axis=0)
            submodel = np.delete(submodel, range(i, len(wr)), axis=0)
            model = np.delete(model, range(i, len(wr)), axis=0)
            scnr = np.delete(scnr, range(i, len(wr)), axis=0)
            tctnr = np.delete(tctnr, range(i, len(wr)), axis=0)
            tctndow = np.delete(tctndow, range(i, len(wr)), axis=0)
            dn = np.delete(dn, range(i, len(wr)), axis=0)
            dt = np.delete(dt, range(i, len(wr)), axis=0)
            wr = np.delete(wr, range(i, len(wr)), axis=0)
            break
    end = time.time()
    print('Time Spent in calculation:')
    print(end - start)

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
        if i < 3:
            sx2[i] = sx_model[i]
            sy2[i] = sy_model[i]
            txy2[i] = txy_model[i]
        else:
            sx2[i] = sx2[i - 1] + (dsx_dex[i - 1] + dsx_dex[i]) / 2 * (ex_model[i] - ex_model[i - 1]) + (
                        dsx_dey[i - 1] + dsx_dey[i]) / 2 * (ey_model[i] - ey_model[i - 1]) + (
                                 dsx_dgxy[i - 1] + dsx_dgxy[i]) / 2 * (gxy_model[i] - gxy_model[i - 1])
            sy2[i] = sy2[i - 1] + (dsy_dex[i - 1] + dsy_dex[i]) / 2 * (ex_model[i] - ex_model[i - 1]) + (
                        dsy_dey[i - 1] + dsy_dey[i]) / 2 * (ey_model[i] - ey_model[i - 1]) + (
                                 dsy_dgxy[i - 1] + dsy_dgxy[i]) / 2 * (gxy_model[i] - gxy_model[i - 1])
            txy2[i] = txy2[i - 1] + (dtxy_dex[i - 1] + dtxy_dex[i]) / 2 * (ex_model[i] - ex_model[i - 1]) + (
                        dtxy_dey[i - 1] + dtxy_dey[i]) / 2 * (ey_model[i] - ey_model[i - 1]) + (
                                  dtxy_dgxy[i - 1] + dtxy_dgxy[i]) / 2 * (gxy_model[i] - gxy_model[i - 1])

    ex2 = np.zeros(len(ex_model) - 1)
    sxdev2 = np.zeros_like(ex2)
    for i in range(len(ex2)):
        ex2[i] = (ex_model[i] + ex_model[i + 1]) / 2
        sxdev2[i] = (sx_model[i + 1] - sx_model[i]) / (ex_model[i + 1] - ex_model[i])

    # ------------------------------------------------------------------------------------------------------------------
    # 3 Plot
    # ------------------------------------------------------------------------------------------------------------------
    # 3.0 Manipulate epsilon to be in microstrain
    ex_model = ex_model * 1000
    ey_model = ey_model * 1000
    gxy_model = gxy_model * 1000
    gxy_test = [i * 1000 for i in gxy_test]
    e1 = e1 * 1000
    e3 = e3 * 1000

    if doplot_main:
        # 3.1 General Plot Paramters
        plt.rcParams['font.size'] = 10
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['mathtext.fontset'] = 'cm'

        # 3.3 First Plot: sigma_x
        fig1, axs = plt.subplots(3, 3)
        fig1.set_figheight(9)
        fig1.set_figwidth(10)

        axs[0, 0].plot(ex_model, sx_model, 'k')
        axs[0, 0].plot(ex_model, sx2, 'b', linestyle='dotted')
        axs[0, 0].plot(ex_test, sx_test, 'r', linestyle='dashed')
        axs[0, 0].set_title('$\epsilon_x-\sigma_x$')
        axs[0, 0].set(ylabel="$\sigma_{x,Layer}$ [MPa]")

        axs[1, 0].plot(ex_model, dsx_dex, 'b')
        axs[1, 0].set(xlabel="$\epsilon$$_{x}$ [m$\epsilon$]")
        axs[1, 0].set(ylabel="d $\sigma_{x}$ / d $\epsilon_{x}$ [MPa]")

        axs[0, 1].plot(ey_model, sx_model, 'k')
        axs[0, 1].plot(ey_model, sx2, 'b', linestyle='dotted')
        axs[0, 1].plot(ey_test, sx_test, 'r', linestyle='dashed')
        axs[0, 1].set_title('$\epsilon_y-\sigma_x$')
        axs[0, 1].set(ylabel="$\sigma_{x,Layer}$ [MPa]")

        axs[1, 1].plot(ey_model, dsx_dey, 'b')
        axs[1, 1].set(xlabel="$\epsilon$$_{y}$ [m$\epsilon$]")
        axs[1, 1].set(ylabel="d $\sigma_{x}$ / d $\epsilon_{y}$ [MPa]")

        axs[0, 2].plot(gxy_model, sx_model, 'k')
        axs[0, 2].plot(gxy_model, sx2, 'b', linestyle='dotted')
        axs[0, 2].plot(gxy_test, sx_test, 'r', linestyle='dashed')
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
        axs[2, 0].legend(['$\epsilon_x$', '$\epsilon_y$', '$\gamma_{xy}$', '$\epsilon_1$', '$\epsilon_3$'])
        axs[2, 0].set_title('Strains')
        axs[2, 0].set(xlabel="$\epsilon_x$ [m$\epsilon$]")
        axs[2, 0].set(ylabel="$\epsilon$")

        axs[2, 1].set_title('Principal Dir')
        axs[2, 1].plot(ex_model, thr * 180 / pi, 'k')
        axs[2, 1].plot(ex_model, thc * 180 / pi, 'r')
        axs[2, 1].legend(['$\\theta_r$', '$\\theta_c$'])
        axs[2, 1].set(xlabel="$\epsilon_x$ [m$\epsilon$]")
        axs[2, 1].set(ylabel="$\\theta$ [°]")

        axs[2, 2].set_title('Submodel')
        axs[2, 2].plot(ex_model, submodel, 'k')
        axs[2, 2].set(xlabel="$\epsilon_x$ [m$\epsilon$]")
        axs[2, 2].set(ylabel="Model")

        fig1.tight_layout(pad=2.0)

        # 3.3 Second Plot: sigma_y
        fig2, axs2 = plt.subplots(3, 3)
        fig2.set_figheight(9)
        fig2.set_figwidth(10)

        axs2[0, 0].plot(ex_model, sy_model, 'k')
        axs2[0, 0].plot(ex_model, sy2, 'b', linestyle='dotted')
        axs2[0, 0].plot(ex_test, sy_test, 'r', linestyle='dashed')
        axs2[0, 0].set_title('$\epsilon_x-\sigma_y$')
        axs2[0, 0].set(ylabel="$\sigma_{y,Layer}$ [MPa]")

        axs2[1, 0].plot(ex_model, dsy_dex, 'b')
        axs2[1, 0].set(xlabel="$\epsilon$$_{x}$ [m$\epsilon$]")
        axs2[1, 0].set(ylabel="d $\sigma_{y}$ / d $\epsilon_{x}$ [MPa]")

        axs2[0, 1].plot(ey_model, sy_model, 'k')
        axs2[0, 1].plot(ey_model, sy2, 'b', linestyle='dotted')
        axs2[0, 1].plot(ey_test, sy_test, 'r', linestyle='dashed')
        axs2[0, 1].set_title('$\epsilon_y-\sigma_y$')
        axs2[0, 1].set(ylabel="$\sigma_{y,Layer}$ [MPa]")

        axs2[1, 1].plot(ey_model, dsy_dey, 'b')
        axs2[1, 1].set(xlabel="$\epsilon$$_{y}$ [m$\epsilon$]")
        axs2[1, 1].set(ylabel="d $\sigma_{y}$ / d $\epsilon_{y}$ [MPa]")

        axs2[0, 2].plot(gxy_model, sy_model, 'k')
        axs2[0, 2].plot(gxy_model, sy2, 'b', linestyle='dotted')
        axs2[0, 2].plot(gxy_test, sy_test, 'r', linestyle='dashed')
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
        axs2[2, 0].legend(['$\epsilon_x$', '$\epsilon_y$', '$\gamma_{xy}$', '$\epsilon_1$', '$\epsilon_3$'])
        axs2[2, 0].set_title('Strains')
        axs2[2, 0].set(xlabel="$\epsilon_y$ [m$\epsilon$]")
        axs2[2, 0].set(ylabel="$\epsilon$")

        axs2[2, 1].set_title('Principal Dir')
        axs2[2, 1].plot(ey_model, thr * 180 / pi, 'k')
        axs2[2, 1].plot(ey_model, thc * 180 / pi, 'r')
        axs2[2, 1].legend(['$\\theta_r$', '$\\theta_c$'])
        axs2[2, 1].set(xlabel="$\epsilon_y$ [m$\epsilon$]")
        axs2[2, 1].set(ylabel="$\\theta$ [°]")

        axs2[2, 2].set_title('Submodel')
        axs2[2, 2].plot(ey_model, submodel, 'k')
        axs2[2, 2].set(xlabel="$\epsilon_y$ [m$\epsilon$]")
        axs2[2, 2].set(ylabel="Model")

        fig2.tight_layout(pad=2.0)

        # 3.3 Third Plot: tau_xy
        fig3, axs3 = plt.subplots(3, 3)
        fig3.set_figheight(9)
        fig3.set_figwidth(10)

        axs3[0, 0].plot(ex_model, txy_model, 'k')
        axs3[0, 0].plot(ex_model, txy2, 'b', linestyle='dotted')
        axs3[0, 0].plot(ex_test, txy_test, 'r', linestyle='dashed')
        axs3[0, 0].set_title('$\epsilon_x-\\tau_{xy}$')
        axs3[0, 0].set(ylabel="$\\tau_{xy,Layer}$ [MPa]")

        axs3[1, 0].plot(ex_model, dtxy_dex, 'b')
        axs3[1, 0].set(xlabel="$\epsilon$$_{x}$ [m$\epsilon$]")
        axs3[1, 0].set(ylabel="d $\\tau_{xy}$ / d $\epsilon_{x}$ [MPa]")

        axs3[0, 1].plot(ey_model, txy_model, 'k')
        axs3[0, 1].plot(ey_model, txy2, 'b', linestyle='dotted')
        axs3[0, 1].plot(ey_test, txy_test, 'r', linestyle='dashed')
        axs3[0, 1].set_title('$\epsilon_y-\\tau_{xy}$')
        axs3[0, 1].set(ylabel="$\\tau_{xy,Layer}$ [MPa]")

        axs3[1, 1].plot(ey_model, dtxy_dey, 'b')
        axs3[1, 1].set(xlabel="$\epsilon$$_{y}$ [m$\epsilon$]")
        axs3[1, 1].set(ylabel="d $\\tau_{xy}$ / d $\epsilon_{y}$ [MPa]")

        axs3[0, 2].plot(gxy_model, txy_model, 'k')
        axs3[0, 2].plot(gxy_model, txy2, 'b', linestyle='dotted')
        axs3[0, 2].plot(gxy_test, txy_test, 'r', linestyle='dashed')
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
        axs3[2, 0].legend(['$\epsilon_x$', '$\epsilon_y$', '$\gamma_{xy}$', '$\epsilon_1$', '$\epsilon_3$'])
        axs3[2, 0].set_title('Strains')
        axs3[2, 0].set(xlabel="$\gamma_{xy}$ [m$\epsilon$]")
        axs3[2, 0].set(ylabel="$\epsilon$")

        axs3[2, 1].set_title('Principal Dir')
        axs3[2, 1].plot(gxy_model, thr * 180 / pi, 'k')
        axs3[2, 1].plot(gxy_model, thc * 180 / pi, 'r')
        axs3[2, 1].legend(['$\\theta_r$', '$\\theta_c$'])
        axs3[2, 1].set(xlabel="$\gamma_{xy}$ [m$\epsilon$]")
        axs3[2, 1].set(ylabel="$\\theta$ [°]")

        axs3[2, 2].set_title('Model/Submodel')
        axs3[2, 2].plot(gxy_model, model, 'b')
        axs3[2, 2].plot(gxy_model, submodel, 'k')
        axs3[2, 2].legend(['Model', 'Submodel'])
        axs3[2, 2].set(xlabel="$\gamma_{xy}$ [m$\epsilon$]")
        axs3[2, 2].set(ylabel="Model")

        fig3.tight_layout(pad=2.0)

        # 3.4 Fourth Plot: Material Stresses
        fig4, axs4 = plt.subplots(3, 3)
        fig4.set_figheight(9)
        fig4.set_figwidth(10)

        axs4[0, 0].plot(e3, sc3, 'k')
        axs4[0, 0].set_title('$\epsilon_3-\sigma_{c3}$')
        axs4[0, 0].set(ylabel="$\sigma_{c3,Layer}$ [MPa]")
        axs4[0, 0].set(xlabel="$\epsilon_{3,Layer}$ [m$\epsilon$]")

        axs4[0, 1].plot(gxy_model, sc3, 'k')
        axs4[0, 1].set_title('$\\gamma_{xy}-\sigma_{c3}$')
        axs4[0, 1].set(ylabel="$\sigma_{c3,Layer}$ [MPa]")
        axs4[0, 1].set(xlabel="$\\gamma_{xy,Layer}$ [MPa]")

        axs4[0, 2].plot(e1, fcs, 'k')
        axs4[0, 2].set_title('$\epsilon_1-f_{c,soft}$')
        axs4[0, 2].set(ylabel="$f_{c,soft}$ [MPa]")
        axs4[0, 2].set(xlabel="$\epsilon_{1,Layer}$ [MPa]")

        axs4[1, 0].plot(gxy_model, scnr, 'r')
        axs4[1, 0].plot(gxy_model, tctnr, 'm')
        axs4[1, 0].plot(gxy_model, tctndow, 'm--')
        axs4[1, 0].set_title('Interlock Stresses')
        axs4[1, 0].set(ylabel="Interlock Stresses [MPa]")
        axs4[1, 0].set(xlabel="$\\gamma_{xy,Layer}$ [MPa]")
        axs4[1, 0].legend(['$\sigma_{cnr}$', '$\\tau_{tctnr,tot}$', '$\\tau_{tctnr,dowel}$'])

        axs4[1, 1].plot(gxy_model, dn, 'r')
        axs4[1, 1].plot(gxy_model, dt, 'm')
        axs4[1, 1].set_title('Crack Kinematics')
        axs4[1, 1].set(ylabel="Crack Displacements [mm]")
        axs4[1, 1].set(xlabel="$\\gamma_{xy,Layer}$ [MPa]")
        axs4[1, 1].legend(['$\delta_{n}$', '$\delta_{t}$'])

        axs4[1, 2].plot(gxy_model, srx, 'k')
        axs4[1, 2].plot(gxy_model, sry, 'gray')
        axs4[1, 2].set_title('Crack Spacings')
        axs4[1, 2].set(ylabel="Crack Spacings [mm]")
        axs4[1, 2].set(xlabel="$\\gamma_{xy,Layer}$ [MPa]")
        axs4[1, 2].legend(['$s_{rx}$', '$s_{ry}$'])

        axs4[2, 0].plot(ex_model, ssrx, 'b')
        axs4[2, 0].plot(ex_model, ssry, 'c')
        axs4[2, 0].set_title('$\epsilon_x-\sigma_{sr}$')
        axs4[2, 0].set(ylabel="$\sigma_{sr}$ [MPa]")
        axs4[2, 0].set(xlabel="$\epsilon_{x}$ [MPa]")
        axs4[2, 0].legend(['$\sigma_{sxr}$', '$\sigma_{syr}$'])

        axs4[2, 1].plot(ey_model, ssrx, 'b')
        axs4[2, 1].plot(ey_model, ssry, 'c')
        axs4[2, 1].set_title('$\epsilon_y-\sigma_{sr}$')
        axs4[2, 1].set(ylabel="$\sigma_{sr}$ [MPa]")
        axs4[2, 1].set(xlabel="$\epsilon_{y}$ [MPa]")
        axs4[2, 1].legend(['$\sigma_{sxr}$', '$\sigma_{syr}$'])

        axs4[2, 2].plot(gxy_model, ssrx, 'b')
        axs4[2, 2].plot(gxy_model, ssry, 'c')
        axs4[2, 2].set_title('$\\gamma_{xy}-\sigma_{sr}$')
        axs4[2, 2].set(ylabel="$\sigma_{sr}$ [MPa]")
        axs4[2, 2].set(xlabel="$\\gamma_{xy}$ [MPa]")
        axs4[2, 2].legend(['$\sigma_{sxr}$', '$\sigma_{syr}$'])

        fig4.tight_layout(pad=2.0)

        if sig.cms_klij == 4:
            plt.figure(figsize=(6, 6))
            plt.subplot(4, 1, 1)
            plt.plot(xcr_01, [i * 1000 for i in es_all_01])
            plt.plot(xcr_01, [i * 1000 for i in ep_all_01])
            plt.ylabel("$\epsilon$$_{x} [m\epsilon]$")

            plt.subplot(4, 1, 2)
            plt.plot(xcr_01, ss_all_01)
            plt.plot(xcr_01, sp_all_01)
            plt.ylabel("$\sigma$$_{x} [MPa]$")

            plt.subplot(4, 1, 3)
            plt.plot(xcr_01, tbs_all_01)
            plt.plot(xcr_01, tbp_all_01)
            plt.ylabel("$\u03C4$$_{b} [MPa]$")

            plt.subplot(4, 1, 4)
            plt.plot(xcr_01, ds_all_01)
            plt.plot(xcr_01, dp_all_01)
            plt.ylabel("$\delta$$_{x} [mm]$")
            plt.xlabel("$x [mm]$")
            plt.subplots_adjust(hspace=1)
        plt.show()

    # ------------------------------------------------------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------------------------------------------------------
    import shutil

    if target_stress == True:
        str_name = Tests[j] +'_'+ str(cm_klij) + str(cmcc_klij) + str(cmcs_klij) + str(cms_klij) + str(cmtn_klij) + str(do_itkin) + str(do_char) + '_'
        # path_res = os.path.join(path, str_name)
        # print(path_res)
        # Save for LUSET test
        np.savetxt(os.path.join(path, str_name + 'gxy.txt'), gxy_model)
        np.savetxt(os.path.join(path, str_name + 'ex.txt'), ex_model)
        np.savetxt(os.path.join(path, str_name + 'ey.txt'), ey_model)
        np.savetxt(os.path.join(path, str_name + 'txy_model.txt'), txy_model)
        np.savetxt(os.path.join(path, str_name + 'txy_test.txt'), txy_test)
        np.savetxt(os.path.join(path, str_name + 'gxy_test.txt'), gxy_test)
        np.savetxt(os.path.join(path, str_name + 'ex_test.txt'), ex_test)
        np.savetxt(os.path.join(path, str_name + 'ey_test.txt'), ey_test)
        np.savetxt(os.path.join(path, str_name + 'sc3_model.txt'), sc3)
        np.savetxt(os.path.join(path, str_name + 'thc_model.txt'), thc)



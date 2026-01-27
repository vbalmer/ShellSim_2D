from math import *
from Standard import s_c3, s_sc,e_principal
from Mesh_gmsh import MATK, GEOMK
import numpy as np
import cmath
# 0 Lambda for TCM
lbd = 0.67
"""------------------------------------------------------------------------------------------------------------------"""
def s_sr2(e, e1, e3, th, k, l, dir):
    """ ------------------------ Get Steel Stresses for Given Two-dimensional Strain state -----------------------------
        --------------------------------------------    INPUT: ------------------------------------------------------
        - e: Normal strain in regarded direction
        - e1, e3: Principal strains
        - th: Principal direction
        - k: Element number
        - l: Layer number
        - dir: regarded direction (0 = x; 1 = y)
        --------------------------------------------- OUTPUT:--------------------------------------------------------
        - ssr: steel stress
    -----------------------------------------------------------------------------------------------------------------"""
    # 0 Material Parameters

    fcp = MATK["fcp"][k]
    fct = MATK["fct"][k]
    v = MATK["vc"][k]
    Ec = MATK["Ec"][k]
    ec0 = MATK["ec0"][k]
    tb1 = MATK["tb1"][k]
    tb2 = MATK["tb2"][k]
    rhox = GEOMK["rhox"][k][l]
    dx =  GEOMK["dx"][k][l]
    Esx = MATK["Esx"][k]
    Eshx = MATK["Eshx"][k]
    fsyx = MATK["fsyx"][k]
    fsux = MATK["fsux"][k]
    rhoy = GEOMK["rhoy"][k][l]
    dy = GEOMK["dy"][k][l]
    Esy = MATK["Esy"][k]
    Eshy = MATK["Eshy"][k]
    fsyy = MATK["fsyy"][k]
    fsuy = MATK["fsuy"][k]

    # 1 Assign Material Parameters Depending on given Direction

    # 1.1 x-direction
    if dir == 0:
        rho = rhox
        d = dx
        fsy = fsyx
        fsu = fsux
        Es = Esx
        Esh = Eshx

    # 1.2 y-direction
    elif dir == 1:
        rho = rhoy
        d = dy
        fsy = fsyy
        fsu = fsuy
        Es = Esy
        Esh = Eshy

    # 2 Crack Spacing
    # 2.1 Diagonal Crack Spacing
    srm0 = s_rm0(th, fct, rhox, rhoy, dx, dy, tb1)
    srm_diag = srm0 * lbd

    # 2.2 Crack Spacing in Regarded Direction
    # 2.2.1 Biaxial Tension
    th = abs(th)
    if e3 >= 0:
        srm = srm_diag

    # 2.2.2 CMM
    else:
        if dir == 0:
            srm = srm_diag/sin(th)
        elif dir == 1:
            srm = srm_diag/cos(th)

    # 3 Assign Steel Stress
    # 3.1 Compressive Strain: bare steel
    if e <= 0:
        ssr = s_sc(e, fsy, Es, Esh)
    # 3.2 Tensile Strain: TCM
    else:
        ssr = TCM(e, srm, rho, d, fsy, fsu, Es, Esh, tb1, tb2, Ec)

    return ssr


def s_sr1(e, k, l, dir):
    """ ------------------------ Get Steel Stresses for Given One-dimensional Strain state -----------------------------
        --------------------------------------------    INPUT: ------------------------------------------------------
        - e: normal strain
        - k: element number
        - l: layer number
        - dir: direction of steel stress in local coordinates
        --------------------------------------------- OUTPUT:--------------------------------------------------------
        - ssr: steel stress
    -----------------------------------------------------------------------------------------------------------------"""
    # 0 Material Parameters
    fcp = MATK["fcp"][k]
    fct = MATK["fct"][k]
    v = MATK["vc"][k]
    Ec = MATK["Ec"][k]
    ec0 = MATK["ec0"][k]
    tb1 = MATK["tb1"][k]
    tb2 = MATK["tb2"][k]
    rhox = GEOMK["rhox"][k][l]
    dx =  GEOMK["dx"][k][l]
    Esx = MATK["Esx"][k]
    Eshx = MATK["Eshx"][k]
    fsyx = MATK["fsyx"][k]
    fsux = MATK["fsux"][k]
    rhoy = GEOMK["rhoy"][k][l]
    dy = GEOMK["dy"][k][l]
    Esy = MATK["Esy"][k]
    Eshy = MATK["Eshy"][k]
    fsyy = MATK["fsyy"][k]
    fsuy = MATK["fsuy"][k]

    # 1 Assign Material Parameters Depending on given Direction
    # 1.1 x-direction
    if dir == 0:
        rho = rhox
        d = dx
        fsy = fsyx
        fsu = fsux
        Es = Esx
        Esh = Eshx

    # 1.2 y-direction
    elif dir == 1:
        rho = rhoy
        d = dy
        fsy = fsyy
        fsu = fsuy
        Es = Esy
        Esh = Eshy

    # 2 Crack Spacing
    srm0 = (fct * d * (1 - rho)) / (2 * tb1 * rho)
    srm = lbd*srm0

    # 3 Assign Steel Stress
    # 3.1 Compressive Strain: bare steel
    if e <= 0:
        ssr = s_sc(e, fsy, Es, Esh)

    # 3.2 Tensile Strain: TCM
    else:
        ssr = TCM(e, srm, rho, d, fsy, fsu, Es, Esh, tb1, tb2, Ec)

    return ssr


def TCM(e, srm, rho, d, fsy, fsu, Es, Esh, tb1, tb2, Ec):
    """ ------------------------------------- Calculate Steel Stress with the TCM -----------------------------------
        --------------------------------------------    INPUT: ------------------------------------------------------
        - e: normal strain
        - srm: Crack spacing: srm = lambda * sr0
        - rho: Reinforcement content
        - d: Reinforcement diameter
        - fsy,fsu,Es,Esh: Steel parameters
        - tb1,tb2: Bond stresses elastic and plastic
        - Ec: Concrete E-modulus
        --------------------------------------------- OUTPUT:--------------------------------------------------------
        - ssr: steel stress
    -----------------------------------------------------------------------------------------------------------------"""
    # 1 Seelhofer

    # 1.1 Material and Geometric properties
    n = Es / Ec
    alpha = 1 + n * rho

    # 1.2 Elastic crack element
    c1 = sqrt(n * n * rho * rho + Es * e / tb1 * d / srm) - n * rho
    x1 = srm/2*c1
    x1 = min(max(x1,0),srm/2)
    ssr = x1*4*tb1/d*(1+n*rho)

    # 1.3 Elastic - Plastic crack element
    if ssr > fsy:
        c2 = sqrt(4 * alpha * Es / Esh * (
                    srm * tb2 / (d * fsy) * (alpha * Es * e / fsy - n * rho) - tb2 / (4 * alpha * tb1)) + 1) - 1
        x2 = d * fsy * Esh / (4 * tb2 * alpha * Es) * c2
        x2 = min(max(x2, 0), srm / 2)
        ssr = fsy+x2*4*tb2/d
        x21 = (fsy-ssr*n*rho/(1+n*rho))*d/(4*tb1)
        x1 = x2+x21

    # 2. TCM
    if x1 >= srm/2:

        # 2.1 Bare steel stress
        st_naked = s_sc(e, fsy, Es, Esh)

        # 2.2 Steel stress for fully elastic crack element
        s1 = st_naked + tb1*srm/d

        # 2.3 Steel stress for fully plastic element
        s3 = fsy + Esh * (e - fsy / Es) + tb2 * srm / d

        # 2.4 Assign according to stress level
        # 2.4.1 Fully elastic
        if s1 <= fsy:
            ssr = s1

        # 2.4.2 Fully plastic
        elif s1 > fsy and s3-(2*tb2*srm/d) >= fsy:
            ssr = s3

        # 2.4.3 Partially elastic
        else:
            s2 = (fsy - Es * e) * tb2 * srm / d * (tb1 / tb2 - Es / Esh)
            s2 = s2 + Es / Esh * tb1 * tb2 * srm ** 2 / d ** 2
            s2 = tb1 * srm / d - sqrt(s2)
            s2 = fsy + 2 * s2 / (tb1 / tb2 - Es / Esh)
            ssr = s2
    # 2.5 If stress > ultimate stress, assign ultimate stress
    if ssr > fsu:
        ssr = fsu

    return ssr

    
def s_rm0(th, fct, rhox, rhoy, dx, dz, tb1):
    """ --------------------------------- Calculate diagonal crack spacing-------------------------------------------
        --------------------------------------------    INPUT: ------------------------------------------------------
        - th: Principal direction
        - fct: concrete tensile strength
        - rhox, rhoy: Reinforcement contents
        - dx,dy: Reinforcement diameters
        - tb1: bond stress elastic
        --------------------------------------------- OUTPUT:--------------------------------------------------------
        - s_rm0 --> Maximum diagonal crack spacing
    -----------------------------------------------------------------------------------------------------------------"""
    # 0 Manipulate theta
    th = abs(th)

    # 1 Initial Assumption (Formula 5.7 in Kaufmann, 1998)
    srmx0 = (fct * dx * (1 - rhox)) / (2 * tb1 * rhox)
    srmy0 = (fct * dz * (1 - rhoy)) / (2 * tb1 * rhoy)
    sr0   = 1/((sin(th)/srmx0)+(cos(th)/srmy0))

    return float(sr0)
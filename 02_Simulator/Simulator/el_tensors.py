import numpy as np
from Standard import e_principal,s_c3,f_cs,tmat,s_sc
from CMM import s_sr1,s_sr2
from math import *
from Mesh_gmsh import MATK,GEOMK

def ET_0():
    """ ------------------------------- Generate "Elastic Tensor" for cutout-----------------------------------------
        --------------------------------------------    INPUT: ------------------------------------------------------
        - E <1
        --------------------------------------------- OUTPUT:--------------------------------------------------------
        -ET: "Elasticity Tensor"
    -----------------------------------------------------------------------------------------------------------------"""
    ET = np.array([[0.1, 0.03, 0], [0.03, 0.1, 0], [0, 0, 0.05]])
    return ET

def ET_1(E, v):
    """ ----------------------- Generate Plane Stress Elasticity Matrix for Linear Elasticity -----------------------
        --------------------------------------------    INPUT: ------------------------------------------------------
        - E: Young's Modulus
        - v: Poisson's Ratio
        --------------------------------------------- OUTPUT:--------------------------------------------------------
        -ET: Elasticity Matrix
    -----------------------------------------------------------------------------------------------------------------"""
    ET = E / (1 - v * v) * np.array([[1, v, 0], [v, 1, 0], [0, 0, 0.5 * (1 - v)]])
    return ET

def ET_2(ex, ey, gxy, k, l):
    """ ---------------------- Generate Elastic Matrix for "CMM-" Model plane stress---------------------------------
        --------------------------------------------    INPUT: ------------------------------------------------------
        - Strain state
        - Element number
        --------------------------------------------- OUTPUT:--------------------------------------------------------
        -ET: Elasticity Matrix
        !!!! ATTENTION: DEPRACETED! NEEDS TO BE UPDATED ACCORDING TO ET_3: TENSION/COMPRESSION/CMM- !!!!!!
    -----------------------------------------------------------------------------------------------------------------"""
    """------------------------------------------ Material Parameters -----------------------------------------------"""
    fcp = MATK["fcp"][k]
    Ec = MATK["Ec"][k]
    ec0 = MATK["ec0"][k]
    rhox = GEOMK["rhox"][k][l]
    Esx = MATK["Esx"][k]
    Eshx = MATK["Eshx"][k]
    fsyx = MATK["fsyx"][k]
    rhoy = GEOMK["rhoy"][k][l]
    Esy = MATK["Esy"][k]
    Eshy = MATK["Eshy"][k]
    fsyy = MATK["fsyy"][k]
    v = MATK["vc"][k]

    """--------------------------------------------------------------------------------------------------------------"""
    [e1, e3, th] = e_principal(ex, ey, gxy)
    sc3 = s_c3(e3, e1, fcp, ec0)
    sc1 = s_c3(e1, 0, fcp, ec0)
    T = tmat(th)
    if e1 < 0:
        E1 = abs(sc1 / e1)
    else:
        E1 = 100
    if e3 < 0:
        E3 = abs(sc3 / e3)
    else:
        E3 = 100
    if abs(e3 - e1) > 0:
        G = max(abs(0.5 * sc3 / (e3 - e1)),50)
    else:
        G = 50
    Ec13 = np.array([[E1, 0, 0], [0, E3, 0], [0, 0, G]])
    Ec = np.matmul(np.transpose(T), np.matmul(Ec13, T))

    Esx = Esx
    Esy = Esy
    Es = np.array([[rhox * Esx, 0, 0], [0, rhoy * Esy, 0], [0, 0, 0]])
    return Ec + Es

def ET_3_1(ex, ey, gxy, k, l):
    """ -------------------------- Generate Constitutive Matrix for Tension - Tension ------------------------------------
        --------------------------------------------    INPUT: ------------------------------------------------------
        - Strain state
        - Element and Layer number
        --------------------------------------------- OUTPUT:--------------------------------------------------------
        -ET: Consitutive Matrix
    -----------------------------------------------------------------------------------------------------------------"""
    # 1 Material Parameters
    # 1.1 Reinforcement content
    rhox = GEOMK["rhox"][k][l]
    rhoy = GEOMK["rhoy"][k][l]

    # 1.2 Poisson's ratio and E-Modulus of concrete
    v = MATK["vc"][k]
    Ec = MATK["Ec"][k]

    # 1.3 Assumed strain at cracking
    ect = 0 * MATK["fct"][k] / Ec

    # 2 Concrete Constitutive Matrix
    # 2.1 If ex < cracking strain: Assign linear elastic Law in x/y
    #     Else: Set "zero" stiffness in tensile direction: 100 for numerical stability
    if ex <= ect:
        Ex = Ec
    else:
        Ex = 100
    if ey <= ect:
        Ey = Ec
    else:
        Ey = 100

    # 2.2 Constitutive Matrix is formulated directly in x-y space
    Ec = np.array([[Ex,0,0],[0,Ey,0],[0,0,(Ex+Ey)/4]])

    # 3 Steel Stiffness Matrix
    # 3.1 Steel Stresses in reinforcement layers
    if rhox>0:
        ssx = s_sr1(ex, k, l, 0)
    else:
        ssx = 0
    if rhoy>0:
        ssy = s_sr1(ey, k, l, 1)
    else:
        ssy=0

    # 3.2 Steel Secant Stiffness Matrix
    if abs(ex) > 0:
        Esx = ssx / ex
    else:
        Esx = 0
    if abs(ey) > 0:
        Esy = ssy / ey
    else:
        Esy = 0

    # 3.3 Constitutive Matrix in x-y space
    Ds = np.array([[rhox*Esx,0,0],[0,rhoy*Esy,0],[0,0,0]])

    # 4 Return Constitutive Matrix of Steel and Concrete
    return Ec + Ds

def ET_3_2(ex, ey, gxy, k, l):
    """ ----------------------- Generate Constitutive Model for Compression - Compression ---------------------------
        --------------------------------------------    INPUT: ------------------------------------------------------
        - Strain state
        - Element and layer number
        --------------------------------------------- OUTPUT:--------------------------------------------------------
        -ET: Constitutive Matrix
    -----------------------------------------------------------------------------------------------------------------"""
    # 1 Material Parameters
    # 1.1 Concrete
    fcp = MATK["fcp"][k]
    ec0 = MATK["ec0"][k]
    v = MATK["vc"][k]

    # 1.2 Steel
    Esx = MATK["Esx"][k]
    Eshx = MATK["Eshx"][k]
    fsyx = MATK["fsyx"][k]
    Esy = MATK["Esy"][k]
    Eshy = MATK["Eshy"][k]
    fsyy = MATK["fsyy"][k]

    # 1.3 Reinforcement content
    rhox = GEOMK["rhox"][k][l]
    rhoy = GEOMK["rhoy"][k][l]

    # 2 Concrete Constitutive Matrix
    # 2.1 Principal Strains
    [e1, e3, th] = e_principal(ex, ey, gxy)

    # 2.2 Principal Concrete stresses
    sc3 = s_c3(e3, 0, fcp, ec0)
    sc1 = s_c3(e1, 0, fcp, ec0)

    # 2.3 Concrete secant stiffness matrix
    #     If concrete in 1- and 3- directions: assign secant stiffness
    #     Else: assign "zero" (not possible in compression-compression case)
    if e1 < 0:
        E1 = abs(sc1 / e1)
    else:
        E1 = 100
    if e3 < 0:
        E3 = abs(sc3 / e3)
    else:
        E3 = 100
    if abs(e3 - e1) > 0:
        G = max(abs(0.5 * (sc3-sc1) / (e3 - e1)), 50)
    else:
        G = 50

    # 2.4 Assign concrete secant constitutive matrix in principal directions
    Ec13 = np.array([[E1, 0, 0], [0, E3, 0], [0, 0, G]])

    # 3 Transform from principal to local coordinate System
    #   Remember: inv(Tsigma) = transpose(Tepsilon)
    Tsigma = np.array([[sin(th) ** 2, cos(th) ** 2, 2 * sin(th) * cos(th)],
                       [cos(th) ** 2, sin(th) ** 2, -2 * sin(th) * cos(th)],
                       [- sin(th) * cos(th), sin(th) * cos(th), sin(th) ** 2 - cos(th) ** 2]])
    Tepsilon = np.array([[sin(th) ** 2, cos(th) ** 2, sin(th) * cos(th)],
                         [cos(th) ** 2, sin(th) ** 2, -sin(th) * cos(th)],
                         [-2 * sin(th) * cos(th), 2 * sin(th) * cos(th), sin(th) ** 2 - cos(th) ** 2]])
    Ec = np.linalg.inv(Tsigma)@Ec13@Tepsilon

    # 4 Steel Constitutive Matrix
    # 4.1 Steel Stresses in reinforcement layers
    if rhox>0:
        ssx = s_sc(ex, fsyx, Esx, Eshx)
    else:
        ssx = 0
    if rhoy>0:
        ssy = s_sc(ey, fsyy, Esy, Eshy)
    else:
        ssy=0

    # 4.2 Steel Secant Stiffness Moduli
    if abs(ex) > 0:
        Esx = ssx / ex
    else:
        Esx = 0
    if abs(ey) > 0:
        Esy = ssy / ey
    else:
        Esy = 0

    # 4.3 Constitutive Matrix in x-y space
    Ds = np.array([[rhox*Esx,0,0],[0,rhoy*Esy,0],[0,0,0]])

    # 5 Return Constitutive Matrix of Steel and Concrete
    return Ec + Ds

def ET_3_3(ex, ey, gxy, k, l):
    """ --------------------------- Generate Elastic Tensor for CMM plane stress-------------------------------------
        --------------------------------------------    INPUT: ------------------------------------------------------
        - Strain state
        - Element and layer number
        --------------------------------------------- OUTPUT:--------------------------------------------------------
        -ET: Constitutive Matrix
    -----------------------------------------------------------------------------------------------------------------"""

    # 1 Material Parameters
    # 1.1 Reinforcement content
    rhox = GEOMK["rhox"][k][l]
    rhoy = GEOMK["rhoy"][k][l]

    # 1.2 Concrete
    v = MATK["vc"][k]
    Ec = MATK["Ec"][k]
    fcp = MATK["fcp"][k]
    ec0 = MATK["ec0"][k]

    # 1.3 Assumed strain at cracking
    ect = 0 * MATK["fct"][k] / Ec

    # 2 Concrete Constitutive Matrix
    # 2.1 Principal Strains
    [e1, e3, th] = e_principal(ex, ey, gxy)

    # 2.2 Principal Concrete stresses
    sc3 = s_c3(e3, 0, fcp, ec0)
    sc1 = s_c3(e1, 0, fcp, ec0)

    # 2.3 Concrete secant stiffness matrix
    #     If concrete in 1- and 3- directions: assign secant stiffness
    #     Else: Ec if uncracked
    #           "zero" if e1 > cracking strain
    if e1 < 0:
        E1 = abs(sc1 / e1)
    elif e1 <= ect:
        E1 = Ec
    else:
        E1 = 100
    if e3 < 0:
        E3 = abs(sc3 / e3)
    elif e3 <= ect:
        E3 = Ec
    else:
        E3 = 100
    if abs(e3 - e1) > 0:
        G = max(abs(0.5 * (sc3-sc1) / (e3 - e1)),50)
    else:
        G = 50

    # 2.4 Assign concrete secant constitutive matrix in principal directions
    Ec13 = np.array([[E1, 0, 0], [0, E3, 0], [0, 0, G]])

    # 3 Transform from principal to local coordinate System
    #   Remember: inv(Tsigma) = transpose(Tepsilon)

    Tsigma = np.array([[sin(th) ** 2, cos(th) ** 2, 2 * sin(th) * cos(th)],
                       [cos(th) ** 2, sin(th) ** 2, -2 * sin(th) * cos(th)],
                       [- sin(th) * cos(th), sin(th) * cos(th), sin(th) ** 2 - cos(th) ** 2]])
    Tepsilon = np.array([[sin(th) ** 2, cos(th) ** 2, sin(th) * cos(th)],
                         [cos(th) ** 2, sin(th) ** 2, -sin(th) * cos(th)],
                         [-2 * sin(th) * cos(th), 2 * sin(th) * cos(th), sin(th) ** 2 - cos(th) ** 2]])
    Ec = np.linalg.inv(Tsigma)@Ec13@Tepsilon

    # 4 Steel Constitutive Matrix
    # 4.1 Steel Stresses in reinforcement layers
    if rhox>0:
        ssx = s_sr2(ex, e1, e3, th, k, l, 0)
    else:
        ssx = 0
    if rhoy>0:
        ssy = s_sr2(ey, e1, e3, th, k, l, 1)
    else:
        ssy=0

    # 4.2 Steel Secant Stiffness Moduli
    if abs(ex) > 0:
        Esx = ssx / ex
    else:
        Esx = 0
    if abs(ey) > 0:
        Esy = ssy / ey
    else:
        Esy = 0

    # 4.3 Constitutive Matrix in x-y space
    Ds = np.array([[rhox*Esx,0,0],[0,rhoy*Esy,0],[0,0,0]])
    
    # 5 Return Constitutive Matrix of Steel and Concrete
    return Ec + Ds

def ET_1_shear(Ex, Ey, v):
    """ ---------------------- Generate Shear Stiffness Matrix in xz and yz Direction -------------------------------
        --------------------------------------------    INPUT: ------------------------------------------------------
        - E: Young's Modulus
        - v: Poisson's Ratio
        --------------------------------------------- OUTPUT:--------------------------------------------------------
        -ET: Elastic Matrix
    -----------------------------------------------------------------------------------------------------------------"""
    G = (Ex+Ey)/(4*(1+v))
    ET = np.array([[5/6*G,0],[0,5/6*G]])
    return ET


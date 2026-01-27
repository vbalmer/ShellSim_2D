global u
""" ----------------------------------------- Import Input and Meshing ----------------------------------------------"""
from Mesh_gmsh import order,gauss_order,BC,Load_el,Load_n,ELS,GEOMK,MATK,COORD,copln,it_type
ELEMENTS = ELS[0]
"""-------------------------------------- Import General Settings and Tools------------------------------------------"""
from math import *
from Stresses_mixreinf import stress
import numpy as np
import time
from scipy.linalg import lu_factor, lu_solve
import matplotlib.pyplot as plt


"""---------------------------------------------- Time Measurement---------------------------------------------------"""
class gettime:
    def _updatetime(self,update=True,delta_t=0):
        if update:
            self.time_spent += delta_t
            # print(self.time_spent)
        else:
            self.time_spent = 0
time_stress = gettime()
time_stress._updatetime(update=False)
time_strain = gettime()
time_strain._updatetime(update=False)
time_K = gettime()
time_K._updatetime(update=False)
time_B = gettime()
time_B._updatetime(update=False)
time_Kinv = gettime()
time_Kinv._updatetime(update=False)
time_sh = gettime()
time_sh._updatetime(update=False)
time_eh = gettime()
time_eh._updatetime(update=False)

"""----------------------------------------- Save old Stiffness Matrix ----------------------------------------------"""
class ki:
    def _updatek(self,update=False,knew=0):
        if update:
            self.itstep += 1
            self.k = knew
        else:
            self.itstep = 0
            self.k = knew
kold = ki()
kold._updatek()
"""------------------------------------------------ Integration------------------------------------------------------"""


def gauss_points(nn,go):
    """ ------------------------------ Gauss Points for Quadrilaterals and Triangles -----------------------------------
        --------------------------------------------    INPUT: ------------------------------------------------------
        - nn: Number of nodes
        - go: Gauss order
        --------------------------------------------- OUTPUT:--------------------------------------------------------
        - xi: Gauss points in one dimension
        - w: Gauss weights in one dimension
    -----------------------------------------------------------------------------------------------------------------"""
    # 0 Gauss Points for Quadrilaterals
    if nn == 4:
        if go == 1:
            xi = np.array([0])
            w = np.array([2])
        elif go == 2:
            xi = np.array([-sqrt(1/3),sqrt(1/3)])
            w = np.array([1,1])
        elif go == 3:
            xi = np.array([-0.774597, 0, 0.774597])
            w = np.array([5/9,8/9,5/9])
        elif go == 4:
            xi = np.array([-0.861136311594053,-0.339981043584856,0.339981043584856,0.861136311594053])
            w = np.array([0.3478548451374538,0.6521451548625461,0.6521451548625461,0.3478548451374538])
        elif go == 5:
            xi = np.array([-0.9061798459386640,-0.5384693101056831,0,0.5384693101056831,0.9061798459386640])
            w = np.array([0.2369268850561891,0.4786286704993665,0.5688888888888889,0.4786286704993665,0.2369268850561891])
        elif go == 6:
            xi1 = -0.9324695142031521
            xi2 = -0.6612093864662645
            xi3 = -0.2386191860831969
            xi4 = 0.2386191860831969
            xi5 = 0.661209386466264
            xi6 = 0.9324695142031521
            a1 = 0.1713244923791704
            a2 = 0.3607615730481387
            a3 = 0.4679139345726913
            a4 = a3
            a5 = a2
            a6 = a1
            xi = np.array([xi1,xi2,xi3,xi4,xi5,xi6])
            w = np.array([a1,a2,a3,a4,a5,a6])
        elif go == 30:
            w = [0.102852653,0.102852653,0.10176239,0.10176239,0.099593421,0.099593421,0.096368737,0.096368737,0.092122522,0.092122522,
                 0.086899787,0.086899787,0.080755895,0.080755895,0.073755975,0.073755975,0.06597423,0.06597423,0.057493156,0.057493156,
                 0.048402673,0.048402673,0.038799193,0.038799193,0.028784708,0.028784708,0.018466468,0.018466468,0.007968192,0.007968192]
            xi = [-0.051471843,0.051471843,-0.153869914,0.153869914,-0.254636926,0.254636926,-0.352704726,0.352704726,-0.44703377,0.44703377,
                  -0.536624148,0.536624148,-0.620526183,0.620526183,-0.697850495,0.697850495,-0.767777432,0.767777432,-0.829565762,0.829565762,
                  -0.882560536,0.882560536,-0.926200047,0.926200047,-0.960021865,0.960021865,-0.983668123,0.983668123,-0.996893484,0.996893484]
        elif go == 40:
            w = [0.077505948,
                0.077505948,
                0.077039818,
                0.077039818,
                0.076110362,
                0.076110362,
                0.074723169,
                0.074723169,
                0.072886582,
                0.072886582,
                0.070611647,
                0.070611647,
                0.067912046,
                0.067912046,
                0.064804013,
                0.064804013,
                0.061306242,
                0.061306242,
                0.057439769,
                0.057439769,
                0.053227847,
                0.053227847,
                0.048695808,
                0.048695808,
                0.043870908,
                0.043870908,
                0.038782168,
                0.038782168,
                0.033460195,
                0.033460195,
                0.027937007,
                0.027937007,
                0.022245849,
                0.022245849,
                0.016421058,
                0.016421058,
                0.010498285,
                0.010498285,
                0.004521277,
                0.004521277]
            xi = [-0.038772418,
            0.038772418,
            -0.116084071,
            0.116084071,
            -0.192697581,
            0.192697581,
            -0.268152185,
            0.268152185,
            -0.341994091,
            0.341994091,
            -0.413779204,
            0.413779204,
            -0.483075802,
            0.483075802,
            -0.549467125,
            0.549467125,
            -0.61255389,
            0.61255389,
            -0.671956685,
            0.671956685,
            -0.727318255,
            0.727318255,
            -0.778305651,
            0.778305651,
            -0.824612231,
            0.824612231,
            -0.865959503,
            0.865959503,
            -0.902098807,
            0.902098807,
            -0.932812808,
            0.932812808,
            -0.957916819,
            0.957916819,
            -0.97725995,
            0.97725995,
            -0.990726239,
            0.990726239,
            -0.99823771,
            0.99823771]
        elif go == 50:
            w = [0.062176617,
                0.062176617,
                0.061936067,
                0.061936067,
                0.0614559,
                0.0614559,
                0.060737971,
                0.060737971,
                0.059785059,
                0.059785059,
                0.05860085,
                0.05860085,
                0.057189926,
                0.057189926,
                0.055557745,
                0.055557745,
                0.053710622,
                0.053710622,
                0.051655703,
                0.051655703,
                0.049400938,
                0.049400938,
                0.046955051,
                0.046955051,
                0.044327504,
                0.044327504,
                0.041528463,
                0.041528463,
                0.038568757,
                0.038568757,
                0.035459836,
                0.035459836,
                0.032213728,
                0.032213728,
                0.028842994,
                0.028842994,
                0.025360674,
                0.025360674,
                0.021780243,
                0.021780243,
                0.018115561,
                0.018115561,
                0.014380823,
                0.014380823,
                0.010590548,
                0.010590548,
                0.006759799,
                0.006759799,
                0.002908623,
                0.002908623]
            xi = [-0.031098338,
                0.031098338,
                -0.093174702,
                0.093174702,
                -0.15489059,
                0.15489059,
                -0.216007237,
                0.216007237,
                -0.276288194,
                0.276288194,
                -0.335500245,
                0.335500245,
                -0.393414312,
                0.393414312,
                -0.449806335,
                0.449806335,
                -0.504458145,
                0.504458145,
                -0.557158305,
                0.557158305,
                -0.607702927,
                0.607702927,
                -0.655896466,
                0.655896466,
                -0.701552469,
                0.701552469,
                -0.744494302,
                0.744494302,
                -0.784555833,
                0.784555833,
                -0.821582071,
                0.821582071,
                -0.855429769,
                0.855429769,
                -0.88596798,
                0.88596798,
                -0.913078557,
                0.913078557,
                -0.936656619,
                0.936656619,
                -0.956610955,
                0.956610955,
                -0.972864385,
                0.972864385,
                -0.985354084,
                0.985354084,
                -0.994031969,
                0.994031969,
                -0.998866404,
                0.998866404]

    # 1 Gauss Points for Triangles
    elif nn == 3:
        if go == 1:
            xi = np.array([1/3])
            w = np.array([1/sqrt(2)])
        elif go == 2:
            xi = np.array([1/6,2/3])
            w = np.array([1/sqrt(6),1/sqrt(6)])
    return xi,w


"""----------------------------------------------- Material Laws-----------------------------------------------------"""


def get_et(cm_klij,e_klij,s_klij, k, l,i,j):
    """ ------------------------------------- Define Constitutive Matrix -----------------------------------------------
        --------------------------------------------    INPUT: ------------------------------------------------------
        - cm_klij
            - 1: Linear Elasticity
            - 2: CMM-, CMM without tension stiffening (NOT IMPLEMENTED YET)
            - 3: CMM
        - k: Element number
        - l: Layer number
        - [ex_klij, ey_klij, gxy_klij]: Strain state
        --------------------------------------------- OUTPUT:--------------------------------------------------------
        -ET: Secant Constitutive Matrix
    -----------------------------------------------------------------------------------------------------------------"""
    s = s_klij
    if it_type == 1:
        if cm_klij == 1:
            E = MATK["Ec"][k]
            v = MATK["vc"][k]
            ET = E / (1 - v * v) * np.array([[1, v, 0], [v, 1, 0], [0, 0, 0.5 * (1 - v)]])
        else:

            ET = np.array([[s[0].sx.imag, s[1].sx.imag, s[2].sx.imag],
                           [s[0].sy.imag, s[1].sy.imag, s[2].sy.imag],
                           [s[0].txy.imag, s[1].txy.imag, s[2].txy.imag]])/0.0000000000000001
    elif it_type == 2:
        if cm_klij == 1:
            E = MATK["Ec"][k]
            v = MATK["vc"][k]
            ET = E / (1 - v * v) * np.array([[1, v, 0], [v, 1, 0], [0, 0, 0.5 * (1 - v)]])
        else:
            # Concrete Secant Stiffness Matrix
            if s[0].e1 < 0:
                E1 = abs(s[0].sc1 / s[0].e1)
            else:
                E1 = .10
            if s[0].e3 < 0:
                E3 = abs(s[0].sc3 / s[0].e3)
            else:
                E3 = .10
            if abs(s[0].e3 - s[0].e1) > 0:
                G = max(abs(0.5 * (s[0].sc3 - s[0].sc1) / (s[0].e3 - s[0].e1)), .05)
            else:
                G = .05
            Ec13 = np.array([[E1, 0, 0], [0, E3, 0], [0, 0, G]])
            s[0].t_mat()
            Ec = np.linalg.inv(s[0].Tsigma) @ Ec13 @ s[0].Tepsilon

            # Steel and CFRP Stiffness Matrix
            # if abs(s[0].ex) > 0:
            #     Esx = s[0].ssx / (s[0].ex)
            #     Epx = s[0].spx / (s[0].ex)
            # else:
            #     Esx = s[0].Esx
            #     Epx = s[0].Epx
            # if abs(s[0].ey-s[0].ecy) > 0:
            #     Esy = s[0].ssy / (s[0].ey)
            #     Epy = s[0].spy / (s[0].ey)
            # else:
            #     Esy = s[0].Esy
            #     Epy = s[0].Epy
            # # Esx = 200000
            # # Esy = 200000
            # # Epx = 150000
            # # Epy = 150000
            # Ds = np.array([[s[0].rhox * Esx, 0, 0], [0, s[0].rhoy * Esy, 0], [0, 0, 0]])
            # Dp = np.array([[s[0].rhopx * Epx, 0, 0], [0, s[0].rhopy * Epy, 0], [0, 0, 0]])
            Ds = np.array([[(s[0].sx-Ec[0,1]*s[0].ey-Ec[0,2]*s[0].gxy)/s[0].ex, 0, 0],
                           [0, (s[0].sy-Ec[1,0]*s[0].ex-Ec[1,2]*s[0].gxy)/s[0].ey, 0],
                           [0, 0, 0]])
            # Assemble
            ET = Ec + Ds
    return ET


"""------------------------------------------------ B_Matrices-------------------------------------------------------"""

def jacobi(k, i, j, go):
    """ -------------------------------------- Get Jacobian of Element  ------------------------------------------------
            ----------------------------------------------- INPUT: --------------------------------------------------
            - k: element number
            - i,j: Gauss point numbers in xi - and eta space
            - go: gauss order
            ---------------------------------------------- OUTPUT: --------------------------------------------------
            - xi,eta: Gauss points
            - J: Jacobian
            - J_inv: Inverse of jacobian
            - J_det: Determinant of jacobian
    -----------------------------------------------------------------------------------------------------------------"""
    # 1 Values of Importance

    # 1.1 Nodes of element k
    e_k = ELEMENTS[k, :]
    e_k = e_k[e_k<10**5]

    # 1.2 Area of element k
    a_k = GEOMK["ak"][k]

    # 1.3 Local coordinates of nodes of element k
    NODESL = COORD["n"][2][a_k]
    v = np.array(NODESL[e_k])

    # -----------------------------------------------------------------------------------------------------------------#
    # 2 Jacobian for quadrilaterals
    # -----------------------------------------------------------------------------------------------------------------#
    if len(e_k) == 4:
        if order == 1:

            # 2.1 Gauss points and weights
            gp, w = gauss_points(4,go)
            xi = gp[j]
            eta = gp[i]

            # 2.2 Derivatives of Shape Functions in xi-eta-space
            N1xi = float(- (1 - eta) / 4)
            N2xi = float((1 - eta) / 4)
            N3xi = float((1 + eta) / 4)
            N4xi = float(- (1 + eta) / 4)
            N1eta = float(- (1 - xi) / 4)
            N2eta = float(- (1 + xi) / 4)
            N3eta = float((1 + xi) / 4)
            N4eta = float((1 - xi) / 4)

            # 2.3 Gradient Matrix
            Grad_Mat = np.array([[N1xi, N2xi, N3xi, N4xi], [N1eta, N2eta, N3eta, N4eta]])

            # 2.4 Jacobian
            J = np.matmul(Grad_Mat, v)
            J_inv = np.linalg.inv(J)
            J_det = np.linalg.det(J)
    # -----------------------------------------------------------------------------------------------------------------#
    # 3 Jacobian of triangles
    # -----------------------------------------------------------------------------------------------------------------#
    else:
        if order == 1:

            # 3.1 Coordinates of element nodes: delete entry 100'001
            v = np.array(NODESL[ELEMENTS[k][0:3]])

            # 3.2 Gauss Points
            gp, w = gauss_points(3, go)
            xi = gp[j]
            eta = gp[i]

            # 3.3 Derivatives Shape Functions in xi-eta space
            N1xi = -1
            N2xi = 1
            N3xi = 0
            N1eta = -1
            N2eta = 0
            N3eta = 1

            # 3.4 Gradient Matrix
            Grad_Mat = np.array([[N1xi, N2xi, N3xi], [N1eta, N2eta, N3eta]])

            # 3.5 Jacobian
            J = np.matmul(Grad_Mat, v)
            J_inv = np.linalg.inv(J)
            J_det = np.linalg.det(J)

    return xi,eta,J,J_inv,J_det


def b_kij(k, i, j, go,rot=True):
    """ ---------------------------------------- Create B-Matrices  -------------------------------------------------
        ----------------------------------------------- INPUT: ------------------------------------------------------
        - k: element number
        - i,j: Gauss point numbers in xi - and eta space
        - go: gauss order
        - rot: True, if B-Matrices shall be rotated in global coordinate system
        ---------------------------------------------- OUTPUT: ------------------------------------------------------
        - B_m: Membrane strain matrix
        - B_b: Bending strain matrix
        - B_s: Transverse shear strain matrix.
            - For quadrilaterals: locking-free element with prescribed transverse shear strain profile
            - For triangles: locking-free element with reduced integration of shear terms
        - J_det: Determinant of Jacobian
    -----------------------------------------------------------------------------------------------------------------"""

    # 0 Initiate Time Measurement
    start = time.time()

    # 1 Values of Importance

    # 1.1 Nodes of element k
    e_k = ELEMENTS[k, :]
    e_k = e_k[e_k<10**5]

    # 1.2 Area of element k
    a_k = GEOMK["ak"][k]

    # 1.3 Local coordinates of nodes of element k
    NODESL = COORD["n"][2][a_k]
    v = np.array(NODESL[e_k])

    # -----------------------------------------------------------------------------------------------------------------#
    # 2 Integration of quadrilaterals
    # -----------------------------------------------------------------------------------------------------------------#
    if len(e_k) == 4:
        if order == 1:
            # 2.1 Integration of Bending and Membrane Terms

            # 2.1.1 Gauss points and weights
            gp, w = gauss_points(4,go)
            xi = gp[j]
            eta = gp[i]

            # 2.1.2 Shape Functions in xi-eta-space
            N1xe = 1/4*(1-xi)*(1-eta)/1
            N2xe = 1/4*(1+xi)*(1-eta)/1
            N3xe = 1/4*(1+xi)*(1+eta)/1
            N4xe = 1/4*(1-xi)*(1+eta)/1

            # 2.1.3 Derivatives of Shape Functions in xi-eta-space
            N1xi = float(- (1 - eta) / 4)
            N2xi = float((1 - eta) / 4)
            N3xi = float((1 + eta) / 4)
            N4xi = float(- (1 + eta) / 4)
            N1eta = float(- (1 - xi) / 4)
            N2eta = float(- (1 + xi) / 4)
            N3eta = float((1 + xi) / 4)
            N4eta = float((1 - xi) / 4)
            N1xi_eta = np.array([[N1xi], [N1eta]])
            N2xi_eta = np.array([[N2xi], [N2eta]])
            N3xi_eta = np.array([[N3xi], [N3eta]])
            N4xi_eta = np.array([[N4xi], [N4eta]])

            # 2.1.4 Gradient Matrix
            Grad_Mat = np.array([[N1xi, N2xi, N3xi, N4xi], [N1eta, N2eta, N3eta, N4eta]])

            # 2.1.5 Jacobian
            J = np.matmul(Grad_Mat, v)
            J_inv = np.linalg.inv(J)
            J_det = np.linalg.det(J)

            # 2.1.6 Derivatives of Shape Functions in x-y-Space
            N1x_y = np.matmul(J_inv, N1xi_eta)
            N2x_y = np.matmul(J_inv, N2xi_eta)
            N3x_y = np.matmul(J_inv, N3xi_eta)
            N4x_y = np.matmul(J_inv, N4xi_eta)
            N1x = float(N1x_y[0])
            N1y = float(N1x_y[1])
            N2x = float(N2x_y[0])
            N2y = float(N2x_y[1])
            N3x = float(N3x_y[0])
            N3y = float(N3x_y[1])
            N4x = float(N4x_y[0])
            N4y = float(N4x_y[1])

            # 2.1.7 Strain matrices in bending and membrane, transverse shear strain matrix for full integration
            #       without prescribed shear transverse shear strain (with locking)
            #       - Rotated global coordinates with Tk
            Tk = rotLG(k)[0]
            B_sf = np.array([[0, 0, N1x, -N1xe, 0, 0,0, 0, N2x, -N2xe, 0, 0,0, 0, N3x, -N3xe, 0, 0,0, 0, N4x, -N4xe, 0,0],
                              [0, 0, N1y, 0, -N1xe, 0, 0, 0, N2y, 0, -N2xe, 0, 0, 0, N3y, 0, -N3xe, 0, 0, 0, N4y, 0, -N4xe, 0]])
            B_b = np.array([[0,0,0, N1x,0,0,0, 0, 0, N2x, 0,0,0,0,0, N3x, 0,0, 0,0,0, N4x, 0,0],
                            [0,0,0,0, N1y, 0,0, 0,0,0, N2y, 0,0, 0,0,0, N3y, 0,0, 0,0,0, N4y,0],
                            [0,0,0,N1y, N1x,0,0,0,0,N2y,N2x,0,0,0,0,N3y,N3x,0,0,0,0,N4y,N4x,0]])
            B_m = np.array([[N1x,0,0,0, 0, 0, N2x,0, 0,0,0,0, N3x,0, 0, 0,0,0, N4x,0, 0,0,0,0],
                            [0, N1y,0, 0, 0,0,0, N2y,0, 0, 0,0,0, N3y,0, 0, 0,0,0, N4y,0,0, 0,0],
                            [N1y, N1x,0, 0,0,0,N2y,N2x,0,0,0,0,N3y,N3x,0,0,0,0,N4y,N4x,0,0,0,0]])

            # 2.2 Integration of Shear Terms: Assumed Transverse Shear Element
            Js = np.zeros((4,2,2))
            for ijs in range(4):

                # 2.2.1 Integration points in edge centers
                xis = np.array([0, 1, 0, -1])[ijs]
                etas = np.array([-1, 0, 1, 0])[ijs]

                # 2.2.2 Shape Functions in xi-etas-space
                N1s = 1 / 4 * (1 - xis) * (1 - etas)/1
                N2s = 1 / 4 * (1 + xis) * (1 - etas)/1
                N3s = 1 / 4 * (1 + xis) * (1 + etas)/1
                N4s = 1 / 4 * (1 - xis) * (1 + etas)/1

                # 2.2.3 Derivatives of Shape Functions in xis-etas-space
                N1xis = float(- (1 - etas) / 4)
                N2xis = float((1 - etas) / 4)
                N3xis = float((1 + etas) / 4)
                N4xis = float(- (1 + etas) / 4)
                N1etas = float(- (1 - xis) / 4)
                N2etas = float(- (1 + xis) / 4)
                N3etas = float((1 + xis) / 4)
                N4etas = float((1 - xis) / 4)
                N1xis_etas = np.array([[N1xis], [N1etas]])
                N2xis_etas = np.array([[N2xis], [N2etas]])
                N3xis_etas = np.array([[N3xis], [N3etas]])
                N4xis_etas = np.array([[N4xis], [N4etas]])

                # 2.1.4 Gradient Matrix
                Grad_Mats = np.array([[N1xis, N2xis, N3xis, N4xis], [N1etas, N2etas, N3etas, N4etas]])

                # 2.1.5 Jacobian
                Js[ijs][:][:] = np.matmul(Grad_Mats, v)
                Js_inv = np.linalg.inv(Js[ijs][:][:])
                Js_det = np.linalg.det(Js[ijs][:][:])

                # 2.2.4 Derivatives of Shape Functions in x-y-Space
                N1x_y = np.matmul(Js_inv, N1xis_etas)
                N2x_y = np.matmul(Js_inv, N2xis_etas)
                N3x_y = np.matmul(Js_inv, N3xis_etas)
                N4x_y = np.matmul(Js_inv, N4xis_etas)
                N1x = float(N1x_y[0])
                N1y = float(N1x_y[1])
                N2x = float(N2x_y[0])
                N2y = float(N2x_y[1])
                N3x = float(N3x_y[0])
                N3y = float(N3x_y[1])
                N4x = float(N4x_y[0])
                N4y = float(N4x_y[1])

                # 2.2.5 Bijs-Matrix
                B_ijs = np.array([[0, 0, N1x, -N1s, 0, 0, 0, 0, N2x, -N2s, 0, 0, 0, 0, N3x, -N3s, 0, 0, 0, 0, N4x, -N4s, 0, 0],
                                   [0, 0, N1y, 0, -N1s, 0,0, 0, N2y, 0, -N2s, 0,0, 0, N3y, 0, -N3s, 0,0, 0, N4y, 0, -N4s, 0]
                                   ])

                # 2.2.6 Assembly to B-bar Matrix
                if ijs == 0:
                    B_bar = B_ijs
                else:
                    B_bar = np.append(B_bar,B_ijs,axis=0)

            # 2.2.7 Auxiliary Matrices for prescribed transverse shear strain field

            Cmat = np.array([[Js[0][0, 0], Js[0][0, 1], 0, 0, 0, 0, 0, 0],
                             [Js[0][1, 0], Js[0][1, 1], 0, 0, 0, 0, 0, 0],
                             [0, 0, Js[1][0, 0], Js[1][0, 1], 0, 0, 0, 0],
                             [0, 0, Js[1][1, 0], Js[1][1, 1], 0, 0, 0, 0],
                             [0, 0, 0, 0, Js[2][0, 0], Js[2][0, 1], 0, 0],
                             [0, 0, 0, 0, Js[2][1, 0], Js[2][1, 1], 0, 0],
                             [0, 0, 0, 0, 0, 0, Js[3][0, 0], Js[3][0, 1]],
                             [0, 0, 0, 0, 0, 0, Js[3][1, 0], Js[3][1, 1]],
                             ])
            Tmat = np.array([[1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 1]])
            Pmat = np.array([[1, -1, 0, 0], [0, 0, 1, 1], [1, 1, 0, 0], [0, 0, 1, -1]])
            Amat = np.array([[1, gp[i], 0, 0], [0, 0, 1, gp[j]]])

            # 2.2.8 B_s-Matrix
            B_sats = np.matmul(J_inv,np.matmul(Amat,np.matmul(np.linalg.inv(Pmat),np.matmul(Tmat,np.matmul(Cmat,B_bar)))))

            # 2.3 Integration of Shear Terms: reduced integration
            # 2.3.1 Gauss points and weights
            # gp, w = gauss_points(4, 1)
            # xi = gp[0]
            # eta = gp[0]
            #
            # # 2.3.2 Shape Functions in xi-eta-space
            # N1xe = 1 / 4 * (1 - xi) * (1 - eta)
            # N2xe = 1 / 4 * (1 + xi) * (1 - eta)
            # N3xe = 1 / 4 * (1 + xi) * (1 + eta)
            # N4xe = 1 / 4 * (1 - xi) * (1 + eta)
            #
            # # 2.3.3 Derivatives of Shape Functions in xi-eta-space
            # N1xi = float(- (1 - eta) / 4)
            # N2xi = float((1 - eta) / 4)
            # N3xi = float((1 + eta) / 4)
            # N4xi = float(- (1 + eta) / 4)
            # N1eta = float(- (1 - xi) / 4)
            # N2eta = float(- (1 + xi) / 4)
            # N3eta = float((1 + xi) / 4)
            # N4eta = float((1 - xi) / 4)
            # N1xi_eta = np.array([[N1xi], [N1eta]])
            # N2xi_eta = np.array([[N2xi], [N2eta]])
            # N3xi_eta = np.array([[N3xi], [N3eta]])
            # N4xi_eta = np.array([[N4xi], [N4eta]])
            #
            # # 2.3.4 Gradient Matrix
            # Grad_Mat = np.array([[N1xi, N2xi, N3xi, N4xi], [N1eta, N2eta, N3eta, N4eta]])
            #
            # # 2.3.5 Jacobian
            # J = np.matmul(Grad_Mat, v)
            # J_inv = np.linalg.inv(J)
            # J_det = np.linalg.det(J)
            #
            # # 2.3.6 Derivatives of Shape Functions in x-y-Space
            # N1x_y = np.matmul(J_inv, N1xi_eta)
            # N2x_y = np.matmul(J_inv, N2xi_eta)
            # N3x_y = np.matmul(J_inv, N3xi_eta)
            # N4x_y = np.matmul(J_inv, N4xi_eta)
            # N1x = float(N1x_y[0])
            # N1y = float(N1x_y[1])
            # N2x = float(N2x_y[0])
            # N2y = float(N2x_y[1])
            # N3x = float(N3x_y[0])
            # N3y = float(N3x_y[1])
            # N4x = float(N4x_y[0])
            # N4y = float(N4x_y[1])
            #
            # # 2.1.7 Shear Strain matrix with reduced integration
            # #       - Rotated global coordinates with Tk
            # Tk = rotLG(k)[0]
            # B_sred = np.array(
            #     [[0, 0, N1x, -N1xe, 0, 0, 0, 0, N2x, -N2xe, 0, 0, 0, 0, N3x, -N3xe, 0, 0, 0, 0, N4x, -N4xe, 0, 0],
            #      [0, 0, N1y, 0, -N1xe, 0, 0, 0, N2y, 0, -N2xe, 0, 0, 0, N3y, 0, -N3xe, 0, 0, 0, N4y, 0, -N4xe, 0]])

            # Return wished for Shear matrix
            # B_s = B_sf
            B_s = B_sats
            # B_s = B_sred
    # -----------------------------------------------------------------------------------------------------------------#
    # 3 Integration of triangles
    # -----------------------------------------------------------------------------------------------------------------#
    else:
        if order == 1:
            # 3.1 Integration of Bending and Membrane Terms

            # 3.1.1 Coordinates of element nodes: delete entry 100'001
            v = np.array(NODESL[ELEMENTS[k][0:3]])

            # 3.1.2 Gauss Points
            gp, w = gauss_points(3, go)
            xi = gp[j]
            eta = gp[i]

            # 3.1.3 Shape Functions in xi-eta space
            N1xe = 1 - xi - eta
            N2xe = xi
            N3xe = eta

            # 3.1.4 Derivatives Shape Functions in xi-eta space
            N1xi = -1
            N2xi = 1
            N3xi = 0
            N1eta = -1
            N2eta = 0
            N3eta = 1

            N1xi_eta = np.array([[N1xi], [N1eta]])
            N2xi_eta = np.array([[N2xi], [N2eta]])
            N3xi_eta = np.array([[N3xi], [N3eta]])

            # 3.1.5 Gradient Matrix
            Grad_Mat = np.array([[N1xi, N2xi, N3xi], [N1eta, N2eta, N3eta]])

            # 3.1.6 Jacobian
            J = np.matmul(Grad_Mat, v)
            J_inv = np.linalg.inv(J)
            J_det = np.linalg.det(J)

            # 3.1.7 Derivatives of Shape Functions in x-y-Space
            N1x_y = np.matmul(J_inv, N1xi_eta)
            N2x_y = np.matmul(J_inv, N2xi_eta)
            N3x_y = np.matmul(J_inv, N3xi_eta)
            N1x = float(N1x_y[0])
            N1y = float(N1x_y[1])
            N2x = float(N2x_y[0])
            N2y = float(N2x_y[1])
            N3x = float(N3x_y[0])
            N3y = float(N3x_y[1])

            # 3.1.7 Strain matrices in bending and membrane, transverse shear strain matrix for full integration
            #       without prescribed shear transverse shear strain (with locking)
            #       - Rotated global coordinates with Tk
            B_sf = np.array([[0, 0, N1x, -N1xe, 0, 0, 0, 0, N2x, -N2xe, 0, 0, 0, 0, N3x, -N3xe, 0, 0],
                             [0, 0, N1y, 0, -N1xe, 0, 0, 0, N2y, 0, -N2xe, 0, 0, 0, N3y, 0, -N3xe, 0]])
            B_b = np.array([[0, 0, 0, N1x, 0, 0, 0, 0, 0, N2x, 0, 0, 0, 0, 0, N3x, 0, 0],
                            [0, 0, 0, 0, N1y, 0, 0, 0, 0, 0, N2y, 0, 0, 0, 0, 0, N3y, 0],
                            [0, 0, 0, N1y, N1x, 0, 0, 0, 0, N2y, N2x, 0, 0, 0, 0, N3y, N3x, 0]])
            B_m = np.array([[N1x, 0, 0, 0, 0, 0, N2x, 0, 0, 0, 0, 0, N3x, 0, 0, 0, 0, 0],
                            [0, N1y, 0, 0, 0, 0, 0, N2y, 0, 0, 0, 0, 0, N3y, 0, 0, 0, 0],
                            [N1y, N1x, 0, 0, 0, 0, N2y, N2x, 0, 0, 0, 0, N3y, N3x, 0, 0, 0, 0]])

            # 3.2 Integration of Shear Terms: Assumed Transverse Shear Element: constant! only dep. on a1 and a2
            Js = np.zeros((2, 2, 2))
            xisall = np.array([0.5,0])
            etasall = np.array([0, 0.5])
            for ijs in range(2):

                # 3.2.1 Integration points in edge centers
                xis = xisall[ijs]
                etas = etasall[ijs]

                # 3.2.2 Shape Functions in xi-etas-space
                N1s = 1 - xis - etas
                N2s = xis
                N3s = etas

                # 3.2.3 Derivatives of Shape Functions in xis-etas-space
                N1xis = -1
                N2xis = 1
                N3xis = 0
                N1etas = -1
                N2etas = 0
                N3etas = 1

                N1xis_etas = np.array([[N1xis], [N1etas]])
                N2xis_etas = np.array([[N2xis], [N2etas]])
                N3xis_etas = np.array([[N3xis], [N3etas]])

                # 3.1.4 Gradient Matrix
                Grad_Mats = np.array([[N1xis, N2xis, N3xis], [N1etas, N2etas, N3etas]])

                # 3.1.5 Jacobian
                Js[ijs][:][:] = np.matmul(Grad_Mats, v)
                Js_inv = np.linalg.inv(Js[ijs][:][:])
                Js_det = np.linalg.det(Js[ijs][:][:])

                # 3.2.4 Derivatives of Shape Functions in x-y-Space
                N1x_y = np.matmul(Js_inv, N1xis_etas)
                N2x_y = np.matmul(Js_inv, N2xis_etas)
                N3x_y = np.matmul(Js_inv, N3xis_etas)
                N1x = float(N1x_y[0])
                N1y = float(N1x_y[1])
                N2x = float(N2x_y[0])
                N2y = float(N2x_y[1])
                N3x = float(N3x_y[0])
                N3y = float(N3x_y[1])


                # 3.2.5 Bijs-Matrix
                B_ijs = np.array(
                    [[0, 0, N1x, -N1s, 0, 0, 0, 0, N2x, -N2s, 0, 0, 0, 0, N3x, -N3s, 0, 0],
                     [0, 0, N1y, 0, -N1s, 0, 0, 0, N2y, 0, -N2s, 0, 0, 0, N3y, 0, -N3s, 0]
                     ])

                # 3.2.6 Assembly to B-bar Matrix
                if ijs == 0:
                    B_bar = B_ijs
                else:
                    B_bar = np.append(B_bar, B_ijs, axis=0)

            # 3.2.7 Auxiliary Matrices for prescribed transverse shear strain field

            Cmat = np.array([[Js[0][0, 0], Js[0][0, 1], 0, 0],
                             [Js[0][1, 0], Js[0][1, 1], 0, 0],
                             [0, 0, Js[1][0, 0], Js[1][0, 1]],
                             [0, 0, Js[1][1, 0], Js[1][1, 1]]])
            Tmat = np.array([[1, 0, 0,0], [0,0, 0, 1]])
            Pmat = np.array([[1, 0,], [0,1]])
            Amat = np.array([[1, 0], [0, 1]])

            # 3.2.8 B_s-Matrix
            Tk = rotLG(k)[0]
            Tk = Tk[0:18,0:18]
            B_s = np.matmul(J_inv,
                               np.matmul(Amat, np.matmul(np.linalg.inv(Pmat), np.matmul(Tmat, np.matmul(Cmat, B_bar)))))

            # 3.2 Integration of Shear Terms: Reduced Integration
            # 3.2.1 Gauss Points
            # xi = 1/3
            # eta = 1/3
            #
            # # 3.2.2 Shape Functions in xi-eta space
            # N1xe = 1 - xi - eta
            # N2xe = xi
            # N3xe = eta
            #
            # # 3.2.3 Derivatives Shape Functions in xi-eta space
            # N1xi = -1
            # N2xi = 1
            # N3xi = 0
            # N1eta = -1
            # N2eta = 0
            # N3eta = 1
            # N1xi_eta = np.array([[N1xi], [N1eta]])
            # N2xi_eta = np.array([[N2xi], [N2eta]])
            # N3xi_eta = np.array([[N3xi], [N3eta]])
            #
            # # 3.2.4 Gradient Matrix
            # Grad_Mat = np.array([[N1xi, N2xi, N3xi], [N1eta, N2eta, N3eta]])
            #
            # # 3.2.5 Jacobian
            # J = np.matmul(Grad_Mat, v)
            # J_inv = np.linalg.inv(J)
            # J_det = np.linalg.det(J)
            #
            # # 3.2.6 Derivatives of Shape Functions in x-y-Space
            # N1x_y = np.matmul(J_inv, N1xi_eta)
            # N2x_y = np.matmul(J_inv, N2xi_eta)
            # N3x_y = np.matmul(J_inv, N3xi_eta)
            # N1x = float(N1x_y[0])
            # N1y = float(N1x_y[1])
            # N2x = float(N2x_y[0])
            # N2y = float(N2x_y[1])
            # N3x = float(N3x_y[0])
            # N3y = float(N3x_y[1])
            #
            # # 3.2.7 Strain matrix in transverse shear with reduced integration
            # Tk = rotLG(k)[0]
            # Tk = Tk[0:18,0:18]
            # B_s = np.array([[0, 0, N1x, -N1xe, 0, 0, 0, 0, N2x, -N2xe, 0, 0, 0, 0, N3x, -N3xe, 0, 0],
            #                  [0, 0, N1y, 0, -N1xe, 0, 0, 0, N2y, 0, -N2xe, 0, 0, 0, N3y, 0, -N3xe, 0]])


    end = time.time()
    time_B._updatetime(delta_t=end - start)
    # 4 Return B-Matrices in Global or Local Coordinate System
    if rot:
        return B_m@Tk,B_b@Tk,B_s@Tk,J_det
    else:
        return B_m,B_b,B_s,J_det


def find_b(go):
    NODESG = COORD["n"][0]
    nn = len(NODESG[:, 0])
    nk = len(ELEMENTS[:, 0])
    Bmr = {}
    Bbr = {}
    Bsr = {}
    Bmnr = {}
    Bbnr = {}
    Bsnr = {}
    J = {}
    for k in range(0, nk):
        Bmr[k] = {}
        Bbr[k] = {}
        Bsr[k] = {}
        Bmnr[k] = {}
        Bbnr[k] = {}
        Bsnr[k] = {}
        J[k] = {}
        gp, w = gauss_points(ELS[4][k], go)
        n_k = find_nodes(k)
        n_k = n_k[n_k < 100000]
        for i in range(len(gp)):
            Bmr[k][i] = {}
            Bbr[k][i] = {}
            Bsr[k][i] = {}
            Bmnr[k][i] = {}
            Bbnr[k][i] = {}
            Bsnr[k][i] = {}
            J[k][i] = {}
            for j in range(len(gp)):
                if ELS[4][k] == 3 and i == 1 and j == 1:
                    continue
                [Bmr[k][i][j], Bbr[k][i][j], Bsr[k][i][j], J[k][i][j]] = b_kij(k, i, j, go, rot=True)
                [Bmnr[k][i][j], Bbnr[k][i][j], Bsnr[k][i][j], J[k][i][j]] = b_kij(k, i, j, go, rot=False)
    B = {}
    B["Bm"] = {}
    B["Bm"]["r"] = Bmr
    B["Bm"]["nr"] = Bmnr
    B["Bb"] = {}
    B["Bb"]["r"] = Bbr
    B["Bb"]["nr"] = Bbnr
    B["Bs"] = {}
    B["Bs"]["r"] = Bsr
    B["Bs"]["nr"] = Bsnr
    B["Jdet"] = J
    return B

"""---------------------------------------- Stiffness Matrix Calculation---------------------------------------------"""

def rotLG(k):

    Tk1 = GEOMK["Tk1"][k]
    Tk2 = GEOMK["Tk2"][k]
    Tk3 = GEOMK["Tk3"][k]

    temp1 = np.append(Tk1, np.zeros((3, 3)), axis=1)
    temp2 = np.append(np.zeros((3, 3)), Tk2, axis=1)
    temp3 = np.append(temp1, temp2, axis=0)
    temp4 = np.append(temp3, np.zeros((6, 18)), axis=1)
    temp5 = np.append(np.append(np.zeros((6, 6)), temp3, axis=1), np.zeros((6, 12)), axis=1)
    temp6 = np.append(np.append(np.zeros((6, 12)), temp3, axis=1), np.zeros((6, 6)), axis=1)
    temp7 = np.append(np.zeros((6, 18)), temp3, axis=1)
    Tk = np.append(np.append(temp4, temp5, axis=0), np.append(temp6, temp7, axis=0), axis=0)

    temp1 = np.append(np.zeros((3, 3)), np.zeros((3, 3)), axis=1)
    temp2 = np.append(np.zeros((3, 3)),Tk3, axis=1)
    temp3 = np.append(temp1,temp2, axis=0)
    temp4 = np.append(temp3,np.zeros((6,6)),axis = 1)
    temp5 = np.append(np.zeros((6,6)),temp3,axis = 1)
    temp6 = np.append(temp4,temp5,axis=0)
    temp7 = np.append(temp6,np.zeros((12,12)),axis=1)
    temp8 = np.append(np.zeros((12, 12)),temp6, axis=1)
    Tkr = np.append(temp7,temp8,axis=0)

    return Tk,Tkr
    # return np.identity(24),np.identity(24)


def dh_kij(e_kij,s_kij, k, i, j, cm_k):
    Dmh = np.zeros((3, 3))
    Dbh = np.zeros((3, 3))
    Dmbh = np.zeros((3, 3))
    Dsh = np.zeros((2, 2))
    t_k = GEOMK["t"][k]
    nlk = GEOMK["nlk"][k]
    for l in range(nlk):
        z = -t_k / 2 + (2 * l + 1) * t_k / (2 * nlk)
        E = MATK["Ec"][k]
        v = MATK["vc"][k]

        Dp = get_et(cm_k, e_kij[l,0:5],s_kij[l], k, l, i, j)
        Ds = np.array([[5 / 6 * (E + E) / (4 * (1 + v)), 0], [0, 5 / 6 * (E + E) / (4 * (1 + v))]])
        Dmh_l=Dp
        Dmbh_l= -z*Dp
        Dbh_l = z*z*Dp
        Dsh_l = Ds
        Dmh = Dmh + Dmh_l * t_k / nlk
        Dbh = Dbh + Dbh_l * t_k / nlk
        Dmbh = Dmbh + Dmbh_l * t_k / nlk
        Dsh = Dsh + Dsh_l * t_k / nlk
    return Dmh,Dmbh,Dbh,Dsh


def k_k(Bm_k,Bb_k,Bs_k,Jdet_k,e_k,s_k, k, cm_k):
    ne_k = ELEMENTS[k, :]
    ne_k = ne_k[ne_k<10**5]
    a_k = GEOMK["ak"][k]
    NODESL = COORD["n"][2][a_k]
    if len(ne_k) == 4:
        v = np.array(NODESL[ELEMENTS[k]])
        Tkr = rotLG(k)[1]
    else:
        v = np.array(NODESL[ELEMENTS[k][0:3]])
        Tkr = rotLG(k)[1]
        Tkr = Tkr[0:18,0:18]

    """-------------------- Integration of Membrane,Bending,Shear and Coupling Stiffness Matrix ---------------------"""
    gp, w = gauss_points(ELS[4][k],gauss_order)
    Kbe = np.zeros((len(v) * 6, len(v) * 6))
    Kme = np.zeros((len(v) * 6, len(v) * 6))
    Kmbe= np.zeros((len(v) * 6, len(v) * 6))
    Kse = np.zeros((len(v) * 6, len(v) * 6))
    A_k = 0
    for i in range(len(gp)):
        for j in range(len(gp)):
            if ELS[4][k] == 3 and i == 1 and j == 1:
                continue
            Bm = Bm_k[i][j]
            Bb = Bb_k[i][j]
            Bs = Bs_k[i][j]
            Jdet = Jdet_k[i][j]
            # D-Matrices
            Dmh,Dmbh,Dbh,Dsh = dh_kij(e_k[:,i,j,:],s_k[:,i,j], k, i, j, cm_k)
            Kbe = np.add(Kbe, w[i] * w[j] * Jdet * np.matmul(np.transpose(Bb), np.matmul(Dbh, Bb)))
            Kme = np.add(Kme, w[i] * w[j] * Jdet * np.matmul(np.transpose(Bm), np.matmul(Dmh, Bm)))
            Kmbe= np.add(Kmbe, w[i] * w[j] * Jdet * np.matmul(np.transpose(Bm),np.matmul(Dmbh, Bb)))
            Kse = np.add(Kse, w[i] * w[j] * Jdet * np.matmul(np.transpose(Bs), np.matmul(Dsh, Bs)))
            A_k+=Jdet*w[i]*w[j]

    """--------------------------------- Stiffness Term for rotational DOF ------------------------------------------"""
    n_k = ELEMENTS[k]
    if n_k.any() in copln:
        iscoplk = 1
    else:
        iscoplk = 0
    """--------------------------------- Assembly of entire Stiffness Matrix ----------------------------------------"""
    Ke = Kme+Kbe+Kse+Kmbe+np.transpose(Kmbe)+iscoplk*A_k*GEOMK["t"][k]*33600*Tkr*10**-8
    if Jdet < 0:
        print(ELS[0][k])
        print(ELS[0][k+1])
        print(Jdet)
    return Ke


def k_glob(B,e,s, cmk):
    """ ----------------------------------- Create global stiffness matrix ------------------------------------------
        --------------------------------------------    INPUT: ------------------------------------------------------
        - u: Global displacement vector
        - e: Global strain matrix
        - cmk
            - 1 --> Linear Elasticity
            - 2 --> CMM-
            - 3 --> CMM
            given per element
        --------------------------------------------- OUTPUT:--------------------------------------------------------
        - K: Global stiffness matrix
    -----------------------------------------------------------------------------------------------------------------"""
    # 0 Initiate Time Measurement
    start = time.time()

    """------------------------------------------ Calculation and Assembly ------------------------------------------"""
    NODESG = COORD["n"][0]
    nn = len(NODESG[:, 0])
    nk = len(ELEMENTS[:, 0])
    K = np.zeros((6 * nn, 6 * nn))
    for k in range(nk):
        """--------------------------------- Local Stiffness Matrix in Global Coordinates ---------------------------"""
        e_k = e[k,:,:,:]
        s_k = s[k]
        Ke = k_k(B["Bm"]["r"][k],B["Bb"]["r"][k],B["Bs"]["r"][k],B["Jdet"][k],e_k,s_k, k, cmk[k])
        nodes = ELEMENTS[k, :][ELEMENTS[k, :] < 10**5]
        """------------------------------------ Assemble to global Stiffness Matrix ---------------------------------"""
        K = m_assemble(Ke, K, nodes)
    end = time.time()
    time_K._updatetime(delta_t=end - start)
    return K


"""-------------------------------------------- Static Condensation--------------------------------------------------"""


def c_dof():
    """ --------------------------------- Create Vector of condensed DOFs -------------------------------------------
        --------------------------------------------    INPUT: ------------------------------------------------------
        - Boundary conditions from Input file
        --------------------------------------------- OUTPUT:--------------------------------------------------------
        - cDOF --> Vector of DOFs to be condensed
    -----------------------------------------------------------------------------------------------------------------"""
    numb = len(BC[:,0])
    cDOF = np.array([])
    cVAL = np.array([])
    " ----------------------------------- condensed DOFs from Boundary Conditions -------------------------------------"
    for i in range(numb):
        xmin = BC[i,0]
        xmax = BC[i,1]
        ymin = BC[i,2]
        ymax = BC[i,3]
        zmin = BC[i,4]
        zmax = BC[i,5]
        nodes_i = find_node_range(xmin,xmax,ymin,ymax,zmin,zmax)
        if BC[i,6]!=1234:
            cDOF = np.append(cDOF, nodes_i * 6)
            cVAL = np.append(cVAL, BC[i,6]*np.ones_like(nodes_i))
        if BC[i,7]!=1234:
            cDOF = np.append(cDOF, nodes_i * 6 + 1)
            cVAL = np.append(cVAL, BC[i, 7]*np.ones_like(nodes_i))
        if BC[i, 8]!=1234:
            cDOF = np.append(cDOF, nodes_i * 6 + 2)
            cVAL = np.append(cVAL, BC[i, 8]*np.ones_like(nodes_i))
        if BC[i, 9]!=1234:
            cDOF = np.append(cDOF, nodes_i * 6 + 3)
            cVAL = np.append(cVAL, BC[i, 9]*np.ones_like(nodes_i))
        if BC[i, 10]!=1234:
            cDOF = np.append(cDOF, nodes_i * 6 + 4)
            cVAL = np.append(cVAL, BC[i, 10]*np.ones_like(nodes_i))
        if BC[i, 11]!=1234:
            cDOF = np.append(cDOF, nodes_i * 6 + 5)
            cVAL = np.append(cVAL, BC[i, 11]*np.ones_like(nodes_i))
    " ------------------------------------------------ Sort and return ------------------------------------------------"
    indeces = np.argsort(cDOF)
    cDOF = cDOF[indeces]
    cVAL = cVAL[indeces]
    # cDOF = np.sort(cDOF)
    # cDOF = np.unique(cDOF)
    cDOF= cDOF.astype(int)
    return cDOF,cVAL


def v_stat_con(v, cDOF, cVAL):
    """ -------------------------------------- Static condensation --------------------------------------------------
        --------------------------------------------    INPUT: ------------------------------------------------------
        - v         --> (Force) Vector to be condensed
        - cond_DOF  --> Vector of DOFs to be condensed
        --------------------------------------------- OUTPUT:--------------------------------------------------------
        - v         --> Condensed (force) vector
    -----------------------------------------------------------------------------------------------------------------"""
    # v = np.delete(v, cDOF, 0)
    for i in range(len(cDOF)):
        v[cDOF[i]] = cVAL[i]
    return v


def m_stat_con(M, cDOF):
    """ -------------------------------------- Static condensation --------------------------------------------------
        --------------------------------------------    INPUT: ------------------------------------------------------
        - M         --> (Stiffness) Matrix to be condensed
        - cDOF      --> Vector of DOFs to be condensed
        --------------------------------------------- OUTPUT:--------------------------------------------------------
        - M         --> Condensed (stiffness) matrix
    -----------------------------------------------------------------------------------------------------------------"""
    # M = np.delete(M, cDOF, 0)
    # M = np.delete(M, cDOF, 1)
    for i in range(len(cDOF)):
        M[cDOF[i],:] = np.zeros_like(M[cDOF[i],:])
        M[cDOF[i],cDOF[i]] = 1
    return M


"""------------------------------------------------- Assembly--------------------------------------------------------"""


def m_assemble(Ke, K, nodes):
    """ ------------------ Assemble local nodal stiffness matrix to global stiffness matix --------------------------
        --------------------------------------------    INPUT: ------------------------------------------------------
        - Ke    --> Local nodal stiffness matrix of regarded finite element
        - K     --> Global stiffness matrix
        - nodes --> Nodes of regarded element
        --------------------------------------------- OUTPUT:--------------------------------------------------------
        -K: Global stiffness matrix with contributions of the regarded element
    -----------------------------------------------------------------------------------------------------------------"""

    for i in range(int(len(Ke[:, 0]) / 6)):
        for j in range(int(len(Ke[:, 0]) / 6)):
            Ke = np.array(Ke)
            K[nodes[i] * 6:nodes[i] * 6 + 6, nodes[j] * 6:nodes[j] * 6 + 6] = K[nodes[i] * 6:nodes[i] * 6 + 6,
                                                                              nodes[j] * 6:nodes[j] * 6 + 6] \
                                                                              + Ke[i * 6:i * 6 + 6, j * 6:j * 6 + 6]
    return K


def v_assemble(ve, v, nodes):
    """ ------------------------------ Assemble local nodal vector to global vector --------------------------------
        --------------------------------------------    INPUT: ------------------------------------------------------
        - ve    --> Local nodal (force) vector of regarded finite element. Column vector ndarray(n,1)
        - v     --> Global (force) vector. Column vector ndarray(n,1)
        - nodes --> Nodes of regarded element
        --------------------------------------------- OUTPUT:--------------------------------------------------------
        -v: Global (force) vector with contributions of the regarded element. Column vector ndarray(n,1)
    -----------------------------------------------------------------------------------------------------------------"""
    for i in range(int(ve.size/6)):
        v[nodes[i]*6:nodes[i]*6+6] = v[nodes[i]*6:nodes[i]*6+6] + ve[i*6:i*6+6]
    return v


def f_assemble(ils):
    """ --------------------------------- Create Vector of applied forces -------------------------------------------
        --------------------------------------------    INPUT: ------------------------------------------------------
        - Applied force conditions from Input file
        --------------------------------------------- OUTPUT:--------------------------------------------------------
        - f_e --> vector of applied outer forces per DOF
    -----------------------------------------------------------------------------------------------------------------"""

    " -------------------------------------------- Element Loads ------------------------------------------------------"

    NODESG = COORD["n"][0]
    nn = len(NODESG[:, 0])
    f_e = np.zeros((1,6*nn))
    nlel = len(Load_el[ils][:, 0])
    for i in range(nlel):
        xmin = Load_el[ils][i, 0]
        xmax = Load_el[ils][i, 1]
        ymin = Load_el[ils][i, 2]
        ymax = Load_el[ils][i, 3]
        zmin = Load_el[ils][i,4]
        zmax = Load_el[ils][i,5]
        dir = int(Load_el[ils][i, 6])

        els_i = find_el_range(xmin,xmax,ymin,ymax,zmin,zmax)

        q = Load_el[ils][i, 7]
        q_e = np.zeros((6, 1))
        q_e[dir-1][0] = q

        for k in range(len(els_i)):
            el = els_i[k]
            nodes = find_nodes(el)
            nodes = nodes[nodes<10**5]
            J_det = 0
            q_e1 = np.zeros((6, 1))
            q_e2 = np.zeros((6, 1))
            q_e3 = np.zeros((6, 1))
            q_e4 = np.zeros((6, 1))
            for ii in range(2):
                for jj in range(2):
                    if ELS[4][k] == 4:
                        J_det_ij = jacobi(el,ii,jj,2)[4]
                        J_det += J_det_ij
                        xi = gauss_points(4,2)[0][jj]
                        eta = gauss_points(4,2)[0][ii]
                        wxi = gauss_points(4,2)[1][jj]
                        weta = gauss_points(4,2)[1][ii]

                        n1 = 1 / 4 * (1 - xi) * (1 - eta)
                        n2 = 1 / 4 * (1 + xi) * (1 - eta)
                        n3 = 1 / 4 * (1 + xi) * (1 + eta)
                        n4 = 1 / 4 * (1 - xi) * (1 + eta)
                    else:
                        if jj == 1 and ii == 1:
                            continue
                        else:
                            J_det_ij = jacobi(el, ii, jj, 2)[4]
                            J_det += J_det_ij
                            xi = gauss_points(3,2)[0][jj]
                            eta = gauss_points(3,2)[0][ii]
                            wxi = gauss_points(3,2)[1][jj]
                            weta = gauss_points(3,2)[1][ii]

                            n1 = 1 - xi - eta
                            n2 = xi
                            n3 = eta
                            n4 = 0

                    N1 = np.zeros((6,6))
                    np.fill_diagonal(N1,n1)
                    N2 = np.zeros((6,6))
                    np.fill_diagonal(N2,n2)
                    N3 = np.zeros((6,6))
                    np.fill_diagonal(N3,n3)
                    N4 = np.zeros((6,6))
                    np.fill_diagonal(N4,n4)

                    q_e1 += N1@q_e*J_det_ij*wxi*weta
                    q_e2 += N2@q_e*J_det_ij*wxi*weta
                    q_e3 += N3@q_e*J_det_ij*wxi*weta
                    q_e4 += N4@q_e*J_det_ij*wxi*weta
                    q_e_all = [q_e1,q_e2,q_e3,q_e4]
            for n in range(len(nodes)):
                node = nodes[n]
                for vecit in range(6):
                    f_e[0][node*6+vecit] += q_e_all[n][vecit]
    " --------------------------------------------- Nodal Loads -------------------------------------------------------"
    nln = len(Load_n[ils][:, 0])
    for i in range(nln):
        xmin = Load_n[ils][i, 0]
        xmax = Load_n[ils][i, 1]
        ymin = Load_n[ils][i, 2]
        ymax = Load_n[ils][i, 3]
        zmin = Load_n[ils][i, 4]
        zmax = Load_n[ils][i, 5]
        dir = Load_n[ils][i, 6]

        nodes_i = find_node_range(xmin,xmax,ymin,ymax,zmin,zmax)
        # if load is not acting in any existing node: create equivalent force and moment in closest node
        if nodes_i.size == 0:
            from Mesh_gmsh import ms
            step = ms/10
            count = 1
            while nodes_i.size == 0:
                nodes_i = find_node_range(xmin-count*step,xmax+count*step,ymin-count*step,ymax+count*step,zmin-count*step,zmax+count*step)
                count += 1
            if nodes_i.size > 1:
                nodes_i = np.array([nodes_i[0]])
            coordsi = NODESG[nodes_i[0],:]
            diffcordsi = np.array([(xmin+xmax)/2, (ymin+ymax)/2,(zmin+zmax)/2])-coordsi
        else:
            diffcordsi = np.array([0,0,0])
        load_i = Load_n[ils][i,7]
        for j in range(len(nodes_i)):
            node_j = nodes_i[j]
            coord = NODESG[node_j,:]
            node = find_node(coord)
            if dir == 1:
                DOF = node*6
            elif dir == 2:
                DOF = node*6+1
            elif dir == 3:
                DOF = node*6+2
            elif dir == 4:
                DOF = node*6+3
            elif dir == 5:
                DOF = node*6+4
            elif dir == 6:
                DOF = node*6+5
            f_e[0,DOF] = f_e[0,DOF] + load_i
        # Add bending moment for equivalent nodal force
            fii = np.zeros(3)
            fii[int(dir)-1]=load_i
            moments = np.cross(diffcordsi,fii)
            for im in range(0,3):
                f_e[0,int(node*6+3+im)] = moments[im]
    f_e = np.transpose(f_e)
    return f_e


def f0_assemble(B,s0,go):
    """ -------------------- Create Vector of external forces caused by internal stresses ---------------------------
        --------------------------------------------    INPUT: ------------------------------------------------------
        - Internal Stresses (caused by shrinkage)
        --------------------------------------------- OUTPUT:--------------------------------------------------------
        - f_e0 --> vector of applied outer forces per DOF
    -----------------------------------------------------------------------------------------------------------------"""
    NODESG = COORD["n"][0]
    nn = len(NODESG[:, 0])
    nk = len(ELEMENTS[:, 0])
    sh0 = np.zeros((nk, go, go, 8))
    f_0 = np.zeros((6*nn,1))
    for k in range(0,nk):
        gp, w = gauss_points(ELS[4][k], go)
        n_k = find_nodes(k)
        n_k = n_k[n_k<100000]
        f0_k = np.zeros((1,6 * len(n_k)))
        for i in range(len(gp)):
            for j in range(len(gp)):
                if ELS[4][k] == 3 and i == 1 and j == 1:
                    continue
                B_kij = np.append(np.append(B["Bm"]["r"][k][i][j],B["Bb"]["r"][k][i][j],axis=0),B["Bs"]["r"][k][i][j],axis=0)
                sh0_kij = find_sh0_kij(s0[k],k,i,j)
                sh0[k,i,j]=sh0_kij
                f0_k = f0_k - w[i] * w[j] * B["Jdet"][k][i][j] * np.transpose(B_kij)@sh0_kij
        f_0 = v_assemble(np.transpose(f0_k), f_0, n_k)
    return sh0,f_0
"""-------------------------------------------- Auxiliary Functions--------------------------------------------------"""

def find_node(coord):
    """ -------------------------------- Find node number for given coordinates -------------------------------------
        --------------------------------------------    INPUT: ------------------------------------------------------
        - coord --> coordinates of regarded node
        --------------------------------------------- OUTPUT:--------------------------------------------------------
        - nr    --> node number of regarded node
    -----------------------------------------------------------------------------------------------------------------"""
    NODESG = COORD["n"][0]
    for j in range(len(NODESG[:, 0])):
        diff = abs(np.add(coord, -NODESG[j, :]))
        if max(diff) < pow(10, -10):
            nr = j
    return nr


def find_nodes(el):
    """ ------------------------------------ Nodes connected to a given Element ----------------------------------------
            --------------------------------------------    INPUT: -----------------------------------------------------
            - el --> element number
            --------------------------------------------- OUTPUT:-------------------------------------------------------
            - nodes    --> vector of nodes connected to given element
    -----------------------------------------------------------------------------------------------------------------"""
    nodes = ELEMENTS[int(el)]
    return nodes


def find_node_range(xmin, xmax, ymin, ymax, zmin, zmax):
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


def find_el_range(xmin,xmax,ymin,ymax,zmin,zmax):
    centx = COORD["c"][0][:,0]
    centy = COORD["c"][0][:,1]
    centz = COORD["c"][0][:,2]

    ind1 = np.array(np.where(centx<=xmax)).ravel()
    ind2 = np.array(np.where(centx>=xmin)).ravel()
    indx = np.intersect1d(ind1,ind2)

    ind1 = np.array(np.where(centy <= ymax)).ravel()
    ind2 = np.array(np.where(centy >= ymin)).ravel()
    indy = np.intersect1d(ind1, ind2)

    ind1 = np.array(np.where(centz <= zmax)).ravel()
    ind2 = np.array(np.where(centz >= zmin)).ravel()
    indz = np.intersect1d(ind1, ind2)

    indxy  = np.intersect1d(indx,indy)
    ind = np.intersect1d(indxy,indz)
    return ind


def find_el(node):
    """ ------------------------------------ Elements connected to a given node ----------------------------------------
            --------------------------------------------    INPUT: -----------------------------------------------------
            - node --> node number
            --------------------------------------------- OUTPUT:-------------------------------------------------------
            - ki    --> vector of elements connected to node
    -----------------------------------------------------------------------------------------------------------------"""
    els = []
    for k in range(len(ELEMENTS[:,1])):
        nodes = ELEMENTS[k,:]
        if node in nodes:
            els = np.append(els,k)
    return els


def find_dofs_n(node):
    """ ----------------------------------- returns DOFs belonging to given node --------------------------------------
            --------------------------------------------    INPUT: -----------------------------------------------------
            - nodes --> node number
            --------------------------------------------- OUTPUT:-------------------------------------------------------
            - DOFS    --> vector DOFS belongin to given node
    -----------------------------------------------------------------------------------------------------------------"""
    dofs = np.zeros(6,dtype=int)
    dofs[0:6]=[node*6,node*6+1,node*6+2,node*6+3,node*6+4,node*6+5]
    return dofs


def find_dofs_k(el):
    """ ----------------------------------- returns DOFs belonging to given element --------------------------------------
            --------------------------------------------    INPUT: -----------------------------------------------------
            - el --> element number
            --------------------------------------------- OUTPUT:-------------------------------------------------------
            - DOFS    --> vector DOFS belongin to given element
    -----------------------------------------------------------------------------------------------------------------"""
    nodes = find_nodes(el)
    nodes = nodes[nodes < 10**5]
    dofs = np.zeros(6*len(nodes),dtype=int)
    for n in range(len(nodes)):
        dofs[n*6:n*6+6] = [nodes[n] * 6, nodes[n] * 6 + 1, nodes[n] * 6 + 2, nodes[n] * 6 + 3, nodes[n] * 6 + 4, nodes[n]*6+5]
    return dofs


def find_v_el(v, k):
    """ ---------------------- Calculate values of a vector at all nodes of regarded element-------------------------
        --------------------------------------------    INPUT: ------------------------------------------------------
        - v          --> Vector of searched values
        - el_nr      --> Element index of regarded element
        --------------------------------------------- OUTPUT:--------------------------------------------------------
        - v_nodes    --> Values of v at DOFs of nodes of element with index k
    -----------------------------------------------------------------------------------------------------------------"""
    nodes = ELEMENTS[k, :]
    nodes = nodes[nodes < 10 ** 5]
    DOFS = [0]
    for j in range(len(nodes)):
        DOFS = np.append(DOFS, [nodes[j] * 6, nodes[j] * 6 + 1, nodes[j] * 6 + 2, nodes[j] * 6 + 3, nodes[j] * 6 + 4, nodes[j]*6+5])
    DOFS = np.delete(DOFS, 0)
    v_nodes = v[DOFS]
    return v_nodes


def find_fi(B,sh):
    """ ----------------------------------- Calculate vector of inner forces-----------------------------------------
        --------------------------------------------    INPUT: ------------------------------------------------------
        - e_type
            - 1 --> Linear Elasticity
            - 2 --> CMM
        --------------------------------------------- OUTPUT:--------------------------------------------------------
        - fint  --> Vector of inner forces
    -----------------------------------------------------------------------------------------------------------------"""

    NODESG = COORD["n"][0]
    nel = len(ELEMENTS[:,0])
    fint = [np.zeros(len(NODESG[:, 0]) * 6)]
    fint = np.transpose(fint)
    for k in range(nel):
        nodes = ELEMENTS[k, :]
        nodes = nodes[nodes < 10 ** 5]
        DOFS = np.zeros(len(nodes) * 6, int)
        for j in range(len(nodes)):
            DOFS[j * 6] = nodes[j] * 6
            DOFS[j * 6 + 1] = nodes[j] * 6 + 1
            DOFS[j * 6 + 2] = nodes[j] * 6 + 2
            DOFS[j * 6 + 3] = nodes[j] * 6 + 3
            DOFS[j * 6 + 4] = nodes[j] * 6 + 4
            DOFS[j * 6 + 5] = nodes[j] * 6 + 5

        gp, w = gauss_points(ELS[4][k], gauss_order)
        fint_e = np.zeros((ELS[4][k]*6,1))
        for i in range(len(gp)):
            for j in range(len(gp)):
                if ELS[4][k] == 3 and i == 1 and j == 1:
                    continue
                sh_kij = np.array(sh[k][i][j][:]).transpose()
                sh_kij = np.ndarray.reshape(sh_kij,8,1)
                # [Bm, Bb, Bs, Jdet] = b_kij(k, i, j, gauss_order, rot=True)
                Bm = B["Bm"]["r"][k][i][j]
                Bb = B["Bb"]["r"][k][i][j]
                Bs = B["Bs"]["r"][k][i][j]
                Jdet = B["Jdet"][k][i][j]
                B_kij = np.append(np.append(Bm,Bb,axis=0),Bs,axis=0)
                fint_e = np.add(fint_e, w[i] * w[j] * Jdet * np.transpose(B_kij)@sh_kij)
        fint = v_assemble(fint_e, fint, nodes)



    return fint


"""----------------------------------------------- Find Strains -----------------------------------------------------"""


def find_eh_kij(Bm_kij,Bb_kij,Bs_kij,u_k,k,i,j,go):
    # from numpy import dot
    # from numpy.linalg import norm
    #
    # # 0 Initiate Time Measurement
    # start = time.time()
    #
    # # 1 Values of Importance
    #
    # # 1.1 Nodes of element k
    # e_k = ELEMENTS[k, :]
    # e_k = e_k[e_k < 10 ** 5]
    #
    # # 1.2 Area of element k
    # a_k = GEOMK["ak"][k]
    #
    # # 1.3 Local coordinates of nodes of element k
    # NODESL = COORD["n"][2][a_k]
    # v = np.array(NODESL[e_k])
    #
    # def vcos(a, b):
    #     cos_ab = dot(a, b) / (norm(a) * norm(b))
    #     return cos_ab
    #
    # n1 = v[1, :]
    # n2 = v[2, :]
    # x = n2 - n1
    # phi = acos(vcos(x, [1, 0]))

    Tk = rotLG(k)[0]
    if ELS[4][k] == 3:
        Tk = Tk[0:18,0:18]
    u_k = Tk@u_k

    emh_kij = np.matmul(Bm_kij, u_k)
    ebh_kij = np.matmul(Bb_kij, u_k)
    esh_kij = np.matmul(Bs_kij, u_k)
    # esh_kij = np.matmul(Bs_kij, np.linalg.inv(Tk) @ u_k)
    # esh_kij = np.array([[1,1],[-1,1]])@esh_kij
    # esh_kij = np.array([[cos(phi),sin(phi)],[-sin(phi),cos(phi)]])@esh_kij
    eh_kij = np.append(np.append(emh_kij, ebh_kij, axis=0), esh_kij, axis=0)
    return eh_kij


def find_eh(B,u,go):
    # 0 Initiate Time Measurement
    start = time.time()

    num_elements = len(ELEMENTS[:, 0])
    eh = np.zeros((num_elements, go, go, 8))
    for k in range(num_elements):
        u_k = find_v_el(u, k)
        for i in range(go):
            for j in range(go):
                if ELS[4][k] == 3 and i == 1 and j == 1:
                    eh[k][i][j][:] = -10**5*np.ones_like(eh[k][i][j][:])
                else:
                    eh_kij = find_eh_kij(B["Bm"]["nr"][k][i][j],B["Bb"]["nr"][k][i][j],B["Bs"]["nr"][k][i][j],u_k, k, i, j, go)
                    eh[k][i][j][:] = np.transpose(eh_kij)
    end = time.time()
    time_eh._updatetime(delta_t=end - start)
    return eh


def find_sh(s,go):
    # 0 Initiate Time Measurement
    start = time.time()

    num_elements = len(ELEMENTS[:, 0])
    sh = np.zeros((num_elements, go, go, 8))
    for k in range(num_elements):
        nlk = GEOMK["nlk"][k]
        t_k = GEOMK["t"][k]
        for i in range(go):
            for j in range(go):
                if ELS[4][k] == 3 and i == 1 and j == 1:
                    sh[k][i][j][:] = -10**5*np.ones_like(sh[k][i][j][:])
                else:
                    sh_kij = np.zeros((8,1))
                    for l in range(nlk):
                        if go == 2:
                            st = s[k][l][i][j][0]
                            s_klij = np.array([st.sx.real,st.sy.real,st.txy.real,st.txz.real,st.tyz.real])
                        elif go == 1:
                            if ELS[4][k] == 3:
                                s_klij = np.array([np.mean([s[k][l][0][0][0].sx.real,s[k][l][1][0][0].sx.real,s[k][l][0][1][0].sx.real]),
                                                   np.mean([s[k][l][0][0][0].sy.real,s[k][l][1][0][0].sy.real,s[k][l][0][1][0].sy.real]),
                                                   np.mean([s[k][l][0][0][0].txy.real,s[k][l][1][0][0].txy.real,s[k][l][0][1][0].txy.real]),
                                                   np.mean([s[k][l][0][0][0].txz.real,s[k][l][1][0][0].txz.real,s[k][l][0][1][0].txz.real]),
                                                   np.mean([s[k][l][0][0][0].tyz.real,s[k][l][1][0][0].tyz.real,s[k][l][0][1][0].tyz.real])
                                                   ])
                            else:
                                s_klij = np.array([np.mean([s[k][l][0][0][0].sx.real,s[k][l][1][0][0].sx.real,s[k][l][0][1][0].sx.real,s[k][l][1][1][0].sx.real]),
                                                   np.mean([s[k][l][0][0][0].sy.real,s[k][l][1][0][0].sy.real,s[k][l][0][1][0].sy.real,s[k][l][1][1][0].sy.real]),
                                                   np.mean([s[k][l][0][0][0].txy.real,s[k][l][1][0][0].txy.real,s[k][l][0][1][0].txy.real,s[k][l][1][1][0].txy.real]),
                                                   np.mean([s[k][l][0][0][0].txz.real,s[k][l][1][0][0].txz.real,s[k][l][0][1][0].txz.real,s[k][l][1][1][0].txz.real]),
                                                   np.mean([s[k][l][0][0][0].tyz.real,s[k][l][1][0][0].tyz.real,s[k][l][0][1][0].tyz.real,s[k][l][1][1][0].tyz.real])
                                                   ])
                        s_klij = np.ndarray.reshape(s_klij, 5, 1)
                        z = -t_k / 2 + (2 * l + 1) * t_k / (2 * nlk)
                        S = np.array([[1, 0, 0, -z, 0, 0, 0, 0],
                                      [0, 1, 0, 0, -z, 0, 0, 0],
                                      [0, 0, 1, 0, 0, -z, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 1, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 1]])
                        sh_kij = sh_kij + np.transpose(S)@s_klij*t_k/nlk
                    sh[k][i][j][:] = sh_kij.reshape(8,)
    end = time.time()
    time_sh._updatetime(delta_t=end - start)
    return sh


def find_e_klij(eh_kij, k, l, i, j):
    """ ------------------------------------------- Shell Elements --------------------------------------------------
        --------------------------- Find strains in element k in layer l and gauss point (i,j) -------------------"""
    t_k = GEOMK["t"][k]
    nlk = GEOMK["nlk"][k]
    z = -t_k/2+(2*l+1)*t_k/(2 * nlk)
    S = np.array([[1, 0, 0, -z, 0, 0, 0, 0],
                  [0, 1, 0, 0, -z, 0, 0, 0],
                  [0, 0, 1, 0, 0, -z, 0, 0],
                  [0, 0, 0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 0, 0, 1]])
    e_klij = np.matmul(S,eh_kij)
    e_klij[e_klij == 0] = 10**-13
    return e_klij


def find_e(e0,eh,go):
    """ ------------------------------------------- Membrane Elements --------------------------------------------------
        ------------------------------------- Calculate element strains ---------------------------------------------
        --------------------------------------------    INPUT: ------------------------------------------------------
        - u         --> Node deformations
        --------------------------------------------- OUTPUT:--------------------------------------------------------
        - e        --> Matrix of element strains [e_xi e_yi g_xyi]
        - ex       --> Element normal strains in x-Direction
        - ey       --> Element normal strains in y-Direction
        - gxy      --> Element shear strains
        - e1       --> Element principal tensile strains
        - e3       --> Element principal compressive strains
        - th       --> Element principal directions
    -----------------------------------------------------------------------------------------------------------------"""
    # 0 Initiate Time Measurement
    start = time.time()

    nel = len(ELEMENTS[:, 0])
    nlk = max(GEOMK["nlk"])
    e = np.zeros((nel, nlk, go, go, 5))
    ex = np.zeros((nel, nlk, go, go))
    ey = np.zeros((nel, nlk, go, go))
    gxy = np.zeros((nel, nlk, go, go))
    for k in range(nel):
        for l in range(nlk):
            for i in range(go):
                for j in range(go):
                    if ELS[4][k] == 3 and i == 1 and j == 1:
                        e[k][l][i][j][:] = -10**5*np.ones_like(e[k][l][i][j][:])
                        ex[k][l][i][j] = -10**5*np.ones_like(ex[k][l][i][j])
                        ey[k][l][i][j] = -10**5*np.ones_like(ey[k][l][i][j])
                        gxy[k][l][i][j] = -10**5*np.ones_like(gxy[k][l][i][j])

                    else:
                        eh_kij = eh[k, i, j, :]
                        e_klij = find_e_klij(eh_kij, k, l, i, j)
                        e0_klij = e0[k][l][i][j][:]
                        e[k][l][i][j][:]=np.transpose(e_klij)-np.transpose(e0_klij)
                        ex[k][l][i][j] = e_klij[0]-e0_klij[0]
                        ey[k][l][i][j] = e_klij[1]-e0_klij[1]
                        gxy[k][l][i][j] = e_klij[2]-e0_klij[2]

    end = time.time()
    time_strain._updatetime(delta_t=end - start)
    return e, ex, ey, gxy


def find_e0(go):
    nel = len(ELEMENTS[:, 0])
    nlk = max(GEOMK["nlk"])
    e = np.zeros((nel, nlk, go, go, 5))
    ex = np.zeros((nel, nlk, go, go))
    ey = np.zeros((nel, nlk, go, go))
    gxy = np.zeros((nel, nlk, go, go))
    for k in range(nel):
        for l in range(nlk):
            for i in range(go):
                for j in range(go):
                    if ELS[4][k] == 3 and i == 1 and j == 1:
                        e[k][l][i][j][:] = -10**5*np.ones_like(e[k][l][i][j][:])
                        ex[k][l][i][j] = -10**5*np.ones_like(ex[k][l][i][j])
                        ey[k][l][i][j] = -10**5*np.ones_like(ey[k][l][i][j])
                        gxy[k][l][i][j] = -10**5*np.ones_like(gxy[k][l][i][j])
                    else:
                        e_klij = np.array([1,1,0,0,0])*MATK["ecs"][k]
                        e[k][l][i][j][:]=np.transpose(e_klij)
                        ex[k][l][i][j] = e_klij[0]
                        ey[k][l][i][j] = e_klij[1]
                        gxy[k][l][i][j] = e_klij[2]

    return e, ex, ey, gxy


def find_sh0_kij(s0_k,k,i,j):
    t_k = GEOMK["t"][k]
    nlk = GEOMK["nlk"][k]
    dz_k = t_k/nlk
    for l in range(0,nlk):
        z = -t_k/2+(2*l+1)*t_k/(2 * nlk)
        S = np.array([[1, 0, 0, -z, 0, 0, 0, 0],
                      [0, 1, 0, 0, -z, 0, 0, 0],
                      [0, 0, 1, 0, 0, -z, 0, 0],
                      [0, 0, 0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 0, 0, 1]])
        s0_klij = np.append(s0_k[l][i][j],np.array([0,0]),axis=0)
        sh0_klij = np.transpose(S)@np.transpose(s0_klij)*dz_k
        if l == 0:
            sh0_kij = sh0_klij
        else:
            sh0_kij = sh0_kij + sh0_klij
    return sh0_kij

"""---------------------------------------------- Find Stresses -----------------------------------------------------"""


def find_s(e,s_prev,go,dolinel = False):
    """ ------------------------------- Calculate element stresses at all gauss points ------------------------------
        --------------------------------------------    INPUT: ------------------------------------------------------
        - e        --> Strains in form [k][l][i][j]
        --------------------------------------------- OUTPUT:--------------------------------------------------------
        - s        --> Matrix of element stresses [s_xi s_yi t_xyi] in form [k][l][i][j]
        - sx       --> Element normal stresses in x-Direction
        - sy       --> Element normal stresses in y-Direction
        - txy      --> Element shear stresses
    -----------------------------------------------------------------------------------------------------------------"""
    # 0 Initiate Time Measurement
    start = time.time()

    nel = len(ELEMENTS[:,0])
    nlk = max(GEOMK["nlk"])
    s = np.ndarray((nel, nlk, go, go,3),dtype=object)
    if isinstance(s_prev,int):
        s_prev = np.ndarray((nel, nlk, go, go,3),dtype=object)
    for k in range(nel):
        for l in range(nlk):
            MAT = [MATK["Ec"][k], MATK["vc"][k], MATK["fcp"][k], MATK["fct"][k], MATK["ec0"][k],
                   0 * MATK["fct"][k] / MATK["Ec"][k], MATK["Dmax"][k], MATK["Esx"][k], MATK["Eshx"][k],
                   MATK["fsyx"][k], MATK["fsux"][k], MATK["Esy"][k], MATK["Eshy"][k], MATK["fsyy"][k],
                   MATK["fsuy"][k], MATK["tb0"][k], MATK["tb1"][k], MATK["Epx"][k], MATK["Epy"][k],
                   MATK["tbp0"][k], MATK["tbp1"][k], MATK["ebp1"][k], MATK["fpux"][k], MATK["fpuy"][k]]
            GEOM = [GEOMK["rhox"][k][l], GEOMK["rhoy"][k][l], GEOMK["dx"][k][l], GEOMK["dy"][k][l],
                    GEOMK["sx"][k][l], GEOMK["sy"][k][l], GEOMK["rhopx"][k][l], GEOMK["rhopy"][k][l],
                    GEOMK["dpx"][k][l], GEOMK["dpy"][k][l]]

            if dolinel:
                cm_k = 1
            else:
                cm_k = MATK["cm"][k]

            for i in range(go):
                for j in range(go):
                    if ELS[4][k] == 3 and i == 1 and j == 1:
                        pass
                    else:
                        e_klij = e[k][l][i][j][:]

                        if it_type == 1:
                            sig = stress(s_prev[k][l][i][j][0],cm_k, MAT, GEOM, k, l, i, j)
                            sig.out(e_klij+[0.0000000000000001j,0,0,0,0])
                            s[k][l][i][j][0] = sig

                            sig = stress(s_prev[k][l][i][j][1],cm_k, MAT, GEOM, k, l, i, j)
                            sig.out(e_klij+[0,0.0000000000000001j,0,0,0])
                            s[k][l][i][j][1] = sig

                            sig = stress(s_prev[k][l][i][j][2],cm_k, MAT, GEOM, k, l, i, j)
                            sig.out(e_klij+[0,0,0.0000000000000001j,0,0])
                            s[k][l][i][j][2] = sig
                        elif it_type == 2:
                            sig = stress(s_prev[k][l][i][j][0],cm_k, MAT, GEOM, k, l, i, j)
                            sig.out(e_klij)
                            s[k][l][i][j][0] = sig

    end = time.time()
    time_stress._updatetime(delta_t=end - start)

    return s


def find_s0():
    """ ---------------- Calculate residual stress state caused by internal strains (shrinkage) ---------------------
        --------------------------------------------    INPUT: ------------------------------------------------------
        - ecs      --> shrinkage strains per element
        --------------------------------------------- OUTPUT:--------------------------------------------------------
        - s0       --> Matrix of residual element stresses [s_xi s_yi t_xyi] in form [k][l][i][j]
    -----------------------------------------------------------------------------------------------------------------"""
    nel = len(ELEMENTS[:,0])
    nlk = max(GEOMK["nlk"])
    s0 = np.zeros((nel, nlk, gauss_order, gauss_order, 3))
    for k in range(nel):
        for l in range(nlk):
            for i in range(gauss_order):
                for j in range(gauss_order):
                    s0[k][l][i][j][:] = find_s0_klij(k,l,i,j)
    return s0


def find_s0_klij(k,l,i,j):
    s0_klij = -np.array([1, 1, 0]) * MATK["Ec"][k] * MATK["ecs"][k] / (1 + MATK["vc"][k] ** 2)
    return s0_klij


def find_ss(s,cmk):
    """ ---------------------------------- Calculate element steel stresses------------------------------------------
        --------------------------------------------    INPUT: ------------------------------------------------------
        - strains
        - Material properties
        --------------------------------------------- OUTPUT:--------------------------------------------------------
        - ssx      --> Element steel stresses in x-Direction
        - ssy      --> Element steel stresses in y-Direction
    -----------------------------------------------------------------------------------------------------------------"""
    nel = len(ELEMENTS[:, 0])
    nlk = max(GEOMK["nlk"])
    ssx = np.zeros((nel, nlk, gauss_order, gauss_order))
    ssy = np.zeros((nel, nlk, gauss_order, gauss_order))
    spx = np.zeros((nel, nlk, gauss_order, gauss_order))
    spy = np.zeros((nel, nlk, gauss_order, gauss_order))
    e1  = np.zeros((nel, nlk, gauss_order, gauss_order))
    e3  = np.zeros((nel, nlk, gauss_order, gauss_order))
    th  = np.zeros((nel, nlk, gauss_order, gauss_order))
    for k in range(nel):
        for l in range(nlk):
            for i in range(gauss_order):
                for j in range(gauss_order):
                    if ELS[4][k] == 3 and i == 1 and j == 1:
                        ssx[k][l][i][j] = -10**-5*np.ones_like(ssx[k][l][i][j])
                        ssy[k][l][i][j] = -10**-5*np.ones_like(ssy[k][l][i][j])
                        spx[k][l][i][j] = -10**-5*np.ones_like(spx[k][l][i][j])
                        spy[k][l][i][j] = -10**-5*np.ones_like(spy[k][l][i][j])
                        e1[k][l][i][j]  = -10**-5*np.ones_like(ssy[k][l][i][j])
                        e3[k][l][i][j]  = -10**-5*np.ones_like(spx[k][l][i][j])
                        th[k][l][i][j]  = -10**-5*np.ones_like(spy[k][l][i][j])
                    elif cmk[k] < 1.5:
                        ssx[k][l][i][j] = -10**-5*np.ones_like(ssx[k][l][i][j])
                        ssy[k][l][i][j] = -10**-5*np.ones_like(ssy[k][l][i][j])
                        spx[k][l][i][j] = -10**-5*np.ones_like(spx[k][l][i][j])
                        spy[k][l][i][j] = -10**-5*np.ones_like(spy[k][l][i][j])
                        e1[k][l][i][j]  = s[k][l][i][j][0].e1.real
                        e3[k][l][i][j]  = s[k][l][i][j][0].e3.real
                        th[k][l][i][j]  = s[k][l][i][j][0].thr.real
                    else:
                        ssx[k][l][i][j] = s[k][l][i][j][0].ssx.real
                        ssy[k][l][i][j] = s[k][l][i][j][0].ssy.real
                        spx[k][l][i][j] = s[k][l][i][j][0].spx.real
                        spy[k][l][i][j] = s[k][l][i][j][0].spy.real
                        e1[k][l][i][j]  = s[k][l][i][j][0].e1.real
                        e3[k][l][i][j]  = s[k][l][i][j][0].e3.real
                        th[k][l][i][j]  = s[k][l][i][j][0].thr.real
    return ssx,ssy,spx,spy,e1,e3,th


# def find_sc(e1, e3):
#     """ ---------------------- Calculate element concrete principla compressive stresses-----------------------------
#         --------------------------------------------    INPUT: ------------------------------------------------------
#         - strains
#         - Material properties
#         --------------------------------------------- OUTPUT:--------------------------------------------------------
#         - s_c3      --> Element concrete principal compressive stresses
#     -----------------------------------------------------------------------------------------------------------------"""
#     nel = len(ELEMENTS[:, 0])
#     nlk = max(GEOMK["nlk"])
#     sc3 = np.zeros((nel, nlk, gauss_order, gauss_order))
#     for k in range(nel):
#         for l in range(nlk):
#             for i in range(gauss_order):
#                 for j in range(gauss_order):
#                     fc_p = MATK["fcp"][k]
#                     e_c0 = MATK["ec0"][k]
#                     if ELS[4][k] == 3 and i == 1 and j == 1:
#                         sc3[k][l][i][j] = -10**-5*np.ones_like(sc3[k][l][i][j])
#                     else:
#                         sc3[k][l][i][j] = s_c3(e3[k][l][i][j], e1[k][l][i][j], fc_p, e_c0)
#     return sc3


"""--------------------------------------------- Solve & Control ----------------------------------------------------"""
def solve_sys(B,fe, cDOF,cVAL, cmk,e,s):
    """ ------------------------------------------- Solve System ----------------------------------------------------
        --------------------------------------------    INPUT: ------------------------------------------------------
        - cmk: Constitutive model to be applied
            - 1 --> Linear Elasticity
            - 3 --> CMM
            given for each element
        --------------------------------------------- OUTPUT:--------------------------------------------------------
        - u     --> Deformed Shape for current iteration step
        - K     --> Stiffness Matrix of current iteration step
    -----------------------------------------------------------------------------------------------------------------"""
    K = k_glob(B,e,s, cmk)

    ### Calculation of spurious zero-energy modes: len(re01) must be equal to 6! ---------------------------------------
    # [e1,e2] = np.linalg.eig(K)
    # re1 = np.zeros_like(e1,dtype = float)
    # for i in range(len(e1)):
    #     re1[i] = e1[i].real
    # re10 = re1[re1<10]
    ### ----------------------------------------------------------------------------------------------------------------
    Kcond = m_stat_con(K, cDOF)
    fecond = v_stat_con(fe, cDOF, cVAL)
    start = time.time()
    # Kinv = np.linalg.inv(Kcond)  # Inverse of Condensed Stiffness Matrix
    # u = np.matmul(Kinv, fecond)
    lu, piv = lu_factor(Kcond)
    u = lu_solve((lu, piv), fecond)
    end = time.time()
    time_Kinv._updatetime(delta_t=end - start)

    # for i in range(len(cDOF)):
    #     if cDOF[i] < len(u):
    #         u = np.insert(u, int(cDOF[i]), 0)  # Add zero deformation from condensed nodes
    #     else:
    #         u = np.append(u, 0)
    # u = [u]
    # u = np.transpose(u)

    return u


def solve_0(B,fe,e, cDOF,cVAL):
    """ --------------------------- Solve Initial Iteration for linear elasticity -----------------------------------"""
    K = k_glob(B,e,e, np.ones_like(fe))
    Kcond = m_stat_con(K, cDOF)
    fecond = v_stat_con(fe, cDOF, cVAL)
    # Kinv = np.linalg.inv(Kcond)  # Inverse of Condensed Stiffness Matrix
    # u = np.matmul(Kinv, fecond)
    lu, piv = lu_factor(Kcond)
    u = lu_solve((lu, piv), fecond)
    # for i in range(len(cDOF)):
    #     if cDOF[i] < len(u):
    #         u = np.insert(u, int(cDOF[i]), 0)  # Add zero deformation from condensed nodes
    #     else:
    #         u = np.append(u, 0)
    # u = [u]
    # u = np.transpose(u)
    return u
"""------------------------------------------------------------------------------------------------------------------"""













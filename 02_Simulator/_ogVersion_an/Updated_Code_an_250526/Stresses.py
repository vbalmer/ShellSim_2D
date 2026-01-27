import numpy as np
from math import *
import cmath
from Mesh_gmsh import MATK,GEOMK
lbd = 0.67

"""------------------------------------------------------------------------------------------------------------------"""

def sqrt(x):
    return cplx(cmath.sqrt(x))


def sin(x):
    return cplx(cmath.sin(x))


def cos(x):
    return cplx(cmath.cos(x))


def tan(x):
    return cplx(cmath.tan(x))


def asin(x):
    return cplx(cmath.asin(x))


def acos(x):
    return cplx(cmath.acos(x))


def atan(x):
    return cplx(cmath.atan(x))

# ---

class cplx(complex):
    """ ----------------------------------- Custom class for complex numbers ----------------------------------------
        - syntax: cplx(real,imag)
        - supported mathematical operations: "+", "-", "*", "/", "**"
        - supported comparisons: ">", ">=", "==", "<=", "<", "!="
        -------------------------------------------------------------------------------------------------------------"""
    def __repr__(self):
        return 'cplx(%r, %r)' % (self.real, self.imag)

    def __add__(self,x):
        return cplx(complex.__add__(self, x))

    def __radd__(self,x):
        return cplx(complex.__radd__(self, x))

    def __sub__(self,x):
        return cplx(complex.__sub__(self, x))

    def __rsub__(self,x):
        return cplx(complex.__rsub__(self, x))

    def __mul__(self,x):
        return cplx(complex.__mul__(self, x))

    def __rmul__(self,x):
        return cplx(complex.__rmul__(self, x))

    def __truediv__(self,x):
        return cplx(complex.__truediv__(self,x))

    def __rtruediv__(self,x):
        return cplx(complex.__rtruediv__(self,x))

    def __pow__(self,x):
        return cplx(complex.__pow__(self, x))

    def __rpow__(self,x):
        return cplx(complex.__rpow__(self, x))

    def __lt__(self,x):
        return self.real < x.real

    def __le__(self,x):
        return self.real <= x.real

    def __gt__(self,x):
        return self.real > x.real

    def __ge__(self,x):
        return self.real >= x.real

    def __eq__(self,x):
        return self.real == x.real

    def __ne__(self,x):
        return self.real != x.real

"""------------------------------------------------------------------------------------------------------------------"""
class stress():

    def __init__(self,cm_klij,e,k,l,i,j):
        """ --------------------------------- Initiate instance of stress class-----------------------------------------
            --------------------------------------------    INPUT: -----------------------------------------------------
            - cm_klij:  Constitutive model used in integration point
            - e:        [ex_klij,ey_klij,gxy_klij,gxz_klij,gyz_klij] strain state in integration point
            - k:        Element number
            - l:        Layer number
            - i:        Index of eta
            - j:        Index of xi
            --------------------------------------------- OUTPUT:-------------------------------------------------------
            - Material, geometry and strain information of instance
        -------------------------------------------------------------------------------------------------------------"""
        self.cm_klij = cm_klij
        self.ex = e[0]
        self.ey = e[1]
        self.gxy = e[2]
        self.gxz = e[3]
        self.gyz = e[4]

        self.Ec = MATK["Ec"][k]
        self.v = MATK["vc"][k]
        self.fcp = MATK["fcp"][k]
        self.fct = MATK["fct"][k]
        self.ec0 = MATK["ec0"][k]
        self.tb1 = MATK["tb1"][k]
        self.tb2 = MATK["tb2"][k]
        self.ect = 0 * MATK["fct"][k] / self.Ec

        self.Esx = MATK["Esx"][k]
        self.Eshx = MATK["Eshx"][k]
        self.fsyx = MATK["fsyx"][k]
        self.fsux = MATK["fsux"][k]
        self.Esy = MATK["Esy"][k]
        self.Eshy = MATK["Eshy"][k]
        self.fsyy = MATK["fsyy"][k]
        self.fsuy = MATK["fsuy"][k]

        self.rhox = GEOMK["rhox"][k][l]
        self.rhoy = GEOMK["rhoy"][k][l]
        self.dx = GEOMK["dx"][k][l]
        self.dy = GEOMK["dy"][k][l]

        self.k = k
        self.l = l
        self.i = i
        self.j = j

    def principal(self):
        """ ------------------------------ Calculation of Principal Strains and Direction ------------------------------
            ----------------------------------------    INPUT (self.): -------------------------------------------------
            - ex, ey, gxy:  In plane strains
            ------------------------------------------- OUTPUT (self.):-------------------------------------------------
            - e1, e3:       Principal strains
            - th:           Principal direction
        -------------------------------------------------------------------------------------------------------------"""
        ex = self.ex
        ey = self.ey
        gxy = self.gxy
        r = 1/2*sqrt((ex - ey) ** 2 + gxy ** 2)
        m = (ex + ey) *1/2

        self.e1 = m + r
        self.e3 = m - r
        if gxy == 0:
            if ex > ey:
                self.th = pi/2
            elif ex < ey:
                self.th = 0
            elif ex == ey:
                self.th = pi/4
        else:
            self.th = atan(gxy / (2 * (self.e1 - ex)))

    def t_mat(self):
        """ ----------------------------------- Calculation of Transformation Matrices ---------------------------------
            ----------------------------------------    INPUT (self.): -------------------------------------------------
            - th:           Principal direction
            ------------------------------------------- OUTPUT (self.):-------------------------------------------------
            - Tsigma:       Stress transformation matrix. [sc1,sc3,tc13]' = Tsigma@[scx,scy,tcxy]'
            - Tepsilon:     Strain transformation matrix. [e1,e3,g13]' = Tepsilon@[ex,ey,txy]'
        -------------------------------------------------------------------------------------------------------------"""
        th = self.th
        self.Tsigma = np.array([[sin(th) ** 2, cos(th) ** 2, 2 * sin(th) * cos(th)],
                                [cos(th) ** 2, sin(th) ** 2, -2 * sin(th) * cos(th)],
                                [- sin(th) * cos(th), sin(th) * cos(th), sin(th) ** 2 - cos(th) ** 2]])
        self.Tepsilon = np.array([[sin(th) ** 2, cos(th) ** 2, sin(th) * cos(th)],
                                  [cos(th) ** 2, sin(th) ** 2, -sin(th) * cos(th)],
                                  [-2 * sin(th) * cos(th), 2 * sin(th) * cos(th), sin(th) ** 2 - cos(th) ** 2]])

    def fcs(self):
        """ ---------------------- Calculation concrete strength with softening as a function of e1---------------------
            ----------------------------------------    INPUT (self.): -------------------------------------------------
            - fcp:          Cylinder compressive strength of concrete
            - e1:           Principal tensile strength
            ----------------------------------------------- OUTPUT:-----------------------------------------------------
            - fc:           Concrete strength considering softening according to Kaufmann (1998)
        -------------------------------------------------------------------------------------------------------------"""
        fc = min((pow(self.fcp, 2 / 3) / (0.4 + 30 * max(self.e1,0))), self.fcp)
        return fc

    def sc_kfm(self, e):
        """ --------------------------------- Calculation concrete compressive stress-----------------------------------
            ----------------------------------------    INPUT (self.): -------------------------------------------------
            - fcs:          Concrete strength considering softening according to Kaufmann (1998)
            - ec0:          Concrete strain at fc
            - ect:          Assumed tensile strain at cracking
            - Ec:           Concrete E-Modulus
            - e:            Normal strain in regarded direction
            ----------------------------------------------- OUTPUT:-----------------------------------------------------
            - sc3:          Concrete compressive stress according to Kaufmann (1998)
                            - e < 0:         sc3 = f(e) according to compression parabola
                            - 0 < e < ect:   sc3 = Ec*e linear elastic in tension
                            - e > ect:       sc3 = "0" (100*e for numerical stability)
        -------------------------------------------------------------------------------------------------------------"""
        fc = self.fcs()
        if e < 0:
            if abs(e) < self.ec0:
                sc3 = fc * (e ** 2 + 2 * e * self.ec0) / (self.ec0 ** 2)
            else:
                sc3 = -self.fcp
        elif e < self.ect:
            sc3 = self.Ec*e
        else:
            sc3 = 100*e
        return sc3

    def ss_bilin(self, es, f_sy, E_s, E_sh):
        """ ------------------------ Calculation of bare reinforcing steel stress: bilinear-----------------------------
            ----------------------------------------    INPUT (self.): -------------------------------------------------
            - Es, Esh:      Reinforcing steel Young's Modulus and Hardening Modulus
            - fsy:          Reinforcing steel yield stress
            ----------------------------------------------- OUTPUT:-----------------------------------------------------
            - ss:           Steel stress
        -------------------------------------------------------------------------------------------------------------"""
        e_sy = f_sy / E_s
        if abs(es) <= e_sy:
            ss = es * E_s
        elif es > 0:
            ss = f_sy + E_sh * (es - e_sy)
        elif es < 0:
            ss = -f_sy + E_sh * (es + e_sy)
        return ss

    def ssr_tcm(self, e, srm, rho, d, fsy, fsu, Es, Esh, tb0, tb1, Ec):
        """ ------------------------------------ Calculate Steel Stress with the TCM -----------------------------------
                --------------------------------------------    INPUT: -------------------------------------------------
                - e: normal strain
                - srm: Crack spacing: srm = lambda * sr0
                - rho: Reinforcement content
                - d: Reinforcement diameter
                - fsy,fsu,Es,Esh: Reinforcing steel parameters
                - tb0,tb1: Bond stresses elastic and plastic
                - Ec: Concrete E-modulus
                --------------------------------------------- OUTPUT:---------------------------------------------------
                - ssr: steel stress at the crack
            ---------------------------------------------------------------------------------------------------------"""
        # 1 Seelhofer

        # 1.1 Material and Geometric properties
        n = Es / Ec
        alpha = 1 + n * rho

        # 1.2 Elastic crack element
        c1 = sqrt(n * n * rho * rho + Es * e / tb0 * d / srm) - n * rho
        x1 = srm / 2 * c1
        x1 = min(max(x1, 0), srm / 2)
        ssr = x1 * 4 * tb0 / d * (1 + n * rho)

        # 1.3 Elastic - Plastic crack element
        if ssr > fsy:
            c2 = sqrt(4 * alpha * Es / Esh * (
                    srm * tb1 / (d * fsy) * (alpha * Es * e / fsy - n * rho) - tb1 / (4 * alpha * tb0)) + 1) - 1
            x2 = d * fsy * Esh / (4 * tb1 * alpha * Es) * c2
            x2 = min(max(x2, 0), srm / 2)
            ssr = fsy + x2 * 4 * tb1 / d
            x21 = (fsy - ssr * n * rho / (1 + n * rho)) * d / (4 * tb0)
            x1 = x2 + x21

        # 2. TCM
        if x1 >= srm / 2:

            # 2.1 Bare steel stress
            st_naked = self.ss_bilin(e, fsy, Es, Esh)

            # 2.2 Steel stress for fully elastic crack element
            s1 = st_naked + tb0 * srm / d

            # 2.3 Steel stress for fully plastic element
            s3 = fsy + Esh * (e - fsy / Es) + tb1 * srm / d

            # 2.4 Assign according to stress level
            # 2.4.1 Fully elastic
            if s1 <= fsy:
                ssr = s1

            # 2.4.2 Fully plastic
            elif s1 > fsy and s3 - (2 * tb1 * srm / d) >= fsy:
                ssr = s3

            # 2.4.3 Partially elastic
            else:
                s2 = (fsy - Es * e) * tb1 * srm / d * (tb0 / tb1 - Es / Esh)
                s2 = s2 + Es / Esh * tb0 * tb1 * srm ** 2 / d ** 2
                s2 = tb0 * srm / d - sqrt(s2)
                s2 = fsy + 2 * s2 / (tb0 / tb1 - Es / Esh)
                ssr = s2
        # 2.5 If stress > ultimate stress, assign ultimate stress
        if ssr > fsu:
            ssr = fsu
        return ssr

    def sr0_vc(self):
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
        th = abs(self.th)
        # 1 Initial Assumption (Formula 5.7 in Kaufmann, 1998)
        srx0 = (self.fct * self.dx * (1 - self.rhox)) / (2 * self.tb1 * self.rhox)
        sry0 = (self.fct * self.dy * (1 - self.rhoy)) / (2 * self.tb1 * self.rhoy)
        self.sr0 = 1 / ((sin(th) / srx0) + (cos(th) / sry0))

    def sigma_cart_1(self):
        E = self.Ec
        v = self.v
        self.sx_x = E/(1-v**2)*self.ex
        self.sx_y = E / (1 - v ** 2) * v * self.ey
        self.sx_xy = 0
        self.sy_x = E/(1-v**2)*v*self.ex
        self.sy_y = E / (1 - v ** 2) * self.ey
        self.sy_xy = 0
        self.txy_x = 0
        self.txy_y = 0
        self.txy_xy = E/(1-v**2)*(1-v)/2*self.gxy

        self.sx = self.sx_x + self.sx_y + self.sx_xy
        self.sy = self.sy_x + self.sy_y + self.sy_xy
        self.txy = self.txy_x + self.txy_y + self.txy_xy

    # def sigma_cart_31(self):
    #     """ -------------------------- Generate Constitutive Matrix for Tension - Tension ------------------------------
    #         --------------------------------------------    INPUT: -----------------------------------------------------
    #         - Strain state
    #         - Element and Layer number
    #         --------------------------------------------- OUTPUT:-------------------------------------------------------
    #         -ET: Consitutive Matrix
    #     -------------------------------------------------------------------------------------------------------------"""
    #     # 1 Layer info
    #     k = self.k
    #     l = self.l
    #
    #     # 2 Concrete Constitutive Matrix
    #
    #     # 1.3 Assumed strain at cracking
    #     ect = 0 * MATK["fct"][k] / self.Ec
    #
    #
    #     # 2.1 If ex < cracking strain: Assign linear elastic Law in x/y
    #     #     Else: Set "zero" stiffness in tensile direction: 100 for numerical stability
    #     if self.ex <= ect:
    #         Ex = self.Ec
    #     else:
    #         Ex = 100
    #     if self.ey <= ect:
    #         Ey = self.Ec
    #     else:
    #         Ey = 100
    #
    #     # 2.2 Constitutive Matrix is formulated directly in x-y space
    #     Ec = np.array([[Ex, 0, 0], [0, Ey, 0], [0, 0, (Ex + Ey) / 4]])
    #
    #     # 3 Steel Stiffness Matrix
    #     # 3.1 Steel Stresses in reinforcement layers
    #     if self.rhox > 0:
    #         self.sr0 = (self.fct * self.dx * (1 - self.rhox)) / (2 * self.tb1 * self.rhox)
    #         self.sr = self.sr0 * lbd
    #         self.ssx = self.ssr_tcm(self.ex, self.sr, self.rhox, self.dx, self.fsyx, self.fsux, self.Esx,
    #                                 self.Eshx, self.tb1, self.tb2, self.Ec)
    #     else:
    #         self.ssx = 0
    #     if self.rhoy > 0:
    #         self.sr0 = (self.fct * self.dy * (1 - self.rhoy)) / (2 * self.tb1 * self.rhoy)
    #         self.sr = self.sr0 * lbd
    #         self.ssy = self.ssr_tcm(self.ey, self.sr, self.rhoy, self.dy, self.fsyy, self.fsuy, self.Esy,
    #                                 self.Eshy, self.tb1, self.tb2, self.Ec)
    #     else:
    #         self.ssy = 0
    #
    #     # 3.2 Steel Secant Stiffness Matrix
    #     if abs(self.ex) > 0:
    #         Esx = self.ssx / self.ex
    #     else:
    #         Esx = 0
    #     if abs(self.ey) > 0:
    #         Esy = self.ssy / self.ey
    #     else:
    #         Esy = 0
    #
    #     # 3.3 Constitutive Matrix in x-y space
    #     Ds = np.array([[self.rhox * Esx, 0, 0], [0, self.rhoy * Esy, 0], [0, 0, 0]])
    #
    #     # 4  Constitutive Matrix of Steel and Concrete
    #     Esec = Ec + Ds
    #
    #     # 5 Stresses
    #     self.sx_x = Esec[0,0]*self.ex
    #     self.sx_y = Esec[0,1]*self.ey
    #     self.sx_xy = Esec[0,2]*self.gxy
    #     self.sy_x = Esec[1,0]*self.ex
    #     self.sy_y = Esec[1,1]*self.ey
    #     self.sy_xy = Esec[1,2]*self.gxy
    #     self.txy_x = Esec[2,0]*self.ex
    #     self.txy_y = Esec[2,1]*self.ey
    #     self.txy_xy = Esec[2,2]*self.gxy
    #
    #     self.sx = self.sx_x + self.sx_y + self.sx_xy
    #     self.sy = self.sy_x + self.sy_y + self.sy_xy
    #     self.txy = self.txy_x + self.txy_y + self.txy_xy
    #
    # def sigma_cart_32(self):
    #     """ ----------------------- Generate Constitutive Model for Compression - Compression --------------------------
    #             --------------------------------------------    INPUT: -------------------------------------------------
    #             - Strain state
    #             - Element and layer number
    #             --------------------------------------------- OUTPUT:---------------------------------------------------
    #             -ET: Constitutive Matrix
    #         ---------------------------------------------------------------------------------------------------------"""
    #     # 1 Layer info
    #     k = self.k
    #     l = self.l
    #
    #     # 2 Concrete Constitutive Matrix
    #
    #     # 2.1 Principal Concrete stresses
    #     self.sc1 = self.sc_kfm(self.e1)
    #     self.sc3 = self.sc_kfm(self.e3)
    #
    #     # 2.2 Concrete secant stiffness matrix
    #     #     If concrete in 1- and 3- directions: assign secant stiffness
    #     #     Else: assign "zero" (not possible in compression-compression case)
    #     if self.e1 < 0:
    #         E1 = abs(self.sc1 / self.e1)
    #     else:
    #         E1 = 100
    #     if self.e3 < 0:
    #         E3 = abs(self.sc3 / self.e3)
    #     else:
    #         E3 = 100
    #     if abs(self.e3 - self.e1) > 0:
    #         G = max(abs(0.5 * (self.sc3 - self.sc1) / (self.e3 - self.e1)), 50)
    #     else:
    #         G = 50
    #
    #     # 2.3 Assign concrete secant constitutive matrix in principal directions
    #     Ec13 = np.array([[E1, 0, 0], [0, E3, 0], [0, 0, G]])
    #
    #     # 3 Transform from principal to local coordinate System
    #     #   Remember: inv(Tsigma) = transpose(Tepsilon)
    #     self.t_mat()
    #     Ec = np.linalg.inv(self.Tsigma) @ Ec13 @ self.Tepsilon
    #
    #     # 4 Steel Constitutive Matrix
    #     # 4.1 Steel Stresses in reinforcement layers
    #     if self.rhox > 0:
    #         self.ssx = self.ss_bilin(self.ex, self.fsyx, self.Esx, self.Eshx)
    #     else:
    #         self.ssx = 0
    #     if self.rhoy > 0:
    #         self.ssy = self.ss_bilin(self.ey, self.fsyy, self.Esy, self.Eshy)
    #     else:
    #         self.ssy = 0
    #
    #     # 4.2 Steel Secant Stiffness Moduli
    #     if abs(self.ex) > 0:
    #         Esx = self.ssx / self.ex
    #     else:
    #         Esx = 0
    #     if abs(self.ey) > 0:
    #         Esy = self.ssy / self.ey
    #     else:
    #         Esy = 0
    #
    #     # 4.3 Constitutive Matrix in x-y space
    #     Ds = np.array([[self.rhox * Esx, 0, 0], [0, self.rhoy * Esy, 0], [0, 0, 0]])
    #
    #     # 5 Constitutive Matrix of Steel and Concrete
    #     Esec = Ec + Ds
    #
    #     # 6 Stresses
    #     self.sx_x = Esec[0,0]*self.ex
    #     self.sx_y = Esec[0,1]*self.ey
    #     self.sx_xy = Esec[0,2]*self.gxy
    #     self.sy_x = Esec[1,0]*self.ex
    #     self.sy_y = Esec[1,1]*self.ey
    #     self.sy_xy = Esec[1,2]*self.gxy
    #     self.txy_x = Esec[2,0]*self.ex
    #     self.txy_y = Esec[2,1]*self.ey
    #     self.txy_xy = Esec[2,2]*self.gxy
    #
    #     self.sx = self.sx_x + self.sx_y + self.sx_xy
    #     self.sy = self.sy_x + self.sy_y + self.sy_xy
    #     self.txy = self.txy_x + self.txy_y + self.txy_xy

    # def sigma_cart_33(self):
    #     """ --------------------------- Generate Elastic Tensor for CMM plane stress------------------------------------
    #             --------------------------------------------    INPUT: -------------------------------------------------
    #             - Strain state
    #             - Element and layer number
    #             --------------------------------------------- OUTPUT:---------------------------------------------------
    #             -ET: Constitutive Matrix
    #         ---------------------------------------------------------------------------------------------------------"""
    #
    #     # 1 Layer info
    #     k = self.k
    #     l = self.l
    #
    #     # 1.3 Assumed strain at cracking
    #     ect = 0 * MATK["fct"][k] / self.Ec
    #
    #     # 2 Concrete Constitutive Matrix
    #     # 2.1 Principal Concrete stresses
    #     self.sc1 = self.sc_kfm(self.e1)
    #     self.sc3 = self.sc_kfm(self.e3)
    #
    #     # 2.3 Concrete secant stiffness matrix
    #     #     If concrete in 1- and 3- directions: assign secant stiffness
    #     #     Else: Ec if uncracked
    #     #           "zero" if e1 > cracking strain
    #     if self.e1 < 0:
    #         E1 = abs(self.sc1 / self.e1)
    #     elif self.e1 <= ect:
    #         E1 = self.Ec
    #     else:
    #         E1 = 100
    #     if self.e3 < 0:
    #         E3 = abs(self.sc3 / self.e3)
    #     elif self.e3 <= ect:
    #         E3 = self.Ec
    #     else:
    #         E3 = 100
    #     if abs(self.e3 - self.e1) > 0:
    #         G = max(abs(0.5 * (self.sc3 - self.sc1) / (self.e3 - self.e1)), 50)
    #     else:
    #         G = 50
    #
    #     # 2.4 Assign concrete secant constitutive matrix in principal directions
    #     Ec13 = np.array([[E1, 0, 0], [0, E3, 0], [0, 0, G]])
    #
    #     # 3 Transform from principal to local coordinate System
    #     #   Remember: inv(Tsigma) = transpose(Tepsilon)
    #     self.t_mat()
    #     Ec = np.linalg.inv(self.Tsigma) @ Ec13 @ self.Tepsilon
    #
    #     # 4 Steel Constitutive Matrix
    #     # 4.1 Steel Stresses in reinforcement layers
    #     if self.rhox > 0:
    #         # self.ssx = s_sr2(self.ex, self.e1, self.e3, self.th, k, l, 0)
    #         if self.ex > 0:
    #             self.sr0_vc()
    #             self.sr = self.sr0 * lbd
    #             self.ssx = self.ssr_tcm(self.ex, self.sr / sin(abs(self.th)), self.rhox, self.dx, self.fsyx, self.fsux, self.Esx,
    #                                     self.Eshx, self.tb1, self.tb2, self.Ec)
    #         else:
    #             self.ssx = self.ss_bilin(self.ex, self.fsyx, self.Esx, self.Eshx)
    #     else:
    #         self.ssx = 0
    #     if self.rhoy > 0:
    #         if self.ey > 0:
    #             self.sr0_vc()
    #             self.sr = self.sr0 * lbd
    #             self.ssy = self.ssr_tcm(self.ey, self.sr / cos(abs(self.th)), self.rhoy, self.dy, self.fsyy, self.fsuy, self.Esy,
    #                                     self.Eshy, self.tb1, self.tb2, self.Ec)
    #         else:
    #             self.ssy = self.ss_bilin(self.ey, self.fsyy, self.Esy, self.Eshy)
    #     else:
    #         self.ssy = 0
    #
    #     # 4.2 Steel Secant Stiffness Moduli
    #     if abs(self.ex) > 0:
    #         Esx = self.ssx / self.ex
    #     else:
    #         Esx = 0
    #     if abs(self.ey) > 0:
    #         Esy = self.ssy / self.ey
    #     else:
    #         Esy = 0
    #
    #     # 4.3 Constitutive Matrix in x-y space
    #     Ds = np.array([[self.rhox * Esx, 0, 0], [0, self.rhoy * Esy, 0], [0, 0, 0]])
    #
    #     # 5 Constitutive Matrix of Steel and Concrete
    #     Esec = Ec + Ds
    #
    #     # 6 Stresses
    #     self.sx_x = Esec[0,0]*self.ex
    #     self.sx_y = Esec[0,1]*self.ey
    #     self.sx_xy = Esec[0,2]*self.gxy
    #     self.sy_x = Esec[1,0]*self.ex
    #     self.sy_y = Esec[1,1]*self.ey
    #     self.sy_xy = Esec[1,2]*self.gxy
    #     self.txy_x = Esec[2,0]*self.ex
    #     self.txy_y = Esec[2,1]*self.ey
    #     self.txy_xy = Esec[2,2]*self.gxy
    #
    #     self.sx = self.sx_x + self.sx_y + self.sx_xy
    #     self.sy = self.sy_x + self.sy_y + self.sy_xy
    #     self.txy = self.txy_x + self.txy_y + self.txy_xy

    def sigma_cart_31(self):
        """ -------------------------- Generate Constitutive Matrix for Tension - Tension ------------------------------
            --------------------------------------------    INPUT: -----------------------------------------------------
            - Strain state
            - Element and Layer number
            --------------------------------------------- OUTPUT:-------------------------------------------------------
            -ET: Consitutive Matrix
        -------------------------------------------------------------------------------------------------------------"""
        # 1 Layer info
        k = self.k
        l = self.l

        # 2 Concrete Constitutive Matrix

        # 2.1 If ex < cracking strain: Assign linear elastic Law in x/y
        #     Else: Set "zero" stiffness in tensile direction: 100 for numerical stability
        if self.ex <= self.ect:
            self.scx = self.Ec*self.ex
        else:
            self.scx = 100*self.ex
        if self.ey <= self.ect:
            self.scy = self.Ec*self.ey
        else:
            self.scy = 100*self.ey

        # 3 Steel Stiffness Matrix
        # 3.1 Steel Stresses in reinforcement layers
        if self.rhox > 0:
            self.sr0 = (self.fct * self.dx * (1 - self.rhox)) / (2 * self.tb1 * self.rhox)
            self.sr = self.sr0 * lbd
            self.ssx = self.ssr_tcm(self.ex, self.sr, self.rhox, self.dx, self.fsyx, self.fsux, self.Esx,
                                    self.Eshx, self.tb1, self.tb2, self.Ec)
        else:
            self.ssx = 0
        if self.rhoy > 0:
            self.sr0 = (self.fct * self.dy * (1 - self.rhoy)) / (2 * self.tb1 * self.rhoy)
            self.sr = self.sr0 * lbd
            self.ssy = self.ssr_tcm(self.ey, self.sr, self.rhoy, self.dy, self.fsyy, self.fsuy, self.Esy,
                                    self.Eshy, self.tb1, self.tb2, self.Ec)
        else:
            self.ssy = 0

        self.v = 0
        self.sx = 1/(1-self.v**2)*(self.scx + self.v*self.scy) + self.rhox*self.ssx
        self.sy = 1/(1-self.v**2)*(self.scy + self.v*self.scx) + self.rhoy*self.ssy
        self.txy = 500*self.gxy

    def sigma_cart_32(self):
        """ ----------------------- Generate Constitutive Model for Compression - Compression --------------------------
                --------------------------------------------    INPUT: -------------------------------------------------
                - Strain state
                - Element and layer number
                --------------------------------------------- OUTPUT:---------------------------------------------------
                -ET: Constitutive Matrix
            ---------------------------------------------------------------------------------------------------------"""
        # 1 Layer info
        k = self.k
        l = self.l

        # 2 Concrete Constitutive Matrix

        # 2.1 Principal Concrete stresses
        self.sc1 = self.sc_kfm(self.e1)
        self.sc3 = self.sc_kfm(self.e3)

        # 2.2 Concrete secant stiffness matrix
        #     If concrete in 1- and 3- directions: assign secant stiffness
        #     Else: assign "zero" (not possible in compression-compression case)
        self.scx = self.sc3*cos(self.th)**2+self.sc1*sin(self.th)**2
        self.scy = self.sc3 * sin(self.th)**2 + self.sc1 * cos(self.th)**2
        self.txy = (self.sc1-self.sc3)*sin(self.th)*cos(self.th)

        # 4 Steel Constitutive Matrix
        # 4.1 Steel Stresses in reinforcement layers
        if self.rhox > 0:
            self.ssx = self.ss_bilin(self.ex, self.fsyx, self.Esx, self.Eshx)
        else:
            self.ssx = 0
        if self.rhoy > 0:
            self.ssy = self.ss_bilin(self.ey, self.fsyy, self.Esy, self.Eshy)
        else:
            self.ssy = 0

        self.sx = 1/(1-self.v**2)*(self.scx + self.v*self.scy) + self.rhox*self.ssx
        self.sy = 1/(1-self.v**2)*(self.scy + self.v*self.scx) + self.rhoy*self.ssy

    def sigma_cart_33(self):
        """ --------------------------- Generate Elastic Tensor for CMM plane stress------------------------------------
                --------------------------------------------    INPUT: -------------------------------------------------
                - Strain state
                - Element and layer number
                --------------------------------------------- OUTPUT:---------------------------------------------------
                -ET: Constitutive Matrix
            ---------------------------------------------------------------------------------------------------------"""

        # 1 Layer info
        k = self.k
        l = self.l

        # 1.3 Assumed strain at cracking

        # 2 Concrete Constitutive Matrix
        # 2.1 Principal Concrete stresses
        self.sc1 = self.sc_kfm(self.e1)
        self.sc3 = self.sc_kfm(self.e3)
        # if self.sc1 >= -10**-5:
        #     self.sc1 = 100*self.e1

        # 2.3 Concrete secant stiffness matrix
        #     If concrete in 1- and 3- directions: assign secant stiffness
        #     Else: Ec if uncracked
        #           "zero" if e1 > cracking strain
        self.scx = self.sc3*cos(self.th)**2+self.sc1*sin(self.th)**2
        self.scy = self.sc3 * sin(self.th) ** 2 + self.sc1 * cos(self.th) ** 2
        self.txy = (self.sc1-self.sc3)*sin(self.th)*cos(self.th)
        # print(self.sc3,self.sc1)

        # 4 Steel Constitutive Matrix
        # 4.1 Steel Stresses in reinforcement layers
        if self.rhox > 0:
            # self.ssx = s_sr2(self.ex, self.e1, self.e3, self.th, k, l, 0)
            if self.ex > 0:
                self.sr0_vc()
                self.sr = self.sr0 * lbd
                self.ssx = self.ssr_tcm(self.ex, self.sr / sin(abs(self.th)), self.rhox, self.dx, self.fsyx, self.fsux, self.Esx,
                                        self.Eshx, self.tb1, self.tb2, self.Ec)
            else:
                self.ssx = self.ss_bilin(self.ex, self.fsyx, self.Esx, self.Eshx)
        else:
            self.ssx = 0
        if self.rhoy > 0:
            if self.ey > 0:
                self.sr0_vc()
                self.sr = self.sr0 * lbd
                self.ssy = self.ssr_tcm(self.ey, self.sr / cos(abs(self.th)), self.rhoy, self.dy, self.fsyy, self.fsuy, self.Esy,
                                        self.Eshy, self.tb1, self.tb2, self.Ec)
            else:
                self.ssy = self.ss_bilin(self.ey, self.fsyy, self.Esy, self.Eshy)
        else:
            self.ssy = 0

        # 4.2 Steel Secant Stiffness Moduli
        self.v = 0
        self.sx = 1/(1-self.v**2)*(self.scx + self.v*self.scy) + self.rhox*self.ssx
        self.sy = 1/(1-self.v**2)*(self.scy + self.v*self.scx) + self.rhoy*self.ssy

    def sigma_shear(self):
        """ ---------------------- Generate Shear Stiffness Matrix in xz and yz Direction -------------------------------
            --------------------------------------------    INPUT: ------------------------------------------------------
            - E: Young's Modulus
            - v: Poisson's Ratio
            --------------------------------------------- OUTPUT:--------------------------------------------------------
            -ET: Elastic Matrix
        -----------------------------------------------------------------------------------------------------------------"""
        G = (self.Ec + self.Ec) / (4 * (1 + self.v))
        self.txz = 5/6*G*self.gxz
        self.tyz = 5/6*G*self.gyz

    def sigma_cart(self):
        if self.cm_klij == 1:
            self.sigma_cart_1()
        elif self.cm_klij == 3:
            self.principal()
            if self.e1 > 0 and self.e3 > 0:
                self.sigma_cart_31()
            elif self.e1 < 0 and self.e3 < 0:
                self.sigma_cart_32()
            else:
                self.sigma_cart_33()
        self.sigma_shear()
        self.s = [self.sx,self.sy,self.txy,self.txz,self.tyz]



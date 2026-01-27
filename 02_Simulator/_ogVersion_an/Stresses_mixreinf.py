import numpy as np
from math import *
lbd = 0.67
from defcplx import *

class stress():
    def __init__(self,cm_klij,MAT,GEOM):
        """ --------------------------------- Initiate instance of stress class-----------------------------------------
            --------------------------------------------    INPUT: -----------------------------------------------------
            - cm_klij: Constitutive Model to be used in integration point
                        - cm_klij = 1: Linear Elastic
                        - cm_klij = 3: Nonlinear
            - MAT:     Material Information for Integration Point
            - GEOM:    Geometry Information for Integration Point
            --------------------------------------------- OUTPUT:-------------------------------------------------------
            - Material, geometry and strain information of instance
        -------------------------------------------------------------------------------------------------------------"""
        self.cm_klij = cm_klij

        self.Ec = MAT[0]
        self.v = MAT[1]
        self.fcp = MAT[2]
        self.fct = MAT[3]
        self.ec0 = MAT[4]
        self.ect = MAT[5]

        self.Esx = MAT[6]
        self.Eshx = MAT[7]
        self.fsyx = MAT[8]
        self.fsux = MAT[9]
        self.Esy = MAT[10]
        self.Eshy = MAT[11]
        self.fsyy = MAT[12]
        self.fsuy = MAT[13]
        self.tb0 = MAT[14]
        self.tb1 = MAT[15]

        self.Epx = MAT[16]
        self.Epy = MAT[17]
        self.tbp0 = MAT[18]
        self.tbp1 = MAT[19]
        self.ebp1 = MAT[20]
        self.fpux = MAT[21]
        self.fpuy = MAT[22]

        self.rhox = GEOM[0]
        self.rhoy = GEOM[1]
        self.dx = GEOM[2]
        self.dy = GEOM[3]

        self.rhopx = GEOM[4]
        self.rhopy = GEOM[5]
        self.dpx = GEOM[6]
        self.dpy = GEOM[7]

    def principal(self):
        """ ------------------------------ Calculation of Principal Strains and Direction ------------------------------
            ----------------------------------------    INPUT (self.): -------------------------------------------------
            - ex, ey, gxy:  In plane strains
            ------------------------------------------- OUTPUT (self.):-------------------------------------------------
            - e1, e3:       Principal strains
            - th:           Principal direction (in range -pi/2 to pi/2
        -------------------------------------------------------------------------------------------------------------"""
        ex = self.ex
        ey = self.ey
        gxy = self.gxy
        r = 1/2*sqrt((ex - ey) ** 2 + gxy ** 2)
        m = (ex + ey) *1/2

        self.e1 = m + r
        self.e3 = m - r
        if abs(gxy) < 10**-8:
            if ex > ey:
                self.th = pi/2-10**(-10)
            elif ex < ey:
                self.th = 10**(-10)
            elif ex == ey:
                self.th = pi/4
        else:
            self.th = atan(gxy / (2 * (self.e1 - ex)))
        # print(self.th)

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
            - fc:           Concrete strength considering softening according to Vecchio & Collins (1986)
        -------------------------------------------------------------------------------------------------------------"""
        # fc = min((pow(self.fcp, 2 / 3) / (0.4 + 30 * max(self.e1,0))), self.fcp) # Kaufmann
        # fc = min((self.fcp/(0.8+0.34*self.ec0*self.e1)),self.fcp)                  # Vecchio & Collins

        if self.e1 <= 0.2 * self.ec0 / 0.34:                                        # CSFM
            kc = 1 + (0.2 * self.ec0 / 0.34 - self.e1) * 0.001
        elif self.e1 >= 0.22 / 35.75:
            kc = 1 / (1.2 + 55 * self.e1)
        else:
            x = 228906275 / (64 * (5281250 * self.ec0 ** 2 + 357))
            y = -13 * (377609375 * self.ec0 ** 2 + 160973) / (40 * (5281250 * self.ec0 ** 2 + 357))
            z = (8376062500 * self.ec0 ** 2 + 837097) / (2000 * (5281250 * self.ec0 ** 2 + 357))
            kc = x * self.e1 ** 2 + y * self.e1 + z
        fc = kc * self.fcp

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
                            - e > ect:       sc3 = "0" (10*e for numerical stability)
                            - alpha parameter: from e = alpha*ec0, a linear course with the inclination of the derivation
                              at alpha*ec0 is assumed for numerical stability
        -------------------------------------------------------------------------------------------------------------"""
        fc = self.fcs()
        alpha = 0.98
        if e < 0:
            if abs(e) < alpha * self.ec0:
                sc3 = fc * (e ** 2 + 2 * e * self.ec0) / (self.ec0 ** 2)
            else:
                sc3 = fc * (alpha ** 2 - 2 * alpha) + fc * (2 * alpha - 2) / self.ec0 * (- e - alpha * self.ec0)
        elif e < self.ect:
            sc3 = self.Ec * e
        else:
            sc3 = 10*e
        return sc3

    def sc_sargin(self,e):
        # Formula 3.16, p36 in Sargin, 1971
        fc = self.fcs()
        A = self.Ec*self.ec0/fc
        D = 0.3
        if e < 0:
            x = -e/self.ec0
            sc3 = -fc*(A*x+(D-1)*x**2)/(1+(A-2)*x+D*x**2)
            if sc3 > 0:
                sc3 = 0
            if x > 1:
                sc3 = -fc*(1+(x-1)/100)
        elif e < self.ect:
            sc3 = self.Ec * e
        else:
            sc3 = 10 * e
        return sc3

    def sc_linel(self,e):
        if e<0:
            if e<-self.ec0:
                sc3 = -self.fcs()+(e-self.ec0)*10
            else:
                sc3 = self.Ec*e
        elif e < self.ect:
            sc3 = self.Ec * e
        else:
            sc3 = 10 * e
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
                ------------------------------------------ INPUT (self.): ----------------------------------------------
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
        """ --------------------------------- Calculate diagonal crack spacing------------------------------------------
            --------------------------------------------    INPUT: -----------------------------------------------------
            - th: Principal direction
            - fct: concrete tensile strength
            - rhox, rhoy: Reinforcement contents
            - dx,dy: Reinforcement diameters
            - tb0: bond stress elastic
            --------------------------------------------- OUTPUT:-------------------------------------------------------
            - srx0, sry0: Maximum crack spacing of tension chord in x and y direction
            - sr: Diagonal crack spacing
            - srx, sry: Crack spacing in reinforcement direction
        -------------------------------------------------------------------------------------------------------------"""
        th = self.th
        if th < 0:
            th = -th
        # 1 Initial Assumption (Formula 5.7 in Kaufmann, 1998 extended for prestressing)
        if self.rhox + self.rhopx > 0:
            self.srx0 = (2*self.tb0/self.dx*self.rhox+2*self.tbp0/self.dpx*self.rhopx)**(-1)*self.fct*(1-self.rhox-self.rhopx)
        else:
            self.srx0 = 10**6
        if self.rhoy + self.rhopy > 0:
            self.sry0 = (2 * self.tb0 / self.dy * self.rhoy + 2 * self.tbp0 / self.dpy * self.rhopy) ** (-1) * self.fct * (
                        1 - self.rhoy - self.rhopy)
        else:
            self.sry0 = 10**6

        self.sr = lbd / ((sin(th) / self.srx0) + (cos(th) / self.sry0))
        self.srx = self.sr/(sin(th))
        self.sry =  self.sr/(cos(th))

    def sigma_cart_1(self):
        """ ---------------------- Get In Plane Stresses for Linear Elastic Material Law ----------------------------"""
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

    def sigma_cart_31(self):
        """ ------------------------------ Get In Plane Stresses for Tension - Tension ---------------------------------
            --------------------------------------------    INPUT: -----------------------------------------------------
            - Strain state
            - Integration Point Information
            --------------------------------------------- OUTPUT:-------------------------------------------------------
            - self.sx, self.sy, self.txy: Normal- and shear stresses in Integration Point
            - self.ssx, self.ssy        : Steel stresses in x and y direction
            - self.scx, self,scy        : Concrete stresses in x and y direction
        -------------------------------------------------------------------------------------------------------------"""
        # 1 Layer info

        # 2 Concrete Contribution
        # 2.1 If ex < cracking strain: Assign linear elastic Law in x/y
        #     Else: Set "zero" stiffness in tensile direction: 100 for numerical stability
        self.sc1 = 0
        self.sc3 = 0
        if self.ex <= self.ect:
            self.scx = self.Ec*self.ex
        else:
            self.scx = 10*self.ex
        if self.ey <= self.ect:
            self.scy = self.Ec*self.ey
        else:
            self.scy = 10*self.ey

        # 3 Steel Contribution
        if self.rhox > 0:
            self.sr0_vc()
            self.ssx = self.ssr_tcm(self.ex, self.srx0*lbd, self.rhox, self.dx, self.fsyx, self.fsux, self.Esx,
                                    self.Eshx, self.tb0, self.tb1, self.Ec)
        else:
            self.ssx = 0
        if self.rhoy > 0:
            self.sr0_vc()
            self.ssy = self.ssr_tcm(self.ey, self.sry0*lbd, self.rhoy, self.dy, self.fsyy, self.fsuy, self.Esy,
                                    self.Eshy, self.tb0, self.tb1, self.Ec)
        else:
            self.ssy = 0

        # 4 CFRP Contribution
        if self.rhopx > 0:
            self.sr0_vc()
            self.spx = self.ssr_tcm(self.ex, self.srx0*lbd, self.rhopx, self.dpx, self.ebp1*self.Epx, self.fpux, self.Epx,
                                    self.Epx, self.tbp0, self.tbp1, self.Ec)
        else:
            self.spx = 0
        if self.rhopy > 0:
            self.sr0_vc()
            self.spy = self.ssr_tcm(self.ey, self.sry0*lbd, self.rhopy, self.dpy, self.ebp1*self.Epy, self.fpuy, self.Epy,
                                    self.Epy, self.tbp0, self.tbp1, self.Ec)
        else:
            self.spy = 0
        self.v = 0
        self.sx = 1/(1-self.v**2)*(self.scx + self.v*self.scy) + self.rhox*self.ssx + self.rhopx*self.spx
        self.sy = 1/(1-self.v**2)*(self.scy + self.v*self.scx) + self.rhoy*self.ssy + self.rhopy*self.spy
        self.txy = 5*self.gxy

    def sigma_cart_32(self):
        """ -------------------------- Get In Plane Stresses for Compression - Compression -----------------------------
            --------------------------------------------    INPUT: -----------------------------------------------------
            - Strain state
            - Integration Point Information
            --------------------------------------------- OUTPUT:-------------------------------------------------------
            - self.sx, self.sy, self.txy: Normal- and shear stresses in Integration Point
            - self.ssx, self.ssy        : Steel stresses in x and y direction
            - self.scx, self,scy        : Concrete stresses in x and y direction
        -------------------------------------------------------------------------------------------------------------"""
        # 1 Layer info

        # 2 Concrete Constitutive Matrix

        # 2.1 Principal Concrete stresses
        self.sc1 = self.sc_linel(self.e1)
        self.sc3 = self.sc_linel(self.e3)

        # 2.2 Concrete secant stiffness matrix
        #     If concrete in 1- and 3- directions: assign secant stiffness
        #     Else: assign "zero" (not possible in compression-compression case)
        self.scx = self.sc3*cos(self.th)**2+self.sc1*sin(self.th)**2
        self.scy = self.sc3 * sin(self.th)**2 + self.sc1 * cos(self.th)**2
        self.txy = (self.sc1-self.sc3)*sin(self.th)*cos(self.th)

        # 3 Steel Contribution
        if self.rhox > 0:
            self.ssx = self.ss_bilin(self.ex, self.fsyx, self.Esx, self.Eshx)
        else:
            self.ssx = 0
        if self.rhoy > 0:
            self.ssy = self.ss_bilin(self.ey, self.fsyy, self.Esy, self.Eshy)
        else:
            self.ssy = 0

        # 4 CFRP Contribution
        if self.rhopx > 0:
            self.spx = self.ex*self.Epx
        else:
            self.spx = 0
        if self.rhopy > 0:
            self.spy = self.ey*self.Epy
        else:
            self.spy = 0

        self.sx = 1/(1-self.v**2)*(self.scx + self.v*self.scy) + self.rhox*self.ssx + self.rhopx*self.spx
        self.sy = 1/(1-self.v**2)*(self.scy + self.v*self.scx) + self.rhoy*self.ssy + self.rhopy*self.spy

    def sigma_cart_33(self):
        """ ---------------------------- Get In Plane Stresses for Cracked Membrane Model ------------------------------
            --------------------------------------------    INPUT: -----------------------------------------------------
            - Strain state
            - Integration Point Information
            --------------------------------------------- OUTPUT:-------------------------------------------------------
            - self.sx, self.sy, self.txy: Normal- and shear stresses in Integration Point
            - self.ssx, self.ssy        : Steel stresses in x and y direction
            - self.scx, self,scy        : Concrete stresses in x and y direction
        -------------------------------------------------------------------------------------------------------------"""

        # 1 Layer info

        # 1.3 Assumed strain at cracking

        # 2 Concrete Constitutive Matrix
        # 2.1 Principal Concrete stresses
        self.sc1 = self.sc_linel(self.e1)
        self.sc3 = self.sc_linel(self.e3)
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

        # 3 Steel Contribution
        if self.rhox > 0:
            if self.ex > 0:
                self.sr0_vc()
                self.ssx = self.ssr_tcm(self.ex, self.srx, self.rhox, self.dx, self.fsyx, self.fsux, self.Esx,
                                        self.Eshx, self.tb0, self.tb1, self.Ec)
            else:
                self.ssx = self.ss_bilin(self.ex, self.fsyx, self.Esx, self.Eshx)
        else:
            self.ssx = 0
        if self.rhoy > 0:
            if self.ey > 0:
                self.sr0_vc()
                self.ssy = self.ssr_tcm(self.ey, self.sry, self.rhoy, self.dy, self.fsyy, self.fsuy, self.Esy,
                                        self.Eshy, self.tb0, self.tb1, self.Ec)
            else:
                self.ssy = self.ss_bilin(self.ey, self.fsyy, self.Esy, self.Eshy)
        else:
            self.ssy = 0

        # 4 CFRP Contribution
        if self.rhopx > 0:
            if self.ex > 0:
                self.sr0_vc()
                self.spx = self.ssr_tcm(self.ex, self.srx, self.rhopx, self.dpx, self.ebp1*self.Epx,
                                        self.fpux, self.Epx,
                                        self.Epx, self.tbp0, self.tbp1, self.Ec)
            else:
                self.spx = self.ex*self.Epx
        else:
            self.spx = 0
        if self.rhopy > 0:
            if self.ey > 0:
                self.sr0_vc()
                self.spy = self.ssr_tcm(self.ey, self.sry, self.rhopy, self.dpy, self.ebp1*self.Epy,
                                        self.fpuy, self.Epy,
                                        self.Epy, self.tbp0, self.tbp1, self.Ec)
            else:
                self.spy = self.ey*self.Epy
        else:
            self.spy = 0

        self.v = 0
        self.sx = 1/(1-self.v**2)*(self.scx + self.v*self.scy) + self.rhox*self.ssx + self.rhopx*self.spx
        self.sy = 1/(1-self.v**2)*(self.scy + self.v*self.scx) + self.rhoy*self.ssy + self.rhopy*self.spy

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

    def out(self,e):
        """ ------------------------------------------- Define Output---------------------------------------------------
            --------------------------------------------    INPUT: -----------------------------------------------------
            - e:        [ex_klij,ey_klij,gxy_klij,gxz_klij,gyz_klij] strain state in integration point
            --------------------------------------------- OUTPUT:-------------------------------------------------------
            - s:        [sx_klij,sy_klij,txy_klij,txz_klij,tyz_klij] stress state in integration point
        -------------------------------------------------------------------------------------------------------------"""
        # 1 Assign Strain Values
        self.ex = e[0]
        self.ey = e[1]
        self.gxy = e[2]
        self.gxz = e[3]
        self.gyz = e[4]

        # 2 Calculate Stresses based on given constitutive model and strain state
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

        # 3 Output
        self.s = [self.sx,self.sy,self.txy,self.txz,self.tyz]

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

        # Get 3x3 Constitutive (Tangent) Matrix relating in-plane [ex ey gxy] to [sx sy txy] (linear elastic or with Cracked Membrane Model)
        Dp = get_et(cm_k, e_kij[l,0:5],s_kij[l], k, l, i, j)

        # 2x2 Constitutive (Tangent) Matrix relating out of plane shear strains to stresses
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











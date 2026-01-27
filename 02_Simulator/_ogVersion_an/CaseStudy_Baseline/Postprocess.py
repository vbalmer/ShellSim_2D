
from Mesh_gmsh_copy import COORD,GEOMK,ELS,GEOMA
ELEMENTS = ELS[0]
import numpy as np

na = GEOMA["na"]
nl_a = GEOMA["nla"]

def post(ux,uy,uz,r,thx,thy,thz,Nx,Ny,Nxy,Mx,My,Mxy,Qx,Qy,ex,ey,gxy,e3,e1,th,ssx,ssy,spx,spy,rele,relun,relthn):

    RNx = np.ndarray.flatten(r[0::6])
    RNy = np.ndarray.flatten(r[1::6])
    RNxy = np.ndarray.flatten(r[2::6])
    RMx = np.ndarray.flatten(r[3::6])
    RMy = np.ndarray.flatten(r[4::6])
    RMxy = np.ndarray.flatten(r[5::6])

    relunx = relun[0::3]
    reluny = relun[1::3]
    relunz = relun[2::3]
    relthnx = relthn[0::3]
    relthny = relthn[1::3]
    relthnz = relthn[2::3]

    ux = np.ndarray.flatten(ux)
    uy = np.ndarray.flatten(uy)
    uz = np.ndarray.flatten(uz)
    thx = np.ndarray.flatten(thx) * pow(10, 3)
    thy = np.ndarray.flatten(thy) * pow(10, 3)
    thz = np.ndarray.flatten(thz) * pow(10, 3)

    Nx = np.ndarray.flatten(Nx)
    Ny = np.ndarray.flatten(Ny)
    Nxy = np.ndarray.flatten(Nxy)
    Mx = np.ndarray.flatten(Mx) * pow(10, -3)
    My = np.ndarray.flatten(My) * pow(10, -3)
    Mxy = np.ndarray.flatten(Mxy) * pow(10, -3)
    Qx = np.ndarray.flatten(Qx)
    Qy = np.ndarray.flatten(Qy)

    xna = ['']*na
    yna = ['']*na
    zna = [''] * na
    uxa = ['']*na
    uya = ['']*na
    uza = [''] * na
    thxa = [''] * na
    thya = [''] * na
    thza = [''] * na

    relunxa = [''] * na
    relunya = [''] * na
    relunza = [''] * na
    relthnxa = [''] * na
    relthnya = [''] * na
    relthnza = [''] * na

    RNxa = [''] * na
    RNya = [''] * na
    RNxya = [''] * na
    RMxa = [''] * na
    RMya = [''] * na
    RMxya = [''] * na

    Nxa = [''] * na
    Nya = [''] * na
    Nxya = [''] * na
    Mxa = [''] * na
    Mya = [''] * na
    Mxya = [''] * na
    Qxa = [''] * na
    Qya = [''] * na

    exinfa = [''] * na
    eyinfa = [''] * na
    gxyinfa = [''] * na
    e3infa = [''] * na
    e1infa = [''] * na
    thinfa = [''] * na
    ssxinfa = [''] * na
    ssyinfa = [''] * na
    spxinfa = [''] * na
    spyinfa = [''] * na
    exsupa = [''] * na
    eysupa = [''] * na
    gxysupa = [''] * na
    e3supa = [''] * na
    e1supa = [''] * na
    thsupa = [''] * na
    ssxsupa = [''] * na
    ssysupa = [''] * na
    spxsupa = [''] * na
    spysupa = [''] * na

    relex = rele[:,:,:,:,0]
    reley = rele[:,:,:,:,1]
    relgxy = rele[:,:,:,:,2]
    relexinfa = [''] * na
    releyinfa = [''] * na
    relgxyinfa = [''] * na
    relexsupa = [''] * na
    releysupa = [''] * na
    relgxysupa = [''] * na

    for ia in range(na):
        k_a = np.where(np.array(GEOMK["ak"]) == ia)
        k_a = np.array(k_a)
        k_a = np.ndarray.flatten(k_a)
        n_a = np.unique(ELEMENTS[k_a,:])
        n_a = n_a[n_a<10**5]
        # xna[ia] = COORD["n"][2][ia][n_a][:,0]
        # yna[ia] = COORD["n"][2][ia][n_a][:,1]
        # zna[ia] = COORD["n"][2][ia][n_a][:, 2]
        xna[ia] = COORD["n"][0][n_a,0]
        yna[ia] = COORD["n"][0][n_a,1]
        zna[ia] = COORD["n"][0][n_a,2]
        uxa[ia] = ux[n_a]
        uya[ia] = uy[n_a]
        uza[ia] = uz[n_a]
        thxa[ia] = thx[n_a]
        thya[ia] = thy[n_a]
        thza[ia] = thz[n_a]

        relunxa[ia] = relunx[n_a]
        relunya[ia] = reluny[n_a]
        relunza[ia] = relunz[n_a]
        relthnxa[ia] = relthnx[n_a]
        relthnya[ia] = relthny[n_a]
        relthnza[ia] = relthnz[n_a]

        RNxa[ia] = RNx[n_a]
        RNya[ia] = RNy[n_a]
        RNxya[ia] = RNxy[n_a]
        RMxa[ia] = RMx[n_a]
        RMya[ia] = RMy[n_a]
        RMxya[ia] = RMxy[n_a]

        Nxa[ia] = Nx[k_a]
        Nya[ia] = Ny[k_a]
        Nxya[ia] = Nxy[k_a]
        Mxa[ia] = Mx[k_a]
        Mya[ia] = My[k_a]
        Mxya[ia] = Mxy[k_a]
        Qxa[ia] = Qx[k_a]
        Qya[ia] = Qy[k_a]

        layer = 2 # 0 für untersten
        exinfa[ia] = np.ndarray.flatten(ex[k_a, layer, :, :]*10**3)
        indkeep = [exinfa[ia]>-10**5][0]
        exinfa[ia] = exinfa[ia][indkeep]
        eyinfa[ia] = np.ndarray.flatten(ey[k_a, layer, :, :]*10**3)[indkeep]
        gxyinfa[ia] = np.ndarray.flatten(gxy[k_a, layer, :, :]*10**3)[indkeep]
        e3infa[ia] = np.ndarray.flatten(e3[k_a, layer, :, :]*10**3)[indkeep]
        e1infa[ia] = np.ndarray.flatten(e1[k_a, layer, :, :] * 10 ** 3)[indkeep]
        thinfa[ia] = np.ndarray.flatten(th[k_a, layer, :, :] * 180/3.14159)[indkeep]
        ssxinfa[ia] = np.ndarray.flatten(ssx[k_a, layer, :, :])[indkeep]
        ssyinfa[ia] = np.ndarray.flatten(ssy[k_a, layer, :, :])[indkeep]
        spxinfa[ia] = np.ndarray.flatten(spx[k_a, layer, :, :])[indkeep]
        spyinfa[ia] = np.ndarray.flatten(spy[k_a, layer, :, :])[indkeep]
        relexinfa[ia] = np.ndarray.flatten(relex[k_a, layer, :, :])[indkeep]
        releyinfa[ia] = np.ndarray.flatten(reley[k_a, layer, :, :])[indkeep]
        relgxyinfa[ia] = np.ndarray.flatten(relgxy[k_a, layer, :, :])[indkeep]
        layer = nl_a[ia] - 3 # -1 für obersten
        exsupa[ia] = np.ndarray.flatten(ex[k_a, layer, :, :]*10**3)[indkeep]
        eysupa[ia] = np.ndarray.flatten(ey[k_a, layer, :, :]*10**3)[indkeep]
        gxysupa[ia] = np.ndarray.flatten(gxy[k_a, layer, :, :]*10**3)[indkeep]
        e3supa[ia] = np.ndarray.flatten(e3[k_a, layer, :, :]*10**3)[indkeep]
        e1supa[ia] = np.ndarray.flatten(e1[k_a, layer, :, :] * 10 ** 3)[indkeep]
        thsupa[ia] = np.ndarray.flatten(th[k_a, layer, :, :] * 180/3.14159)[indkeep]
        ssxsupa[ia] = np.ndarray.flatten(ssx[k_a, layer, :, :])[indkeep]
        ssysupa[ia] = np.ndarray.flatten(ssy[k_a, layer, :, :])[indkeep]
        spxsupa[ia] = np.ndarray.flatten(spx[k_a, layer, :, :])[indkeep]
        spysupa[ia] = np.ndarray.flatten(spy[k_a, layer, :, :])[indkeep]
        relexsupa[ia] = np.ndarray.flatten(relex[k_a, layer, :, :])[indkeep]
        releysupa[ia] = np.ndarray.flatten(reley[k_a, layer, :, :])[indkeep]
        relgxysupa[ia] = np.ndarray.flatten(relgxy[k_a, layer, :, :])[indkeep]

    POST = {"xn"    : xna,
            "yn"    : yna,
            "zn"    : zna,
            "ux"    : [ux,uxa],
            "uy"    : [uy, uya],
            "uz"    : [uz, uza],
            "thx"   : [thx, thxa],
            "thy"   : [thy, thya],
            "thz"   : [thz, thza],
            "relunx": [relunx, relunxa],
            "reluny": [reluny, relunya],
            "relunz": [relunz, relunza],
            "relthnx": [relthnx, relthnxa],
            "relthny": [relthny, relthnya],
            "relthnz": [relthnz, relthnza],
            "RNx": [RNx, RNxa],
            "RNy": [RNy, RNya],
            "RNxy": [RNxy, RNxya],
            "RMx"  : [RMx, RMxa],
            "RMy"  : [RMy, RMya],
            "RMxy"  : [RMxy, RMxya],
            "Nx"    : Nxa,
            "Ny"    : Nya,
            "Nxy"   : Nxya,
            "Mx"    : Mxa,
            "My"    : Mya,
            "Mxy"   : Mxya,
            "Qx"    : Qxa,
            "Qy"    : Qya,
            "exinfa": exinfa,
            "exsupa": exsupa,
            "eyinfa": eyinfa,
            "eysupa": eysupa,
            "gxyinfa": gxyinfa,
            "gxysupa": gxysupa,
            "e3infa": e3infa,
            "e1infa": e1infa,
            "thinfa": thinfa,
            "ssxinfa":ssxinfa,
            "ssyinfa":ssyinfa,
            "spxinfa": spxinfa,
            "spyinfa": spyinfa,
            "e3supa": e3supa,
            "e1supa": e1supa,
            "thsupa": thsupa,
            "ssxsupa": ssxsupa,
            "ssysupa": ssysupa,
            "spxsupa": spxsupa,
            "spysupa": spysupa,
            "relexinfa": relexinfa,
            "relexsupa": relexsupa,
            "releyinfa": releyinfa,
            "releysupa": releysupa,
            "relgxyinfa": relgxyinfa,
            "relgxysupa": relgxysupa,
            }
    return POST
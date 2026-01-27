from math import *
import numpy as np
def e_principal(ex, ey, gxy):
    # 1 Mohr's Circle of imposed total strains
    ex = ex
    ey = ey
    gxy = gxy
    r = 1 / 2 * sqrt((ex - ey) ** 2 + gxy ** 2)
    m = (ex + ey) * 1 / 2

    e1 = m + r
    e3 = m - r
    if abs(e1 - ex) < 10 ** -10:
        if ex > ey:
            th = pi / 2 - 10 ** (-10)
        elif ex < ey:
            th = 10 ** (-10)
        elif ex == ey:
            th = pi / 4
    else:
        th = atan(gxy / (2 * (e1 - ex)))

    return [e1, e3, th]
def s_c3(e3, e1, fc_p, e_c0):
    f_c = f_cs(fc_p, e1)
    if e3 < 0:
        if abs(e3) < e_c0:
            sc3 = f_c * (e3 * e3 + 2 * e3 * e_c0) / (e_c0 * e_c0)
        else:
            sc3 = -f_c
    else:
        sc3 = 0
    return sc3
def s_sc(es, f_sy, E_s, E_sh):
    e_sy = f_sy/E_s
    if abs(es) <= e_sy:
        ss = es * E_s
    elif es > 0:
        ss = f_sy + E_sh*(es - e_sy)
    elif es < 0:
        ss = -f_sy + E_sh*(es + e_sy)
    return ss
def f_cs(f_c, e1):
        if e1 < 0:
            e1 = 0
        f_cs = min((pow(f_c,2/3) / (0.4 + 30 * e1)), f_c)
        return f_cs
def tmat(th):
    th = pi/2+th
    T = np.array([[cos(th) * cos(th), sin(th) * sin(th), sin(th) * cos(th)],
                  [sin(th) * sin(th), cos(th) * cos(th), -sin(th) * cos(th)],
                  [-2*sin(th) * cos(th), 2*sin(th) * cos(th), cos(th) * cos(th) - sin(th) * sin(th)]])
    # T = np.array([[cos(th) * cos(th), sin(th) * sin(th), 2*sin(th) * cos(th)],
    #               [sin(th) * sin(th), cos(th) * cos(th), -2*sin(th) * cos(th)],
    #               [-sin(th) * cos(th), sin(th) * cos(th), cos(th) * cos(th) - sin(th) * sin(th)]])
    return T
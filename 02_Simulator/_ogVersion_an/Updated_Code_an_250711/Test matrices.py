import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import numpy as np
import scipy.optimize as opto
import os
import shutil
from math import *

# style.use('fivethirtyeight')
#
# fig = plt.figure()
# ax1 = fig.add_subplot(1,1,1)

# def animate(i):
#     graph_data = open('example.txt','r').read()
#     lines = graph_data.split('\n')
#     xs = []
#     ys = []
#     for line in lines:
#         if len(line)>1:
#             x,y = line.split(',')
#             xs.append(x)
#             ys.append(y)
#     ax1.clear
#     ax1.plot(xs,ys)
# ani = animation.FuncAnimation(fig,animate,interval=1000)
# plt.show()

# for i in range(10):
#     plt.scatter(i, i + 1)
#     plt.pause(0.5)
#
# plt.show()
# a = np.zeros((2,6))
# print(a)

# # B_s = np.array([[0,0,N1x,-N1,0,0,0,N2x,-N2,0,0,0,N3x,-N3,0,0,0,N4x,-N4,0],
# #                 [0,0,N1y,0,-N1,0,0,N2y,0,-N2,0,0,N3y,0,-N3,0,0,N4y,0,-N4]])
# # Bmat_s = np.array([[0, 0, N1x, -N1, 0, 0, 0, N2x, -N2, 0, 0, 0, N3x, -N3, 0, 0, 0, N4x, -N4, 0],
# #                    [0, 0, N1y, 0, -N1, 0, 0, N2y, 0, -N2, 0, 0, N3y, 0, -N3, 0, 0, N4y, 0, -N4],
# #                    [0, 0, N1x, -N1, 0, 0, 0, N2x, -N2, 0, 0, 0, N3x, -N3, 0, 0, 0, N4x, -N4, 0],
# #                    [0, 0, N1y, 0, -N1, 0, 0, N2y, 0, -N2, 0, 0, N3y, 0, -N3, 0, 0, N4y, 0, -N4],
# #                    [0, 0, N1x, -N1, 0, 0, 0, N2x, -N2, 0, 0, 0, N3x, -N3, 0, 0, 0, N4x, -N4, 0],
# #                    [0, 0, N1y, 0, -N1, 0, 0, N2y, 0, -N2, 0, 0, N3y, 0, -N3, 0, 0, N4y, 0, -N4],
# #                    [0, 0, N1x, -N1, 0, 0, 0, N2x, -N2, 0, 0, 0, N3x, -N3, 0, 0, 0, N4x, -N4, 0],
# #                    [0, 0, N1y, 0, -N1, 0, 0, N2y, 0, -N2, 0, 0, N3y, 0, -N3, 0, 0, N4y, 0, -N4]
# #                    ])
# Bmat_s = np.array([[N1x, -N1, 0, N2x, -N2, 0, N3x, -N3, 0, N4x, -N4, 0],
#                    [N1y, 0, -N1, N2y, 0, -N2, N3y, 0, -N3, N4y, 0, -N4],
#                    [N1x, -N1, 0, N2x, -N2, 0, N3x, -N3, 0, N4x, -N4, 0],
#                    [N1y, 0, -N1, N2y, 0, -N2, N3y, 0, -N3, N4y, 0, -N4],
#                    [N1x, -N1, 0, N2x, -N2, 0, N3x, -N3, 0, N4x, -N4, 0],
#                    [N1y, 0, -N1, N2y, 0, -N2, N3y, 0, -N3, N4y, 0, -N4],
#                    [N1x, -N1, 0, N2x, -N2, 0, N3x, -N3, 0, N4x, -N4, 0],
#                    [N1y, 0, -N1, N2y, 0, -N2, N3y, 0, -N3, N4y, 0, -N4],
#                    ])
# # Bmat_s = np.array([[0, -(1-eta)/4, 0, 0, -(1-eta)/4, 0, 0, -(1+eta)/4, 0, 0, -(1+eta)/4, 0],
# #                    [0, 0, -(1-xi)/4, 0, 0, -(1+xi)/4, 0, 0, -(1+xi)/4, 0, 0, -(1-xi)/4],
# #                    [0, -(1 - eta) / 4, 0, 0, -(1 - eta) / 4, 0, 0, -(1 + eta) / 4, 0, 0, -(1 + eta) / 4, 0],
# #                    [0, 0, -(1 - xi) / 4, 0, 0, -(1 + xi) / 4, 0, 0, -(1 + xi) / 4, 0, 0, -(1 - xi) / 4],
# #                    [0, -(1 - eta) / 4, 0, 0, -(1 - eta) / 4, 0, 0, -(1 + eta) / 4, 0, 0, -(1 + eta) / 4, 0],
# #                    [0, 0, -(1 - xi) / 4, 0, 0, -(1 + xi) / 4, 0, 0, -(1 + xi) / 4, 0, 0, -(1 - xi) / 4],
# #                    [0, -(1 - eta) / 4, 0, 0, -(1 - eta) / 4, 0, 0, -(1 + eta) / 4, 0, 0, -(1 + eta) / 4, 0],
# #                    [0, 0, -(1 - xi) / 4, 0, 0, -(1 + xi) / 4, 0, 0, -(1 + xi) / 4, 0, 0, -(1 - xi) / 4],
# #                    ])
# Cmat = np.array([[J[0, 0], J[0, 1], 0, 0, 0, 0, 0, 0],
#                  [J[1, 0], J[1, 1], 0, 0, 0, 0, 0, 0],
#                  [0, 0, J[0, 0], J[0, 1], 0, 0, 0, 0],
#                  [0, 0, J[1, 0], J[1, 1], 0, 0, 0, 0],
#                  [0, 0, 0, 0, J[0, 0], J[0, 1], 0, 0],
#                  [0, 0, 0, 0, J[1, 0], J[1, 1], 0, 0],
#                  [0, 0, 0, 0, 0, 0, J[0, 0], J[0, 1]],
#                  [0, 0, 0, 0, 0, 0, J[1, 0], J[1, 1]],
#                  ])
# Tmat = np.array([[1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, -1, 0, 0, 0],
#                  [0, 0, 0, 0, 0, 0, 0, 1]])
# Pmat = np.array([[1, -1, 0, 0], [0, 0, 1, 1], [-1, -1, 0, 0], [0, 0, 1, -1]])
# Amat = np.array([[1, eta, 0, 0], [0, 0, 1, xi]])
# B_ss = np.matmul(J_t_inv,
#                  np.matmul(Amat, np.matmul(np.linalg.inv(Pmat), np.matmul(Tmat, np.matmul(Cmat, Bmat_s)))))
# B_s1 = np.append([[0, 0], [0, 0]], B_ss[:, 0:3], axis=1)
# B_s2 = np.append([[0, 0], [0, 0]], B_ss[:, 3:6], axis=1)
# B_s3 = np.append([[0, 0], [0, 0]], B_ss[:, 6:9], axis=1)
# B_s4 = np.append([[0, 0], [0, 0]], B_ss[:, 9:12], axis=1)
# B_s12 = np.append(B_s1, B_s2, axis=1)
# B_s34 = np.append(B_s3, B_s4, axis=1)
# B_s = np.append(B_s12, B_s34, axis=1)

# a = np.array([[1,2],[3,4],[4,5]])
# b = np.array([2,5])
# # if b in a.tolist():
# #     print("jaaaaa")
# if any((a[:]==b).all(1)):
#     print("jaaaa")
# any((a[:]==[1,3]).all(1))

# a = [[1,2],[3,4],[4,5]]
# b = [3,4]
# if b in a:
#     print("jaaa")

# def func1(x):
#     y = x**2+3*x-2
#     return y
#
# x0 = 4
# x = opto.fsolve(func1,x0)
# print(x)
# print(x**2+3*x-2)
# Fs = np.zeros(50)
# print(Fs[5])
# a = np.array([0,1,2,3])
# b = sum(a)
# print(a*a)
# a = np.array([[1],[5],[3],[8],[7]])
# b = np.sort(a,axis=0)
# ind = np.argsort(a,axis=0)
# u=1
# c = a/a
# print(c)

# import matplotlib.pyplot as plt
# # This import registers the 3D projection, but is otherwise unused.
# from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
# from mpl_toolkits.mplot3d.axes3d import get_test_data
# import numpy as np
# fig = plt.figure()
# ax = fig.add_subplot(1,2,1,projection='3d')
# X, Y, Z = get_test_data(0.5)
# C = np.linspace(-5, 5, Z.size).reshape(Z.shape)
# scamap = plt.cm.ScalarMappable(cmap='inferno')
# fcolors = scamap.to_rgba(C)
# ax.plot_surface(X, Y, Z, facecolors=fcolors, cmap='inferno')
# fig.colorbar(scamap)
#
# ax2 = fig.add_subplot(1,2,2,projection='3d')
# X, Y, Z = get_test_data(0.1)
# C = np.linspace(-5, 5, Z.size).reshape(Z.shape)
# scamap = plt.cm.ScalarMappable(cmap='inferno')
# fcolors = scamap.to_rgba(C)
# ax2.plot_surface(X, Y, Z, facecolors=fcolors, cmap='inferno')
# fig.colorbar(scamap)
# plt.show()

# a = np.array([1,0,0])
# b = np.array([1,0,0])
# from numpy import dot
# from numpy.linalg import norm
# cos_ab = dot(a,b)/(norm(a)*norm(b))
# print(cos_ab)

# A = np.array([[1,2,3],[2,3,4],[3,4,5]])
# b = np.where(A==2)
# c = np.cross([1,0,0],[0,1,0])
# print(c)
# A = np.array([[1,0,0],[0,0,1],[0,-1,0]])
# print(np.linalg.inv(A))
# B = np.array([[5,6,7],[6,7,8]])
# B[1:2,1]=np.zeros((2,1))
# print(B)
#
# A = np.array([[1,2,3],[2,3,4],[3,4,5],[1,3,2]])
# b = np.where(A == 2)
# print(A)
# print(A[:,0:2])
# mylist = ['']*3
# mylist[0] = A
# mylist[1] = np.array([[1,2,3],[4,5,6]])
# print(mylist[0])
# print(mylist[1])
# shared_rows =np.zeros((3,1))
# count = 0
# A = np.array([[1,2],[2,3],[3,4]])
# B = np.array([[1,2],[3,4]])
# for row in A:
#     if row in B:
#         shared_rows[count]=1
#     else:
#         shared_rows[count]=0
#     count+=1
#
# print(A[(True,False,True),:])
# # print(c)
# a = np.array(list(range(1,101)))
# print(a)
# b = np.percentile(a,50)
# print(b)
# a=np.array([[1,2,3],[[],1,2]])
# a = np.ndarray.flatten(a)
# a1 = np.array([i for i in a if i])
# print(a1)
# a = np.array([1,2,3,4,5,6,7,8,9])
# b = a[1::2]
# print(b)
# a = np.array([[5],[1],[0]])
# b = a[a<3]
# print(b)
# b=a-0.5
# print(b)

# q_e1 = np.zeros((6,1))
# print(q_e1.flatten())
# q_e[0][3] = 2
# print(q_e[0][1:4])
# test = np.ones((1,6))
# testdia = np.diag(test,1)
# print(test)
# class test():
#     def __init__(self,num):
#         self.n = num*2
# n1 = test(2)
# n2 = test(3)
# bb = np.ndarray((4,3),dtype=object)
# b = np.empty_like(bb)
# bb = np.array()
# print(bb)
# bb[1][1]=n1
# a = np.array([n1,n2])
# print(bb[1][1].n)

# from numpy import save
# from numpy import load
# import pickle
# import os
# a = np.array([[[1],[3]],[[5],[6]]])
# path = "/Users/naesboma/Desktop/FEM_Q/Calculations/"
# save(os.path.join(path, 'data.npy'), a)
# b = load(os.path.join(path, 'data.npy'))
#
# testdic = {"a":    np.array([1,2,3,4,5]),
#             "bb":   [4,3,4],
#             "ccccc": np.array([[1,2],[4,5]])}
# with open(os.path.join(path,'saved_dictionary.pkl'), 'wb') as f:
#     pickle.dump(testdic, f)
# with open(os.path.join(path,'saved_dictionary.pkl'), 'rb') as f:
#     loaded_dict = pickle.load(f)
# print(testdic["bb"])
# print(loaded_dict["bb"])

import numpy as np
# from Stresses_mixreinf import fun1
# f = fun1([3,5,8])
# f.out(4)
# print(f.y)
#
# f.out(7)
# print(f.y)

# a = np.array([1,2,3])
# y = a
# y = np.array([6,5,4])
# print(a)

# a=np.array([10,4,6,0,-4,np.nan])
# print(min(a))
# a = np.array([[1,2,3,4,5,6,7],[9,8,7,6,5,4,3]])
# b = np.array(np.where(a[0,:]>4))
# b = b.flatten()
# c=a[:,b]
# d=np.array(np.where(c[1,:]>3))
# d=d.flatten()
# print(len(d))

# import numpy as np
# import pandas as pd
# from numpy import linspace, meshgrid
# from scipy.interpolate import griddata
# import plotly.graph_objects as go
# #Dummy Data
# x = [2,3,4,6,7,8,9,10]
# y = [2,4,2,5,3,2,5,4]
# z = [100,80,90,70,70,60,40,40]
#
# xi = linspace(min(x),max(x),200)
# yi = linspace(min(y),max(y),200)
# X,Y = meshgrid(xi,yi)
# Z = griddata((x,y),z,(X,Y), method='cubic')
# fig = go.Figure()
# fig.add_trace(go.Contour(x=xi, y=yi, z=Z, line_smoothing=1.3))
# # fig.add_trace(go.Scatter(x=x,y=y, mode='markers+text', text=z))
# fig.update_layout(autosize=False)
# fig.show()

# path = r"C:\Users\naesboma\00_an\FEM_Q\Calculations\24_11_11_Prototyp"
# a = [path,r"\LS",str(1)]
# folder_i = ''.join(a)
#
# try:
#     os.makedirs(folder_i)
# except FileExistsError:
#     shutil.rmtree(folder_i)
#     os.makedirs(folder_i)
# except PermissionError:
#     print(f"Permission denied: Unable to create '{folder_i}'. Load Step results not saved")
# except Exception as e:
#     print(f"An error occurred: {e}. Load Step results not saved")
# print(np.mean([1,2,3,4,5]))

""" Walraven """
# # from scipy.integrate import quad
# from defcplx import *
# from defcplx import cplx
#
# def TPM_ContactArea(w,v,Dmax,p_k):
#     if w<0:
#         w=0
#
#     # u_max = lambda D: (-1/2*w*(w**2+v**2)+1/2*sqrt(w**2*(w**2+v**2)**2-(w**2+v**2)*((w**2+v**2)**2-v**2*D**2)))/(w**2+v**2)
#     # Fl = lambda D: 0.532*(D/Dmax)**0.5-0.212*(D/Dmax)**4-0.072*(D/Dmax)**6-0.036*(D/Dmax)**8-0.025*(D/Dmax)**10
#     # G1l = lambda D: D**-3*(sqrt(D**2-(w**2+v**2))*v/(sqrt(w**2+v**2))**u_max(D,v,w)-w**u_max(D,v,w)-u_max(D,v,w)**2)
#     # G2l = lambda D: D**-3*((v-sqrt(D**2-(w**2+v**2))*w/sqrt(w**2+v**2))*u_max(D,v,w)+(u_max(D,v,w)+w)*
#     #                        sqrt(1/4*D**2-(w+u_max(D,v,w))**2)-w*sqrt(1/4*D**2-w**2)+1/4*D**2*asin((w+u_max(D,v,w))/(1/2*D))-D**2/4*asin(2*w/D))
#     # G3l = lambda D: D**-3*(1/2*D-w)**2
#     # G4l = lambda D: D**-3*(np.pi/8*D**2-w*sqrt(1/4*D**2-w**2)-D**2/4*asin(2*w/D))
#     #
#     # def F(D,Dmax,v,w):
#     #     return Fl(D)
#     # def G1(D):
#     #     return G1l(D)
#     # def G2(D):
#     #     return G2l(D)
#     # def G3(D):
#     #     return G3l(D)
#     # def G4(D):
#     #     return G4l(D)
#     def u_max(D,v,w):
#         u_max = (-1 / 2 * w * (w ** 2 + v ** 2) + 1 / 2 * sqrt(
#             w ** 2 * (w ** 2 + v ** 2) ** 2 - (w ** 2 + v ** 2) * ((w ** 2 + v ** 2) ** 2 - v ** 2 * D ** 2))) / (
#                                       w ** 2 + v ** 2)
#         return u_max
#
#     def F(D,Dmax,v,w):
#         F = 0.532*(D/Dmax)**0.5-0.212*(D/Dmax)**4-0.072*(D/Dmax)**6-0.036*(D/Dmax)**8-0.025*(D/Dmax)**10
#         return F
#
#     def G1(D,Dmax,v,w):
#         # print(u_max(D,v,w))
#         G1 = D**-3*(sqrt(D**2-(w**2+v**2))*v/(sqrt(w**2+v**2))*u_max(D,v,w)-w*u_max(D,v,w)-u_max(D,v,w)**2)
#         return G1
#
#     def G2(D,Dmax,v,w):
#         G2 = D**-3*((v-sqrt(D**2-(w**2+v**2))*w/sqrt(w**2+v**2))*u_max(D,v,w)+(u_max(D,v,w)+w)*sqrt(1/4*D**2-(w+u_max(D,v,w))**2)-
#                     w*sqrt(1/4*D**2-w**2)+1/4*D**2*asin((w+u_max(D,v,w))/(1/2*D))-D**2/4*asin(2*w/D))
#         return G2
#
#     def G3(D,Dmax,v,w):
#         G3 = D**-3*(1/2*D-w)**2
#         return G3
#
#     def G4(D,Dmax,v,w):
#         G4 = D**-3*(np.pi/8*D**2-w*sqrt(1/4*D**2-w**2)-D**2/4*asin(2*w/D))
#         return G4
#
#     def integrate_cplx(function,Dmax,v,w,a,b):
#         steps = 10
#         x = np.linspace(a,b,steps)
#
#         ireal = 0
#         iim = 0
#         for i in range(steps-1):
#
#             x0 = x[i]
#             x1 = x[i+1]
#             y = function((x0+x1)/2,Dmax,v,w)
#             ireal += y.real*(x1.real-x0.real)
#             iim += y.imag*(x1.imag-x0.imag)
#         return cplx(ireal,iim)
#
#     if v < w:
#         if (w ** 2 + v ** 2) / v < Dmax:
#             def ftemp1(D,Dmax,v,w):
#                 return p_k* 4 / np.pi* F(D,Dmax,v,w)* G1(D,Dmax,v,w)
#             a = (w ** 2 + v ** 2) / v
#             b = Dmax
#             A_y_bar = integrate_cplx(ftemp1,Dmax,v,w,a,b)
#
#             def ftemp2(D,Dmax,v,w):
#                 return p_k* 4 / np.pi* F(D,Dmax,v,w)* G2(D,Dmax,v,w)
#             a = (w ** 2 + v ** 2) / v
#             b = Dmax
#             A_x_bar = integrate_cplx(ftemp2,Dmax,v,w,a,b)
#         else:
#             A_x_bar = 0
#             A_y_bar = 0
#     else:
#         if (w ** 2 + v ** 2) / w < Dmax:
#             def ftemp31(D,Dmax,v,w):
#                 return p_k* 4 / np.pi* F(D,Dmax,v,w)* G1(D,Dmax,v,w)
#             def ftemp32(D,Dmax,v,w):
#                 return p_k* 4 / np.pi* F(D,Dmax,v,w)* G3(D,Dmax,v,w)
#             a31 = (w**2+v**2)/w
#             b31 = Dmax
#             a32 = 2 * w
#             b32 = (w ** 2 + v ** 2) / w
#             A_y_bar = integrate_cplx(ftemp31,Dmax,v,w,a31,b31) + integrate_cplx(ftemp32,Dmax,v,w,a32,b32)
#
#             def ftemp41(D,Dmax,v,w):
#                 return p_k* 4 / np.pi* F(D,Dmax,v,w)* G2(D,Dmax,v,w)
#             def ftemp42(D,Dmax,v,w):
#                 return p_k* 4 / np.pi* F(D,Dmax,v,w)* G4(D,Dmax,v,w)
#             a41 = (w**2+v**2)/w
#             b41 = Dmax
#             a42 = 2 * w
#             b42 = (w ** 2 + v ** 2) / w
#             A_x_bar = integrate_cplx(ftemp41,Dmax,v,w,a41,b41) + integrate_cplx(ftemp42,Dmax,v,w,a42,b42)
#
#         elif 2 * w < Dmax:
#             print('adsf')
#             def ftemp5(D,Dmax,v,w):
#                 return p_k* 4 / np.pi* F(D,Dmax,v,w)* G3(D,Dmax,v,w)
#             a = 2 * w
#             b = Dmax
#             A_y_bar = integrate_cplx(ftemp5,Dmax,v,w,a,b)
#             # A_y_bar = quad(ftemp5, 2 * w, Dmax)[0]
#
#             def ftemp6(D,Dmax,v,w):
#                 return p_k* 4 / np.pi* F(D,Dmax,v,w)* G4(D,Dmax,v,w)
#             a = 2 * w
#             b = Dmax
#             A_x_bar = integrate_cplx(ftemp6,Dmax,v,w,a,b)
#             # A_x_bar = quad(ftemp6, 2 * w, Dmax)[0]
#         else:
#             A_x_bar = 0
#             A_y_bar = 0
#     # print([A_y_bar,A_x_bar])
#     return[A_y_bar,A_x_bar]
#
# def TPM_Stresses(dn,dt):
#
#     fc = 40
#     fcc = 35
#     Dmax = 32
#
#     sigma_pu = 6.39*fcc**0.56
#     p_k = 0.75
#     my = 0.4
#
#     sign_dt = np.sign(dt)
#
#     [Ay_bar,Ax_bar] = TPM_ContactArea(dn, abs(dt), Dmax, p_k)
#     print(Ax_bar,Ay_bar)
#     sigma = -sigma_pu * (Ax_bar - my * Ay_bar)
#     tau = sign_dt * sigma_pu * (Ay_bar + my * Ax_bar)
#
#     # Yang
#     if fc >= 65:
#         R_ai = min([0.85 * sqrt((7.2 / (fc - 40) + 1) ** 2 - 1) + 0.34, 1])
#     else:
#         R_ai = 1
#
#     sigma = R_ai * sigma
#     tau = R_ai * tau
#
#     if sigma > 0:
#         sigma = 0
#
#     return[sigma,tau]
#
#
# dn = 0.2676
# dt = 0.0145
# [sigma,tau] = TPM_Stresses(dn,dt)
# print(sigma,tau)



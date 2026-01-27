import cmath
import numpy as np
import scipy

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


def log(x):
    return cplx(cmath.log(x))

def vecmean(x):
    y = np.zeros_like(x)
    z = np.zeros_like(x)
    for i in range(len(x)):
        y[i] = x[i].real
        z[i] = x[i].imag
    return cplx(np.mean(y),np.mean(z))


def abs(x):
    return cplx(cmath.sqrt(x*x))

# def quad(f,a,b):
#     def real_func(x):
#         x = x.real
#         return f(x)
#     def imag_func(x):
#         x = x.imag
#         return f(x)
#     real_integral = quad(real_func, a.real, b.real)
#     imag_integral = quad(imag_func, a.imag, b.imag)
#     return (real_integral[0],1j*imag_integral[0])

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


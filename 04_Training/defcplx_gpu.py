import cmath
import cupy as np
lbd = 0.67

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

# class cplx(complex):
#     """ ----------------------------------- Custom class for complex numbers ----------------------------------------
#         - syntax: cplx(real,imag)
#         - supported mathematical operations: "+", "-", "*", "/", "**"
#         - supported comparisons: ">", ">=", "==", "<=", "<", "!="
#         -------------------------------------------------------------------------------------------------------------"""
#     def __repr__(self):
#         return 'cplx(%r, %r)' % (self.real, self.imag)

#     def __add__(self,x):
#         return cplx(complex.__add__(self, x))

#     def __radd__(self,x):
#         return cplx(complex.__radd__(self, x))

#     def __sub__(self,x):
#         return cplx(complex.__sub__(self, x))

#     def __rsub__(self,x):
#         return cplx(complex.__rsub__(self, x))

#     def __mul__(self,x):
#         return cplx(complex.__mul__(self, x))

#     def __rmul__(self,x):
#         return cplx(complex.__rmul__(self, x))

#     def __truediv__(self,x):
#         return cplx(complex.__truediv__(self,x))

#     def __rtruediv__(self,x):
#         return cplx(complex.__rtruediv__(self,x))

#     def __pow__(self,x):
#         return cplx(complex.__pow__(self, x))

#     def __rpow__(self,x):
#         return cplx(complex.__rpow__(self, x))

#     def __lt__(self,x):
#         return self.real < x.real

#     def __le__(self,x):
#         return self.real <= x.real

#     def __gt__(self,x):
#         return self.real > x.real

#     def __ge__(self,x):
#         return self.real >= x.real

#     def __eq__(self,x):
#         return self.real == x.real

#     def __ne__(self,x):
#         return self.real != x.real




class cplx(np.ndarray):
    """GPU-compatible complex number class using CuPy (imported as np)."""

    def __new__(cls, real=0.0, imag=0.0, use_gpu=True):
        # Create the underlying GPU array (0-D CuPy array)
        val = np.array(real + 1j * imag, dtype=np.complex128).view(cls)
        val.use_gpu = use_gpu
        return val

    def __array_finalize__(self, obj):
        # Preserve attributes when new views/results are created
        if obj is None:
            return
        self.use_gpu = getattr(obj, "use_gpu", True)

    def __repr__(self):
        return f"cplx({self.real.item()}, {self.imag.item()})"

    # ---------- arithmetic ----------
    def __add__(self, other):
        return cplx.from_value(np.add(self.view(np.ndarray), self._unwrap(other)))
    __radd__ = __add__

    def __sub__(self, other):
        return cplx.from_value(np.subtract(self.view(np.ndarray), self._unwrap(other)))
    def __rsub__(self, other):
        return cplx.from_value(np.subtract(self._unwrap(other), self.view(np.ndarray)))

    def __mul__(self, other):
        return cplx.from_value(np.multiply(self.view(np.ndarray), self._unwrap(other)))
    __rmul__ = __mul__

    def __truediv__(self, other):
        return cplx.from_value(np.divide(self.view(np.ndarray), self._unwrap(other)))
    def __rtruediv__(self, other):
        return cplx.from_value(np.divide(self._unwrap(other), self.view(np.ndarray)))

    def __pow__(self, other):
        return cplx.from_value(np.power(self.view(np.ndarray), self._unwrap(other)))
    def __rpow__(self, other):
        return cplx.from_value(np.power(self._unwrap(other), self.view(np.ndarray)))

    # ---------- comparisons ----------
    def __lt__(self, other): return self.real.item() < self._unwrap(other).real
    def __le__(self, other): return self.real.item() <= self._unwrap(other).real
    def __gt__(self, other): return self.real.item() > self._unwrap(other).real
    def __ge__(self, other): return self.real.item() >= self._unwrap(other).real
    def __eq__(self, other): return self.real.item() == self._unwrap(other).real
    def __ne__(self, other): return self.real.item() != self._unwrap(other).real

    # ---------- helpers ----------
    def _unwrap(self, other):
        if isinstance(other, cplx):
            return other.view(np.ndarray)
        return other

    @classmethod
    def from_value(cls, val, use_gpu=True):
        obj = np.array(val, dtype=np.complex128).view(cls)
        obj.use_gpu = use_gpu
        return obj

    def to_cpu(self):
        """Return plain Python complex scalar (transfer from GPU)."""
        return np.asnumpy(self).item()
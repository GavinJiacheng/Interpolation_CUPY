import scipy.special as spec
from scipy.special import comb
import cupy as cp
import numpy as np
import operator
import functools
from scipy.interpolate import _ppoly
from numba import cuda
from numba import jit


@cuda.jit('void(float64[:,:,:], float64[:], float64[:], float64[:,:])')
def evaluate(c, x, xp, out):

    dx = 0
    extrapolate = True
    interval = 0
    start = cuda.grid(1)
    #start = 0
    stride = cuda.gridsize(1)
    #stride =1
    length = len(xp)
    cshape2 = c.shape[2]
    for ip in range(start, length, stride):
        xval = xp[ip]
        # Find correct interval
        # funciton start: -------------------------------------------------------
        a = x[0]
        b = x[x.shape[0]-1]

        it = 0
        if it < 0 or it >= x.shape[0]:
            it = 0

        if not (a <= xval <= b):
        # Out-of-bounds (or nan)
            if xval < a and extrapolate:
            # below
                it = 0
            elif xval > b and extrapolate:
            # above
                it = x.shape[0] - 2
            else:
            # nan or no extrapolation
                it = -1
        elif xval == b:
        # Make the interval closed from the right
            it = x.shape[0] - 2
        else:
        # Find the interval the coordinate is in
        # (binary search with locality)
            if xval >= x[it]:
                low = it
                high = x.shape[0]-2
            else:
                low = 0
                high = it

            if xval < x[low+1]:
                high = low
            while low < high:
                mid = (high + low)//2
                if xval < x[mid]:
                # mid < high
                    high = mid
                elif xval >= x[mid + 1]:
                    low = mid + 1
                else:
                # x[mid] <= xval < x[mid+1]
                    low = mid
                    break

            it = low
        # function end -----------------------------------------------------------------
        i = it
        if i < 0:
            for jp in range(0, cshape2, 1):
                out[ip, jp] = 0
            continue
        else:
            interval = i

        ci = interval
        for jp in range(0, cshape2, 1):
            ss = xval - x[interval]
            cj = jp
            # function start: ----------------------------------------------------------------------
            res = 0.0
            z = 1.0
            cshape1 = c.shape[0]

            for kp in range(0, cshape1, 1):
                # prefactor of term after differentiation
                if dx == 0:
                    prefactor = 1.0
                elif dx > 0:
                    # derivative
                    if kp < dx:
                        continue
                    else:
                        prefactor = 1.0
                        for k in range(kp, kp - dx, -1):
                            prefactor *= k
                else:
                    # antiderivative
                    prefactor = 1.0
                    for k in range(kp, kp - dx):
                        prefactor /= k + 1

                res = res + c[c.shape[0] - kp - 1, ci, cj] * z * prefactor

                if kp < c.shape[0] - 1 and kp >= dx:
                    z *= ss
            # function end ----------------------------------------------------------------------
            out[ip][jp] = res
    #cuda.defer_cleanup()
    # out[1], out[2]
    # out[2], out[1]
    # f(x) = a + bx + cx^2 .....



def prod(x):
    """Product of a list of numbers; ~40x faster vs np.prod for Python tuples"""
    if len(x) == 0:
        return 1
    return functools.reduce(operator.mul, x)

class _PPolyBase(object):
    __slots__ = ('c', 'x', 'extrapolate', 'axis')

    def __init__(self, c, x, extrapolate=None, axis=0):
        self.c = cp.asarray(c)
        self.x = cp.ascontiguousarray(x, dtype=cp.float64)
        if extrapolate is None:
            extrapolate = True
        self.extrapolate = bool(extrapolate)


        if self.c.ndim < 2:
            raise ValueError("Coefficients array must be at least "
                             "2-dimensional.")

        if not (0 <= axis < self.c.ndim - 1):
            raise ValueError("axis=%s must be between 0 and %s" %
                             (axis, self.c.ndim-1))

        self.axis = axis
        if axis != 0:
            self.c = cp.rollaxis(self.c, axis+1)
            self.c = cp.rollaxis(self.c, axis+1)

        if self.x.ndim != 1:
            raise ValueError("x must be 1-dimensional")
        if self.x.size < 2:
            raise ValueError("at least 2 breakpoints are needed")
        if self.c.ndim < 2:
            raise ValueError("c must have at least 2 dimensions")
        if self.c.shape[0] == 0:
            raise ValueError("polynomial must be at least of order 0")
        if self.c.shape[1] != self.x.size-1:
            raise ValueError("number of coefficients != len(x)-1")
        dx = cp.diff(self.x)
        if not (cp.all(dx >= 0) or cp.all(dx <= 0)):
            raise ValueError("`x` must be strictly increasing or decreasing.")

        dtype = self._get_dtype(self.c.dtype)
        self.c = cp.ascontiguousarray(self.c, dtype=dtype)

    def _get_dtype(self, dtype):
        if cp.issubdtype(dtype, cp.complexfloating) \
               or cp.issubdtype(self.c.dtype, cp.complexfloating):
            return cp.complex_
        else:
            return cp.float_

    @classmethod
    def construct_fast(cls, c, x, axis=-1):
        self = object.__new__(cls)
        self.c = c
        self.x = x
        self.axis = axis
        return self

    def _ensure_c_contiguous(self):
        if not self.x.flags.c_contiguous:
            self.x = self.x.copy()
        if not self.c.flags.c_contiguous:
            self.c = self.c.copy()

    def extend(self, c, x, right=None):
        if right is not None:
            warnings.warn("`right` is deprecated and will be removed.")

        c = cp.asarray(c)
        x = cp.asarray(x)

        if c.ndim < 2:
            raise ValueError("invalid dimensions for c")
        if x.ndim != 1:
            raise ValueError("invalid dimensions for x")
        if x.shape[0] != c.shape[1]:
            raise ValueError("x and c have incompatible sizes")
        if c.shape[2:] != self.c.shape[2:] or c.ndim != self.c.ndim:
            raise ValueError("c and self.c have incompatible shapes")

        if c.size == 0:
            return

        dx = cp.diff(x)
        if not (cp.all(dx >= 0) or cp.all(dx <= 0)):
            raise ValueError("`x` is not sorted.")

        if self.x[-1] >= self.x[0]:
            if not x[-1] >= x[0]:
                raise ValueError("`x` is in the different order "
                                 "than `self.x`.")

            if x[0] >= self.x[-1]:
                action = 'append'
            elif x[-1] <= self.x[0]:
                action = 'prepend'
            else:
                raise ValueError("`x` is neither on the left or on the right "
                                 "from `self.x`.")
        else:
            if not x[-1] <= x[0]:
                raise ValueError("`x` is in the different order "
                                 "than `self.x`.")

            if x[0] <= self.x[-1]:
                action = 'append'
            elif x[-1] >= self.x[0]:
                action = 'prepend'
            else:
                raise ValueError("`x` is neither on the left or on the right "
                                 "from `self.x`.")

        dtype = self._get_dtype(c.dtype)

        k2 = max(c.shape[0], self.c.shape[0])
        c2 = cp.zeros((k2, self.c.shape[1] + c.shape[1]) + self.c.shape[2:],
                      dtype=dtype)

        if action == 'append':
            c2[k2-self.c.shape[0]:, :self.c.shape[1]] = self.c
            c2[k2-c.shape[0]:, self.c.shape[1]:] = c
            self.x = cp.r_[self.x, x]
        elif action == 'prepend':
            c2[k2-self.c.shape[0]:, :c.shape[1]] = c
            c2[k2-c.shape[0]:, c.shape[1]:] = self.c
            self.x = cp.r_[x, self.x]

        self.c = c2

    def __call__(self, x, nu=0, extrapolate=None):
        if extrapolate is None:
            extrapolate = self.extrapolate
        x = cp.asarray(x)
        x_shape, x_ndim = x.shape, x.ndim
        x = cp.ascontiguousarray(x.ravel(), dtype=cp.float_)
        out = cp.empty((len(x), prod(self.c.shape[2:])), dtype=self.c.dtype)
        self._ensure_c_contiguous()
        self._evaluate(x, nu, extrapolate, out)
        out = out.reshape(x_shape + self.c.shape[2:])
        if self.axis != 0:
            # transpose to move the calculated values to the interpolation axis
            l = list(range(out.ndim))
            l = l[x_ndim:x_ndim+self.axis] + l[:x_ndim] + l[x_ndim+self.axis:]
            out = out.transpose(l)
        return out


class PPoly(_PPolyBase):

    def _evaluate(self, x, nu, extrapolate, out):
        evaluate[2048,256](self.c.reshape(self.c.shape[0], self.c.shape[1], -1),
                        self.x, x, out)
        cuda.synchronize()
        #threads, cores
        #1,2,3,4
        #256 cores, each core run 1024 threads
        #256 * 1024 functions running at the same time

    @classmethod
    def from_spline(cls, tck, extrapolate=None):
        if isinstance(tck, BSpline):
            t, c, k = tck.tck
            if extrapolate is None:
                extrapolate = tck.extrapolate
        else:
            t, c, k = tck

        cvals = cp.empty((k + 1, len(t)-1), dtype=c.dtype)
        for m in range(k, -1, -1):
            y = fitpack.splev(t[:-1], tck, der=m)
            cvals[k - m, :] = y/spec.gamma(m+1)

        return cls.construct_fast(cvals, t, extrapolate)

    @classmethod
    def from_bernstein_basis(cls, bp, extrapolate=None):
        if not isinstance(bp, BPoly):
            raise TypeError(".from_bernstein_basis only accepts BPoly instances. "
                            "Got %s instead." % type(bp))

        dx = cp.diff(bp.x)
        k = bp.c.shape[0] - 1

        rest = (None,)*(bp.c.ndim-2)

        c = cp.zeros_like(bp.c)
        for a in range(k+1):
            factor = (-1)**a * comb(k, a) * bp.c[a]
            for s in range(a, k+1):
                val = comb(k-a, s-a) * (-1)**s
                c[k-s] += factor * val / dx[(slice(None),)+rest]**s

        if extrapolate is None:
            extrapolate = bp.extrapolate

        return cls.construct_fast(c, bp.x, extrapolate, bp.axis)

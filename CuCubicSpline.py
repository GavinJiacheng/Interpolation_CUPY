import cupy as np
import cupy
from .PPoly import PPoly
from scipy.linalg.blas import _memoize_get_funcs
import scipy.linalg.blas as bla
import scipy.linalg as scl

# these helpful functions are not needed for the core function
'''
def _get_funcs(names, arrays, dtype,
               lib_name, fmodule, cmodule,
               fmodule_name, cmodule_name, alias):

    funcs = []
    unpack = False
    dtype = np.dtype(dtype)
    module1 = (cmodule, cmodule_name)
    module2 = (fmodule, fmodule_name)

    prefix = 'd'
    # d , dtype
    func_name = prefix + names[0]
    func_name = alias.get(func_name, func_name)
    func = getattr(module1[0], func_name, None)
    module_name = module1[1]
    #if func is None:
    #    func = getattr(module2[0], func_name, None)
    #    module_name = module2[1]
    #if func is None:
    #    raise ValueError("error on getfuncs")
    func.module_name, func.typecode = module_name, prefix
    func.dtype = dtype
    func.prefix = prefix  # Backward compatibility
    funcs.append(func)

    return funcs

def get_lapack_funcs(names, arrays=(), dtype=None):
    return _get_funcs(names, arrays, dtype,
                      "LAPACK", 3, 4,
                      "flapack", "clapack", 5)
'''

def solve_banded(l_and_u, ab, b, overwrite_ab=False, overwrite_b=False,
                 debug=None, check_finite=True):
    a1 = cupy.asnumpy(ab)
    b1 = cupy.asnumpy(b)
    if a1.shape[-1] != b1.shape[0]:
        raise ValueError("shapes of ab and b are not compatible.")
    (nlower, nupper) = l_and_u

    overwrite_b = overwrite_b or _datacopied(b1, b)
    if a1.shape[-1] == 1:
        b2 = np.array(b1, copy=(not overwrite_b))
        b2 /= a1[1, 0]
        return b2
    if nlower == nupper == 1:
        overwrite_ab = overwrite_ab or _datacopied(a1, ab) # _datacopied
        gtsv, = scl.get_lapack_funcs(('gtsv',), (a1, b1)) # get_lapack_funcs
        du = a1[0, 1:]
        d = a1[1, :]
        dl = a1[2, :-1]
        du2, d, du, x, info = gtsv(dl, d, du, b1, overwrite_ab, overwrite_ab,
                                   overwrite_ab, overwrite_b) #gtsv
    if info == 0:
        return x
    if info > 0:
        raise LinAlgError("singular matrix")
    raise ValueError('illegal value in %d-th argument of internal '
                     'gbsv/gtsv' % -info)

def prepare_input(x, y, axis, dydx=None):
    x, y = map(np.asarray, (x, y))
    if np.issubdtype(x.dtype, np.complexfloating):
        raise ValueError("`x` must contain real values.")
    x = x.astype(float)

    if np.issubdtype(y.dtype, np.complexfloating):
        dtype = complex
    else:
        dtype = float

    if dydx is not None:
        dydx = np.asarray(dydx)
        if y.shape != dydx.shape:
            raise ValueError("The shapes of `y` and `dydx` must be identical.")
        if np.issubdtype(dydx.dtype, np.complexfloating):
            dtype = complex
        dydx = dydx.astype(dtype, copy=False)

    y = y.astype(dtype, copy=False)
    axis = axis % y.ndim
    if x.ndim != 1:
        raise ValueError("`x` must be 1-dimensional.")
    if x.shape[0] < 2:
        raise ValueError("`x` must contain at least 2 elements.")
    if x.shape[0] != y.shape[axis]:
        raise ValueError("The length of `y` along `axis`={0} doesn't "
                         "match the length of `x`".format(axis))

    if not np.all(np.isfinite(x)):
        raise ValueError("`x` must contain only finite values.")
    if not np.all(np.isfinite(y)):
        raise ValueError("`y` must contain only finite values.")

    if dydx is not None and not np.all(np.isfinite(dydx)):
        raise ValueError("`dydx` must contain only finite values.")

    dx = np.diff(x)
    if np.any(dx <= 0):
        raise ValueError("`x` must be strictly increasing sequence.")

    y = np.rollaxis(y, axis)
    if dydx is not None:
        dydx = np.rollaxis(dydx, axis)

    return x, dx, y, axis, dydx

class CubicHermiteSpline(PPoly):
    def __init__(self, x, y, dydx, axis=0):

        x, dx, y, axis, dydx = prepare_input(x, y, axis, dydx)

        dxr = dx.reshape([dx.shape[0]] + [1] * (y.ndim - 1))
        slope = np.diff(y, axis=0) / dxr
        t = (dydx[:-1] + dydx[1:] - 2 * slope) / dxr

        c = np.empty((4, len(x) - 1) + y.shape[1:], dtype=t.dtype)
        c[0] = t / dxr
        c[1] = (slope - dydx[:-1]) / dxr - t
        c[2] = dydx[:-1]
        c[3] = y[:-1]

        super(CubicHermiteSpline, self).__init__(c, x,) # ????
        self.axis = axis



class CubicSpline(CubicHermiteSpline):
    def __init__(self, x, y, axis=0, bc_type='not-a-knot'):
        x, dx, y, axis, _ = prepare_input(x, y, axis)
        n = len(x)

        bc, y = self._validate_bc(bc_type, y, y.shape[1:], axis)

        dxr = dx.reshape([dx.shape[0]] + [1] * (y.ndim - 1))
        slope = np.diff(y, axis=0) / dxr


        # Find derivative values at each x[i] by solving a tridiagonal
        # system.
        A = np.zeros((3, n))  # This is a banded matrix representation.
        b = np.empty((n,) + y.shape[1:], dtype=y.dtype)

            # Filling the system for i=1..n-2
            #                         (x[i-1] - x[i]) * s[i-1] +\
            # 2 * ((x[i] - x[i-1]) + (x[i+1] - x[i])) * s[i]   +\
            #                         (x[i] - x[i-1]) * s[i+1] =\
            #       3 * ((x[i+1] - x[i])*(y[i] - y[i-1])/(x[i] - x[i-1]) +\
            #           (x[i] - x[i-1])*(y[i+1] - y[i])/(x[i+1] - x[i]))

        A[1, 1:-1] = 2 * (dx[:-1] + dx[1:])  # The diagonal
        A[0, 2:] = dx[:-1]                   # The upper diagonal
        A[-1, :-2] = dx[1:]                  # The lower diagonal

        b[1:-1] = 3 * (dxr[1:] * slope[:-1] + dxr[:-1] * slope[1:])

        bc_start, bc_end = bc

        if bc_start == 'periodic':
            # Due to the periodicity, and because y[-1] = y[0], the linear
            # system has (n-1) unknowns/equations instead of n:
            A = A[:, 0:-1]
            A[1, 0] = 2 * (dx[-1] + dx[0])
            A[0, 1] = dx[-1]

            b = b[:-1]

            # Also, due to the periodicity, the system is not tri-diagonal.
            # We need to compute a "condensed" matrix of shape (n-2, n-2).
            # See https://web.archive.org/web/20151220180652/http://www.cfm.brown.edu/people/gk/chap6/node14.html
            # for more explanations.
            # The condensed matrix is obtained by removing the last column
            # and last row of the (n-1, n-1) system matrix. The removed
            # values are saved in scalar variables with the (n-1, n-1)
            # system matrix indices forming their names:
            a_m1_0 = dx[-2]  # lower left corner value: A[-1, 0]
            a_m1_m2 = dx[-1]
            a_m1_m1 = 2 * (dx[-1] + dx[-2])
            a_m2_m1 = dx[-2]
            a_0_m1 = dx[0]

            b[0] = 3 * (dxr[0] * slope[-1] + dxr[-1] * slope[0])
            b[-1] = 3 * (dxr[-1] * slope[-2] + dxr[-2] * slope[-1])

            Ac = A[:, :-1]
            b1 = b[:-1]
            b2 = np.zeros_like(b1)
            b2[0] = -a_0_m1
            b2[-1] = -a_m2_m1

            # s1 and s2 are the solutions of (n-2, n-2) system
            s1 = solve_banded((1, 1), Ac, b1, overwrite_ab=False,
                                  overwrite_b=False, check_finite=False)

            s2 = solve_banded((1, 1), Ac, b2, overwrite_ab=False,
                                  overwrite_b=False, check_finite=False)

            # computing the s[n-2] solution:
            s_m1 = ((b[-1] - a_m1_0 * s1[0] - a_m1_m2 * s1[-1]) /
                        (a_m1_m1 + a_m1_0 * s2[0] + a_m1_m2 * s2[-1]))

                # s is the solution of the (n, n) system:
            s = np.empty((n,) + y.shape[1:], dtype=y.dtype)
            s[:-2] = s1 + s_m1 * s2
            s[-2] = s_m1
            s[-1] = s[0]
        else:
            if bc_start == 'not-a-knot':
                A[1, 0] = dx[1]
                A[0, 1] = x[2] - x[0]
                d = x[2] - x[0]
                b[0] = ((dxr[0] + 2*d) * dxr[1] * slope[0] +
                        dxr[0]**2 * slope[1]) / d
            elif bc_start[0] == 1:
                A[1, 0] = 1
                A[0, 1] = 0
                b[0] = bc_start[1]
            elif bc_start[0] == 2:
                A[1, 0] = 2 * dx[0]
                A[0, 1] = dx[0]
                b[0] = -0.5 * bc_start[1] * dx[0]**2 + 3 * (y[1] - y[0])

            if bc_end == 'not-a-knot':
                A[1, -1] = dx[-2]
                A[-1, -2] = x[-1] - x[-3]
                d = x[-1] - x[-3]
                b[-1] = ((dxr[-1]**2*slope[-2] +
                         (2*d + dxr[-1])*dxr[-2]*slope[-1]) / d)
            elif bc_end[0] == 1:
                A[1, -1] = 1
                A[-1, -2] = 0
                b[-1] = bc_end[1]
            elif bc_end[0] == 2:
                A[1, -1] = 2 * dx[-1]
                A[-1, -2] = dx[-1]
                b[-1] = 0.5 * bc_end[1] * dx[-1]**2 + 3 * (y[-1] - y[-2])

            s = solve_banded((1, 1), A, b, overwrite_ab=True,
                             overwrite_b=True, check_finite=False)

        super(CubicSpline, self).__init__(x, y, s, axis=0)
        self.axis = axis

    @staticmethod
    def _validate_bc(bc_type, y, expected_deriv_shape, axis):
        """Validate and prepare boundary conditions.
        Returns
        -------
        validated_bc : 2-tuple
            Boundary conditions for a curve start and end.
        y : ndarray
            y casted to complex dtype if one of the boundary conditions has
            complex dtype.
        """
        if isinstance(bc_type, str):
            if bc_type == 'periodic':
                if not np.allclose(y[0], y[-1], rtol=1e-15, atol=1e-15):
                    raise ValueError(
                        "The first and last `y` point along axis {} must "
                        "be identical (within machine precision) when "
                        "bc_type='periodic'.".format(axis))

            bc_type = (bc_type, bc_type)

        else:
            if len(bc_type) != 2:
                raise ValueError("`bc_type` must contain 2 elements to "
                                 "specify start and end conditions.")

            if 'periodic' in bc_type:
                raise ValueError("'periodic' `bc_type` is defined for both "
                                 "curve ends and cannot be used with other "
                                 "boundary conditions.")

        validated_bc = []
        for bc in bc_type:
            if isinstance(bc, str):
                if bc == 'clamped':
                    validated_bc.append((1, np.zeros(expected_deriv_shape)))
                elif bc == 'natural':
                    validated_bc.append((2, np.zeros(expected_deriv_shape)))
                elif bc in ['not-a-knot', 'periodic']:
                    validated_bc.append(bc)
                else:
                    raise ValueError("bc_type={} is not allowed.".format(bc))
            else:
                try:
                    deriv_order, deriv_value = bc
                except Exception:
                    raise ValueError("A specified derivative value must be "
                                     "given in the form (order, value).")

                if deriv_order not in [1, 2]:
                    raise ValueError("The specified derivative order must "
                                     "be 1 or 2.")

                deriv_value = np.asarray(deriv_value)
                if deriv_value.shape != expected_deriv_shape:
                    raise ValueError(
                        "`deriv_value` shape {} is not the expected one {}."
                        .format(deriv_value.shape, expected_deriv_shape))

                if np.issubdtype(deriv_value.dtype, np.complexfloating):
                    y = y.astype(complex, copy=False)

                validated_bc.append((deriv_order, deriv_value))

        return validated_bc, y

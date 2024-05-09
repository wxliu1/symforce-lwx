# -----------------------------------------------------------------------------
# This file was autogenerated by symforce from template:
#     ops/CLASS/group_ops.py.jinja
# Do NOT modify by hand.
# -----------------------------------------------------------------------------

import math
import typing as T

import numpy

import sym  # pylint: disable=useless-suppression,unused-import


class GroupOps(object):
    """
    Python GroupOps implementation for :py:class:`symforce.cam.spherical_camera_cal.SphericalCameraCal`.
    """

    @staticmethod
    def identity():
        # type: () -> sym.SphericalCameraCal

        # Total ops: 0

        # Input arrays

        # Intermediate terms (0)

        # Output terms
        _res = [0.0] * 9
        _res[0] = 0
        _res[1] = 0
        _res[2] = 0
        _res[3] = 0
        _res[4] = 0
        _res[5] = 0
        _res[6] = 0
        _res[7] = 0
        _res[8] = 0
        return sym.SphericalCameraCal.from_storage(_res)

    @staticmethod
    def inverse(a):
        # type: (sym.SphericalCameraCal) -> sym.SphericalCameraCal

        # Total ops: 9

        # Input arrays
        _a = a.data

        # Intermediate terms (0)

        # Output terms
        _res = [0.0] * 9
        _res[0] = -_a[0]
        _res[1] = -_a[1]
        _res[2] = -_a[2]
        _res[3] = -_a[3]
        _res[4] = -_a[4]
        _res[5] = -_a[5]
        _res[6] = -_a[6]
        _res[7] = -_a[7]
        _res[8] = -_a[8]
        return sym.SphericalCameraCal.from_storage(_res)

    @staticmethod
    def compose(a, b):
        # type: (sym.SphericalCameraCal, sym.SphericalCameraCal) -> sym.SphericalCameraCal

        # Total ops: 9

        # Input arrays
        _a = a.data
        _b = b.data

        # Intermediate terms (0)

        # Output terms
        _res = [0.0] * 9
        _res[0] = _a[0] + _b[0]
        _res[1] = _a[1] + _b[1]
        _res[2] = _a[2] + _b[2]
        _res[3] = _a[3] + _b[3]
        _res[4] = _a[4] + _b[4]
        _res[5] = _a[5] + _b[5]
        _res[6] = _a[6] + _b[6]
        _res[7] = _a[7] + _b[7]
        _res[8] = _a[8] + _b[8]
        return sym.SphericalCameraCal.from_storage(_res)

    @staticmethod
    def between(a, b):
        # type: (sym.SphericalCameraCal, sym.SphericalCameraCal) -> sym.SphericalCameraCal

        # Total ops: 9

        # Input arrays
        _a = a.data
        _b = b.data

        # Intermediate terms (0)

        # Output terms
        _res = [0.0] * 9
        _res[0] = -_a[0] + _b[0]
        _res[1] = -_a[1] + _b[1]
        _res[2] = -_a[2] + _b[2]
        _res[3] = -_a[3] + _b[3]
        _res[4] = -_a[4] + _b[4]
        _res[5] = -_a[5] + _b[5]
        _res[6] = -_a[6] + _b[6]
        _res[7] = -_a[7] + _b[7]
        _res[8] = -_a[8] + _b[8]
        return sym.SphericalCameraCal.from_storage(_res)

    @staticmethod
    def inverse_with_jacobian(a):
        # type: (sym.SphericalCameraCal) -> T.Tuple[sym.SphericalCameraCal, numpy.ndarray]

        # Total ops: 9

        # Input arrays
        _a = a.data

        # Intermediate terms (0)

        # Output terms
        _res = [0.0] * 9
        _res[0] = -_a[0]
        _res[1] = -_a[1]
        _res[2] = -_a[2]
        _res[3] = -_a[3]
        _res[4] = -_a[4]
        _res[5] = -_a[5]
        _res[6] = -_a[6]
        _res[7] = -_a[7]
        _res[8] = -_a[8]
        _res_D_a = numpy.zeros((9, 9))
        _res_D_a[0, 0] = -1
        _res_D_a[1, 0] = 0
        _res_D_a[2, 0] = 0
        _res_D_a[3, 0] = 0
        _res_D_a[4, 0] = 0
        _res_D_a[5, 0] = 0
        _res_D_a[6, 0] = 0
        _res_D_a[7, 0] = 0
        _res_D_a[8, 0] = 0
        _res_D_a[0, 1] = 0
        _res_D_a[1, 1] = -1
        _res_D_a[2, 1] = 0
        _res_D_a[3, 1] = 0
        _res_D_a[4, 1] = 0
        _res_D_a[5, 1] = 0
        _res_D_a[6, 1] = 0
        _res_D_a[7, 1] = 0
        _res_D_a[8, 1] = 0
        _res_D_a[0, 2] = 0
        _res_D_a[1, 2] = 0
        _res_D_a[2, 2] = -1
        _res_D_a[3, 2] = 0
        _res_D_a[4, 2] = 0
        _res_D_a[5, 2] = 0
        _res_D_a[6, 2] = 0
        _res_D_a[7, 2] = 0
        _res_D_a[8, 2] = 0
        _res_D_a[0, 3] = 0
        _res_D_a[1, 3] = 0
        _res_D_a[2, 3] = 0
        _res_D_a[3, 3] = -1
        _res_D_a[4, 3] = 0
        _res_D_a[5, 3] = 0
        _res_D_a[6, 3] = 0
        _res_D_a[7, 3] = 0
        _res_D_a[8, 3] = 0
        _res_D_a[0, 4] = 0
        _res_D_a[1, 4] = 0
        _res_D_a[2, 4] = 0
        _res_D_a[3, 4] = 0
        _res_D_a[4, 4] = -1
        _res_D_a[5, 4] = 0
        _res_D_a[6, 4] = 0
        _res_D_a[7, 4] = 0
        _res_D_a[8, 4] = 0
        _res_D_a[0, 5] = 0
        _res_D_a[1, 5] = 0
        _res_D_a[2, 5] = 0
        _res_D_a[3, 5] = 0
        _res_D_a[4, 5] = 0
        _res_D_a[5, 5] = -1
        _res_D_a[6, 5] = 0
        _res_D_a[7, 5] = 0
        _res_D_a[8, 5] = 0
        _res_D_a[0, 6] = 0
        _res_D_a[1, 6] = 0
        _res_D_a[2, 6] = 0
        _res_D_a[3, 6] = 0
        _res_D_a[4, 6] = 0
        _res_D_a[5, 6] = 0
        _res_D_a[6, 6] = -1
        _res_D_a[7, 6] = 0
        _res_D_a[8, 6] = 0
        _res_D_a[0, 7] = 0
        _res_D_a[1, 7] = 0
        _res_D_a[2, 7] = 0
        _res_D_a[3, 7] = 0
        _res_D_a[4, 7] = 0
        _res_D_a[5, 7] = 0
        _res_D_a[6, 7] = 0
        _res_D_a[7, 7] = -1
        _res_D_a[8, 7] = 0
        _res_D_a[0, 8] = 0
        _res_D_a[1, 8] = 0
        _res_D_a[2, 8] = 0
        _res_D_a[3, 8] = 0
        _res_D_a[4, 8] = 0
        _res_D_a[5, 8] = 0
        _res_D_a[6, 8] = 0
        _res_D_a[7, 8] = 0
        _res_D_a[8, 8] = -1
        return sym.SphericalCameraCal.from_storage(_res), _res_D_a

    @staticmethod
    def compose_with_jacobians(a, b):
        # type: (sym.SphericalCameraCal, sym.SphericalCameraCal) -> T.Tuple[sym.SphericalCameraCal, numpy.ndarray, numpy.ndarray]

        # Total ops: 9

        # Input arrays
        _a = a.data
        _b = b.data

        # Intermediate terms (0)

        # Output terms
        _res = [0.0] * 9
        _res[0] = _a[0] + _b[0]
        _res[1] = _a[1] + _b[1]
        _res[2] = _a[2] + _b[2]
        _res[3] = _a[3] + _b[3]
        _res[4] = _a[4] + _b[4]
        _res[5] = _a[5] + _b[5]
        _res[6] = _a[6] + _b[6]
        _res[7] = _a[7] + _b[7]
        _res[8] = _a[8] + _b[8]
        _res_D_a = numpy.zeros((9, 9))
        _res_D_a[0, 0] = 1
        _res_D_a[1, 0] = 0
        _res_D_a[2, 0] = 0
        _res_D_a[3, 0] = 0
        _res_D_a[4, 0] = 0
        _res_D_a[5, 0] = 0
        _res_D_a[6, 0] = 0
        _res_D_a[7, 0] = 0
        _res_D_a[8, 0] = 0
        _res_D_a[0, 1] = 0
        _res_D_a[1, 1] = 1
        _res_D_a[2, 1] = 0
        _res_D_a[3, 1] = 0
        _res_D_a[4, 1] = 0
        _res_D_a[5, 1] = 0
        _res_D_a[6, 1] = 0
        _res_D_a[7, 1] = 0
        _res_D_a[8, 1] = 0
        _res_D_a[0, 2] = 0
        _res_D_a[1, 2] = 0
        _res_D_a[2, 2] = 1
        _res_D_a[3, 2] = 0
        _res_D_a[4, 2] = 0
        _res_D_a[5, 2] = 0
        _res_D_a[6, 2] = 0
        _res_D_a[7, 2] = 0
        _res_D_a[8, 2] = 0
        _res_D_a[0, 3] = 0
        _res_D_a[1, 3] = 0
        _res_D_a[2, 3] = 0
        _res_D_a[3, 3] = 1
        _res_D_a[4, 3] = 0
        _res_D_a[5, 3] = 0
        _res_D_a[6, 3] = 0
        _res_D_a[7, 3] = 0
        _res_D_a[8, 3] = 0
        _res_D_a[0, 4] = 0
        _res_D_a[1, 4] = 0
        _res_D_a[2, 4] = 0
        _res_D_a[3, 4] = 0
        _res_D_a[4, 4] = 1
        _res_D_a[5, 4] = 0
        _res_D_a[6, 4] = 0
        _res_D_a[7, 4] = 0
        _res_D_a[8, 4] = 0
        _res_D_a[0, 5] = 0
        _res_D_a[1, 5] = 0
        _res_D_a[2, 5] = 0
        _res_D_a[3, 5] = 0
        _res_D_a[4, 5] = 0
        _res_D_a[5, 5] = 1
        _res_D_a[6, 5] = 0
        _res_D_a[7, 5] = 0
        _res_D_a[8, 5] = 0
        _res_D_a[0, 6] = 0
        _res_D_a[1, 6] = 0
        _res_D_a[2, 6] = 0
        _res_D_a[3, 6] = 0
        _res_D_a[4, 6] = 0
        _res_D_a[5, 6] = 0
        _res_D_a[6, 6] = 1
        _res_D_a[7, 6] = 0
        _res_D_a[8, 6] = 0
        _res_D_a[0, 7] = 0
        _res_D_a[1, 7] = 0
        _res_D_a[2, 7] = 0
        _res_D_a[3, 7] = 0
        _res_D_a[4, 7] = 0
        _res_D_a[5, 7] = 0
        _res_D_a[6, 7] = 0
        _res_D_a[7, 7] = 1
        _res_D_a[8, 7] = 0
        _res_D_a[0, 8] = 0
        _res_D_a[1, 8] = 0
        _res_D_a[2, 8] = 0
        _res_D_a[3, 8] = 0
        _res_D_a[4, 8] = 0
        _res_D_a[5, 8] = 0
        _res_D_a[6, 8] = 0
        _res_D_a[7, 8] = 0
        _res_D_a[8, 8] = 1
        _res_D_b = numpy.zeros((9, 9))
        _res_D_b[0, 0] = 1
        _res_D_b[1, 0] = 0
        _res_D_b[2, 0] = 0
        _res_D_b[3, 0] = 0
        _res_D_b[4, 0] = 0
        _res_D_b[5, 0] = 0
        _res_D_b[6, 0] = 0
        _res_D_b[7, 0] = 0
        _res_D_b[8, 0] = 0
        _res_D_b[0, 1] = 0
        _res_D_b[1, 1] = 1
        _res_D_b[2, 1] = 0
        _res_D_b[3, 1] = 0
        _res_D_b[4, 1] = 0
        _res_D_b[5, 1] = 0
        _res_D_b[6, 1] = 0
        _res_D_b[7, 1] = 0
        _res_D_b[8, 1] = 0
        _res_D_b[0, 2] = 0
        _res_D_b[1, 2] = 0
        _res_D_b[2, 2] = 1
        _res_D_b[3, 2] = 0
        _res_D_b[4, 2] = 0
        _res_D_b[5, 2] = 0
        _res_D_b[6, 2] = 0
        _res_D_b[7, 2] = 0
        _res_D_b[8, 2] = 0
        _res_D_b[0, 3] = 0
        _res_D_b[1, 3] = 0
        _res_D_b[2, 3] = 0
        _res_D_b[3, 3] = 1
        _res_D_b[4, 3] = 0
        _res_D_b[5, 3] = 0
        _res_D_b[6, 3] = 0
        _res_D_b[7, 3] = 0
        _res_D_b[8, 3] = 0
        _res_D_b[0, 4] = 0
        _res_D_b[1, 4] = 0
        _res_D_b[2, 4] = 0
        _res_D_b[3, 4] = 0
        _res_D_b[4, 4] = 1
        _res_D_b[5, 4] = 0
        _res_D_b[6, 4] = 0
        _res_D_b[7, 4] = 0
        _res_D_b[8, 4] = 0
        _res_D_b[0, 5] = 0
        _res_D_b[1, 5] = 0
        _res_D_b[2, 5] = 0
        _res_D_b[3, 5] = 0
        _res_D_b[4, 5] = 0
        _res_D_b[5, 5] = 1
        _res_D_b[6, 5] = 0
        _res_D_b[7, 5] = 0
        _res_D_b[8, 5] = 0
        _res_D_b[0, 6] = 0
        _res_D_b[1, 6] = 0
        _res_D_b[2, 6] = 0
        _res_D_b[3, 6] = 0
        _res_D_b[4, 6] = 0
        _res_D_b[5, 6] = 0
        _res_D_b[6, 6] = 1
        _res_D_b[7, 6] = 0
        _res_D_b[8, 6] = 0
        _res_D_b[0, 7] = 0
        _res_D_b[1, 7] = 0
        _res_D_b[2, 7] = 0
        _res_D_b[3, 7] = 0
        _res_D_b[4, 7] = 0
        _res_D_b[5, 7] = 0
        _res_D_b[6, 7] = 0
        _res_D_b[7, 7] = 1
        _res_D_b[8, 7] = 0
        _res_D_b[0, 8] = 0
        _res_D_b[1, 8] = 0
        _res_D_b[2, 8] = 0
        _res_D_b[3, 8] = 0
        _res_D_b[4, 8] = 0
        _res_D_b[5, 8] = 0
        _res_D_b[6, 8] = 0
        _res_D_b[7, 8] = 0
        _res_D_b[8, 8] = 1
        return sym.SphericalCameraCal.from_storage(_res), _res_D_a, _res_D_b

    @staticmethod
    def between_with_jacobians(a, b):
        # type: (sym.SphericalCameraCal, sym.SphericalCameraCal) -> T.Tuple[sym.SphericalCameraCal, numpy.ndarray, numpy.ndarray]

        # Total ops: 9

        # Input arrays
        _a = a.data
        _b = b.data

        # Intermediate terms (0)

        # Output terms
        _res = [0.0] * 9
        _res[0] = -_a[0] + _b[0]
        _res[1] = -_a[1] + _b[1]
        _res[2] = -_a[2] + _b[2]
        _res[3] = -_a[3] + _b[3]
        _res[4] = -_a[4] + _b[4]
        _res[5] = -_a[5] + _b[5]
        _res[6] = -_a[6] + _b[6]
        _res[7] = -_a[7] + _b[7]
        _res[8] = -_a[8] + _b[8]
        _res_D_a = numpy.zeros((9, 9))
        _res_D_a[0, 0] = -1
        _res_D_a[1, 0] = 0
        _res_D_a[2, 0] = 0
        _res_D_a[3, 0] = 0
        _res_D_a[4, 0] = 0
        _res_D_a[5, 0] = 0
        _res_D_a[6, 0] = 0
        _res_D_a[7, 0] = 0
        _res_D_a[8, 0] = 0
        _res_D_a[0, 1] = 0
        _res_D_a[1, 1] = -1
        _res_D_a[2, 1] = 0
        _res_D_a[3, 1] = 0
        _res_D_a[4, 1] = 0
        _res_D_a[5, 1] = 0
        _res_D_a[6, 1] = 0
        _res_D_a[7, 1] = 0
        _res_D_a[8, 1] = 0
        _res_D_a[0, 2] = 0
        _res_D_a[1, 2] = 0
        _res_D_a[2, 2] = -1
        _res_D_a[3, 2] = 0
        _res_D_a[4, 2] = 0
        _res_D_a[5, 2] = 0
        _res_D_a[6, 2] = 0
        _res_D_a[7, 2] = 0
        _res_D_a[8, 2] = 0
        _res_D_a[0, 3] = 0
        _res_D_a[1, 3] = 0
        _res_D_a[2, 3] = 0
        _res_D_a[3, 3] = -1
        _res_D_a[4, 3] = 0
        _res_D_a[5, 3] = 0
        _res_D_a[6, 3] = 0
        _res_D_a[7, 3] = 0
        _res_D_a[8, 3] = 0
        _res_D_a[0, 4] = 0
        _res_D_a[1, 4] = 0
        _res_D_a[2, 4] = 0
        _res_D_a[3, 4] = 0
        _res_D_a[4, 4] = -1
        _res_D_a[5, 4] = 0
        _res_D_a[6, 4] = 0
        _res_D_a[7, 4] = 0
        _res_D_a[8, 4] = 0
        _res_D_a[0, 5] = 0
        _res_D_a[1, 5] = 0
        _res_D_a[2, 5] = 0
        _res_D_a[3, 5] = 0
        _res_D_a[4, 5] = 0
        _res_D_a[5, 5] = -1
        _res_D_a[6, 5] = 0
        _res_D_a[7, 5] = 0
        _res_D_a[8, 5] = 0
        _res_D_a[0, 6] = 0
        _res_D_a[1, 6] = 0
        _res_D_a[2, 6] = 0
        _res_D_a[3, 6] = 0
        _res_D_a[4, 6] = 0
        _res_D_a[5, 6] = 0
        _res_D_a[6, 6] = -1
        _res_D_a[7, 6] = 0
        _res_D_a[8, 6] = 0
        _res_D_a[0, 7] = 0
        _res_D_a[1, 7] = 0
        _res_D_a[2, 7] = 0
        _res_D_a[3, 7] = 0
        _res_D_a[4, 7] = 0
        _res_D_a[5, 7] = 0
        _res_D_a[6, 7] = 0
        _res_D_a[7, 7] = -1
        _res_D_a[8, 7] = 0
        _res_D_a[0, 8] = 0
        _res_D_a[1, 8] = 0
        _res_D_a[2, 8] = 0
        _res_D_a[3, 8] = 0
        _res_D_a[4, 8] = 0
        _res_D_a[5, 8] = 0
        _res_D_a[6, 8] = 0
        _res_D_a[7, 8] = 0
        _res_D_a[8, 8] = -1
        _res_D_b = numpy.zeros((9, 9))
        _res_D_b[0, 0] = 1
        _res_D_b[1, 0] = 0
        _res_D_b[2, 0] = 0
        _res_D_b[3, 0] = 0
        _res_D_b[4, 0] = 0
        _res_D_b[5, 0] = 0
        _res_D_b[6, 0] = 0
        _res_D_b[7, 0] = 0
        _res_D_b[8, 0] = 0
        _res_D_b[0, 1] = 0
        _res_D_b[1, 1] = 1
        _res_D_b[2, 1] = 0
        _res_D_b[3, 1] = 0
        _res_D_b[4, 1] = 0
        _res_D_b[5, 1] = 0
        _res_D_b[6, 1] = 0
        _res_D_b[7, 1] = 0
        _res_D_b[8, 1] = 0
        _res_D_b[0, 2] = 0
        _res_D_b[1, 2] = 0
        _res_D_b[2, 2] = 1
        _res_D_b[3, 2] = 0
        _res_D_b[4, 2] = 0
        _res_D_b[5, 2] = 0
        _res_D_b[6, 2] = 0
        _res_D_b[7, 2] = 0
        _res_D_b[8, 2] = 0
        _res_D_b[0, 3] = 0
        _res_D_b[1, 3] = 0
        _res_D_b[2, 3] = 0
        _res_D_b[3, 3] = 1
        _res_D_b[4, 3] = 0
        _res_D_b[5, 3] = 0
        _res_D_b[6, 3] = 0
        _res_D_b[7, 3] = 0
        _res_D_b[8, 3] = 0
        _res_D_b[0, 4] = 0
        _res_D_b[1, 4] = 0
        _res_D_b[2, 4] = 0
        _res_D_b[3, 4] = 0
        _res_D_b[4, 4] = 1
        _res_D_b[5, 4] = 0
        _res_D_b[6, 4] = 0
        _res_D_b[7, 4] = 0
        _res_D_b[8, 4] = 0
        _res_D_b[0, 5] = 0
        _res_D_b[1, 5] = 0
        _res_D_b[2, 5] = 0
        _res_D_b[3, 5] = 0
        _res_D_b[4, 5] = 0
        _res_D_b[5, 5] = 1
        _res_D_b[6, 5] = 0
        _res_D_b[7, 5] = 0
        _res_D_b[8, 5] = 0
        _res_D_b[0, 6] = 0
        _res_D_b[1, 6] = 0
        _res_D_b[2, 6] = 0
        _res_D_b[3, 6] = 0
        _res_D_b[4, 6] = 0
        _res_D_b[5, 6] = 0
        _res_D_b[6, 6] = 1
        _res_D_b[7, 6] = 0
        _res_D_b[8, 6] = 0
        _res_D_b[0, 7] = 0
        _res_D_b[1, 7] = 0
        _res_D_b[2, 7] = 0
        _res_D_b[3, 7] = 0
        _res_D_b[4, 7] = 0
        _res_D_b[5, 7] = 0
        _res_D_b[6, 7] = 0
        _res_D_b[7, 7] = 1
        _res_D_b[8, 7] = 0
        _res_D_b[0, 8] = 0
        _res_D_b[1, 8] = 0
        _res_D_b[2, 8] = 0
        _res_D_b[3, 8] = 0
        _res_D_b[4, 8] = 0
        _res_D_b[5, 8] = 0
        _res_D_b[6, 8] = 0
        _res_D_b[7, 8] = 0
        _res_D_b[8, 8] = 1
        return sym.SphericalCameraCal.from_storage(_res), _res_D_a, _res_D_b

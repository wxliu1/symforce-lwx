# -----------------------------------------------------------------------------
# This file was autogenerated by symforce from template:
#     ops/CLASS/lie_group_ops.py.jinja
# Do NOT modify by hand.
# -----------------------------------------------------------------------------

import math
import typing as T

import numpy

import sym  # pylint: disable=useless-suppression,unused-import


class LieGroupOps(object):
    """
    Python LieGroupOps implementation for :py:class:`symforce.cam.double_sphere_camera_cal.DoubleSphereCameraCal`.
    """

    @staticmethod
    def from_tangent(vec, epsilon):
        # type: (numpy.ndarray, float) -> sym.DoubleSphereCameraCal

        # Total ops: 0

        # Input arrays
        if vec.shape == (6,):
            vec = vec.reshape((6, 1))
        elif vec.shape != (6, 1):
            raise IndexError(
                "vec is expected to have shape (6, 1) or (6,); instead had shape {}".format(
                    vec.shape
                )
            )

        # Intermediate terms (0)

        # Output terms
        _res = [0.0] * 6
        _res[0] = vec[0, 0]
        _res[1] = vec[1, 0]
        _res[2] = vec[2, 0]
        _res[3] = vec[3, 0]
        _res[4] = vec[4, 0]
        _res[5] = vec[5, 0]
        return sym.DoubleSphereCameraCal.from_storage(_res)

    @staticmethod
    def to_tangent(a, epsilon):
        # type: (sym.DoubleSphereCameraCal, float) -> numpy.ndarray

        # Total ops: 0

        # Input arrays
        _a = a.data

        # Intermediate terms (0)

        # Output terms
        _res = numpy.zeros(6)
        _res[0] = _a[0]
        _res[1] = _a[1]
        _res[2] = _a[2]
        _res[3] = _a[3]
        _res[4] = _a[4]
        _res[5] = _a[5]
        return _res

    @staticmethod
    def retract(a, vec, epsilon):
        # type: (sym.DoubleSphereCameraCal, numpy.ndarray, float) -> sym.DoubleSphereCameraCal

        # Total ops: 6

        # Input arrays
        _a = a.data
        if vec.shape == (6,):
            vec = vec.reshape((6, 1))
        elif vec.shape != (6, 1):
            raise IndexError(
                "vec is expected to have shape (6, 1) or (6,); instead had shape {}".format(
                    vec.shape
                )
            )

        # Intermediate terms (0)

        # Output terms
        _res = [0.0] * 6
        _res[0] = _a[0] + vec[0, 0]
        _res[1] = _a[1] + vec[1, 0]
        _res[2] = _a[2] + vec[2, 0]
        _res[3] = _a[3] + vec[3, 0]
        _res[4] = _a[4] + vec[4, 0]
        _res[5] = _a[5] + vec[5, 0]
        return sym.DoubleSphereCameraCal.from_storage(_res)

    @staticmethod
    def local_coordinates(a, b, epsilon):
        # type: (sym.DoubleSphereCameraCal, sym.DoubleSphereCameraCal, float) -> numpy.ndarray

        # Total ops: 6

        # Input arrays
        _a = a.data
        _b = b.data

        # Intermediate terms (0)

        # Output terms
        _res = numpy.zeros(6)
        _res[0] = -_a[0] + _b[0]
        _res[1] = -_a[1] + _b[1]
        _res[2] = -_a[2] + _b[2]
        _res[3] = -_a[3] + _b[3]
        _res[4] = -_a[4] + _b[4]
        _res[5] = -_a[5] + _b[5]
        return _res

    @staticmethod
    def interpolate(a, b, alpha, epsilon):
        # type: (sym.DoubleSphereCameraCal, sym.DoubleSphereCameraCal, float, float) -> sym.DoubleSphereCameraCal

        # Total ops: 18

        # Input arrays
        _a = a.data
        _b = b.data

        # Intermediate terms (0)

        # Output terms
        _res = [0.0] * 6
        _res[0] = _a[0] + alpha * (-_a[0] + _b[0])
        _res[1] = _a[1] + alpha * (-_a[1] + _b[1])
        _res[2] = _a[2] + alpha * (-_a[2] + _b[2])
        _res[3] = _a[3] + alpha * (-_a[3] + _b[3])
        _res[4] = _a[4] + alpha * (-_a[4] + _b[4])
        _res[5] = _a[5] + alpha * (-_a[5] + _b[5])
        return sym.DoubleSphereCameraCal.from_storage(_res)
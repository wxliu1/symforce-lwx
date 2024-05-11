# -----------------------------------------------------------------------------
# This file was autogenerated by symforce from template:
#     cam_package/ops/CLASS/camera_ops.py.jinja
# Do NOT modify by hand.
# -----------------------------------------------------------------------------

import math
import typing as T

import numpy

import sym  # pylint: disable=useless-suppression,unused-import


class CameraOps(object):
    """
    Python CameraOps implementation for :py:class:`symforce.cam.double_sphere_camera_cal.DoubleSphereCameraCal`.
    """

    @staticmethod
    def focal_length(self):
        # type: (sym.DoubleSphereCameraCal) -> numpy.ndarray
        """
        Return the focal length.
        """

        # Total ops: 0

        # Input arrays
        _self = self.data

        # Intermediate terms (0)

        # Output terms
        _focal_length = numpy.zeros(2)
        _focal_length[0] = _self[0]
        _focal_length[1] = _self[1]
        return _focal_length

    @staticmethod
    def principal_point(self):
        # type: (sym.DoubleSphereCameraCal) -> numpy.ndarray
        """
        Return the principal point.
        """

        # Total ops: 0

        # Input arrays
        _self = self.data

        # Intermediate terms (0)

        # Output terms
        _principal_point = numpy.zeros(2)
        _principal_point[0] = _self[2]
        _principal_point[1] = _self[3]
        return _principal_point

    @staticmethod
    def pixel_from_camera_point(self, point, epsilon):
        # type: (sym.DoubleSphereCameraCal, numpy.ndarray, float) -> T.Tuple[numpy.ndarray, float]
        """
        Project a 3D point in the camera frame into 2D pixel coordinates.

        Returns:
            pixel: (x, y) coordinate in pixels if valid
            is_valid: 1 if the operation is within bounds else 0
        """

        # Total ops: 73

        # Input arrays
        _self = self.data
        if point.shape == (3,):
            point = point.reshape((3, 1))
        elif point.shape != (3, 1):
            raise IndexError(
                "point is expected to have shape (3, 1) or (3,); instead had shape {}".format(
                    point.shape
                )
            )

        # Intermediate terms (13)
        _tmp0 = epsilon**2 + point[0, 0] ** 2 + point[1, 0] ** 2
        _tmp1 = math.sqrt(_tmp0 + point[2, 0] ** 2)
        _tmp2 = _self[4] * _tmp1 + point[2, 0]
        _tmp3 = min(0, (0.0 if _self[5] - 0.5 == 0 else math.copysign(1, _self[5] - 0.5)))
        _tmp4 = 2 * _tmp3
        _tmp5 = _self[5] - epsilon * (_tmp4 + 1)
        _tmp6 = -_tmp5
        _tmp7 = 1 / max(epsilon, _tmp2 * (_tmp6 + 1) + _tmp5 * math.sqrt(_tmp0 + _tmp2**2))
        _tmp8 = _tmp3 + _tmp5
        _tmp9 = (1.0 / 2.0) * _tmp4 + _tmp6 + 1
        _tmp10 = _self[4] ** 2
        _tmp11 = _tmp9**2 / _tmp8**2
        _tmp12 = _tmp10 * _tmp11 - _tmp10 + 1

        # Output terms
        _pixel = numpy.zeros(2)
        _pixel[0] = _self[0] * _tmp7 * point[0, 0] + _self[2]
        _pixel[1] = _self[1] * _tmp7 * point[1, 0] + _self[3]
        _is_valid = max(
            0,
            min(
                max(
                    -(0.0 if _self[4] - 1 == 0 else math.copysign(1, _self[4] - 1)),
                    1
                    - max(
                        0,
                        -(
                            0.0
                            if _self[4] * point[2, 0] + _tmp1 == 0
                            else math.copysign(1, _self[4] * point[2, 0] + _tmp1)
                        ),
                    ),
                ),
                max(
                    -(0.0 if _tmp12 == 0 else math.copysign(1, _tmp12)),
                    1
                    - max(
                        0,
                        -(
                            0.0
                            if -_tmp1
                            * (
                                _self[4] * _tmp11
                                - _self[4]
                                - _tmp9 * math.sqrt(max(_tmp12, math.sqrt(epsilon))) / _tmp8
                            )
                            + point[2, 0]
                            == 0
                            else math.copysign(
                                1,
                                -_tmp1
                                * (
                                    _self[4] * _tmp11
                                    - _self[4]
                                    - _tmp9 * math.sqrt(max(_tmp12, math.sqrt(epsilon))) / _tmp8
                                )
                                + point[2, 0],
                            )
                        ),
                    ),
                ),
            ),
        )
        return _pixel, _is_valid

    @staticmethod
    def pixel_from_camera_point_with_jacobians(self, point, epsilon):
        # type: (sym.DoubleSphereCameraCal, numpy.ndarray, float) -> T.Tuple[numpy.ndarray, float, numpy.ndarray, numpy.ndarray]
        """
        Project a 3D point in the camera frame into 2D pixel coordinates.

        Returns:
            pixel: (x, y) coordinate in pixels if valid
            is_valid: 1 if the operation is within bounds else 0
            pixel_D_cal: Derivative of pixel with respect to intrinsic calibration parameters
            pixel_D_point: Derivative of pixel with respect to point
        """

        # Total ops: 136

        # Input arrays
        _self = self.data
        if point.shape == (3,):
            point = point.reshape((3, 1))
        elif point.shape != (3, 1):
            raise IndexError(
                "point is expected to have shape (3, 1) or (3,); instead had shape {}".format(
                    point.shape
                )
            )

        # Intermediate terms (40)
        _tmp0 = epsilon**2 + point[0, 0] ** 2 + point[1, 0] ** 2
        _tmp1 = math.sqrt(_tmp0 + point[2, 0] ** 2)
        _tmp2 = _self[4] * _tmp1 + point[2, 0]
        _tmp3 = math.sqrt(_tmp0 + _tmp2**2)
        _tmp4 = min(0, (0.0 if _self[5] - 0.5 == 0 else math.copysign(1, _self[5] - 0.5)))
        _tmp5 = 2 * _tmp4
        _tmp6 = _self[5] - epsilon * (_tmp5 + 1)
        _tmp7 = -_tmp6
        _tmp8 = _tmp7 + 1
        _tmp9 = _tmp2 * _tmp8 + _tmp3 * _tmp6
        _tmp10 = max(_tmp9, epsilon)
        _tmp11 = 1 / _tmp10
        _tmp12 = _tmp11 * point[0, 0]
        _tmp13 = _tmp11 * point[1, 0]
        _tmp14 = _self[4] * point[2, 0]
        _tmp15 = _tmp4 + _tmp6
        _tmp16 = (1.0 / 2.0) * _tmp5 + _tmp7 + 1
        _tmp17 = _self[4] ** 2
        _tmp18 = _tmp16**2 / _tmp15**2
        _tmp19 = _tmp17 * _tmp18 - _tmp17 + 1
        _tmp20 = _self[0] * point[0, 0]
        _tmp21 = _tmp6 / _tmp3
        _tmp22 = _tmp2 * _tmp21
        _tmp23 = (
            (1.0 / 2.0)
            * ((0.0 if _tmp9 - epsilon == 0 else math.copysign(1, _tmp9 - epsilon)) + 1)
            / _tmp10**2
        )
        _tmp24 = _tmp23 * (_tmp1 * _tmp22 + _tmp1 * _tmp8)
        _tmp25 = _self[1] * point[1, 0]
        _tmp26 = -_tmp2 + _tmp3
        _tmp27 = _tmp20 * _tmp23
        _tmp28 = _tmp23 * _tmp25
        _tmp29 = 1 / _tmp1
        _tmp30 = _self[4] * _tmp29
        _tmp31 = _tmp30 * _tmp8
        _tmp32 = 2 * point[0, 0]
        _tmp33 = _tmp2 * _tmp30
        _tmp34 = (1.0 / 2.0) * _tmp21
        _tmp35 = _tmp23 * (_tmp31 * point[0, 0] + _tmp34 * (_tmp32 * _tmp33 + _tmp32))
        _tmp36 = 2 * point[1, 0]
        _tmp37 = _tmp23 * (_tmp31 * point[1, 0] + _tmp34 * (_tmp33 * _tmp36 + _tmp36))
        _tmp38 = _tmp14 * _tmp29 + 1
        _tmp39 = _tmp22 * _tmp38 + _tmp38 * _tmp8

        # Output terms
        _pixel = numpy.zeros(2)
        _pixel[0] = _self[0] * _tmp12 + _self[2]
        _pixel[1] = _self[1] * _tmp13 + _self[3]
        _is_valid = max(
            0,
            min(
                max(
                    -(0.0 if _self[4] - 1 == 0 else math.copysign(1, _self[4] - 1)),
                    1 - max(0, -(0.0 if _tmp1 + _tmp14 == 0 else math.copysign(1, _tmp1 + _tmp14))),
                ),
                max(
                    -(0.0 if _tmp19 == 0 else math.copysign(1, _tmp19)),
                    1
                    - max(
                        0,
                        -(
                            0.0
                            if -_tmp1
                            * (
                                _self[4] * _tmp18
                                - _self[4]
                                - _tmp16 * math.sqrt(max(_tmp19, math.sqrt(epsilon))) / _tmp15
                            )
                            + point[2, 0]
                            == 0
                            else math.copysign(
                                1,
                                -_tmp1
                                * (
                                    _self[4] * _tmp18
                                    - _self[4]
                                    - _tmp16 * math.sqrt(max(_tmp19, math.sqrt(epsilon))) / _tmp15
                                )
                                + point[2, 0],
                            )
                        ),
                    ),
                ),
            ),
        )
        _pixel_D_cal = numpy.zeros((2, 6))
        _pixel_D_cal[0, 0] = _tmp12
        _pixel_D_cal[1, 0] = 0
        _pixel_D_cal[0, 1] = 0
        _pixel_D_cal[1, 1] = _tmp13
        _pixel_D_cal[0, 2] = 1
        _pixel_D_cal[1, 2] = 0
        _pixel_D_cal[0, 3] = 0
        _pixel_D_cal[1, 3] = 1
        _pixel_D_cal[0, 4] = -_tmp20 * _tmp24
        _pixel_D_cal[1, 4] = -_tmp24 * _tmp25
        _pixel_D_cal[0, 5] = -_tmp26 * _tmp27
        _pixel_D_cal[1, 5] = -_tmp26 * _tmp28
        _pixel_D_point = numpy.zeros((2, 3))
        _pixel_D_point[0, 0] = _self[0] * _tmp11 - _tmp20 * _tmp35
        _pixel_D_point[1, 0] = -_tmp25 * _tmp35
        _pixel_D_point[0, 1] = -_tmp20 * _tmp37
        _pixel_D_point[1, 1] = _self[1] * _tmp11 - _tmp25 * _tmp37
        _pixel_D_point[0, 2] = -_tmp27 * _tmp39
        _pixel_D_point[1, 2] = -_tmp28 * _tmp39
        return _pixel, _is_valid, _pixel_D_cal, _pixel_D_point

    @staticmethod
    def camera_ray_from_pixel(self, pixel, epsilon):
        # type: (sym.DoubleSphereCameraCal, numpy.ndarray, float) -> T.Tuple[numpy.ndarray, float]
        """
        Backproject a 2D pixel coordinate into a 3D ray in the camera frame.

        Returns:
            camera_ray: The ray in the camera frame (NOT normalized)
            is_valid: 1 if the operation is within bounds else 0
        """

        # Total ops: 62

        # Input arrays
        _self = self.data
        if pixel.shape == (2,):
            pixel = pixel.reshape((2, 1))
        elif pixel.shape != (2, 1):
            raise IndexError(
                "pixel is expected to have shape (2, 1) or (2,); instead had shape {}".format(
                    pixel.shape
                )
            )

        # Intermediate terms (12)
        _tmp0 = -_self[2] + pixel[0, 0]
        _tmp1 = -_self[3] + pixel[1, 0]
        _tmp2 = _tmp1**2 / _self[1] ** 2 + _tmp0**2 / _self[0] ** 2
        _tmp3 = -(_self[5] ** 2) * _tmp2 + 1
        _tmp4 = -_tmp2 * (2 * _self[5] - 1) + 1
        _tmp5 = _self[5] * math.sqrt(max(_tmp4, epsilon)) - _self[5] + 1
        _tmp6 = _tmp5 + epsilon * (2 * min(0, (0.0 if _tmp5 == 0 else math.copysign(1, _tmp5))) + 1)
        _tmp7 = _tmp3**2 / _tmp6**2
        _tmp8 = _tmp2 + _tmp7
        _tmp9 = _tmp3 / _tmp6
        _tmp10 = _tmp2 * (1 - _self[4] ** 2) + _tmp7
        _tmp11 = (_self[4] * _tmp9 + math.sqrt(max(_tmp10, epsilon))) / (
            _tmp8 + epsilon * (2 * min(0, (0.0 if _tmp8 == 0 else math.copysign(1, _tmp8))) + 1)
        )

        # Output terms
        _camera_ray = numpy.zeros(3)
        _camera_ray[0] = _tmp0 * _tmp11 / _self[0]
        _camera_ray[1] = _tmp1 * _tmp11 / _self[1]
        _camera_ray[2] = -_self[4] + _tmp11 * _tmp9
        _is_valid = min(
            1 - max(0, -(0.0 if _tmp10 == 0 else math.copysign(1, _tmp10))),
            1 - max(0, -(0.0 if _tmp4 == 0 else math.copysign(1, _tmp4))),
        )
        return _camera_ray, _is_valid

    @staticmethod
    def camera_ray_from_pixel_with_jacobians(self, pixel, epsilon):
        # type: (sym.DoubleSphereCameraCal, numpy.ndarray, float) -> T.Tuple[numpy.ndarray, float, numpy.ndarray, numpy.ndarray]
        """
        Backproject a 2D pixel coordinate into a 3D ray in the camera frame.

        Returns:
            camera_ray: The ray in the camera frame (NOT normalized)
            is_valid: 1 if the operation is within bounds else 0
            point_D_cal: Derivative of point with respect to intrinsic calibration parameters
            point_D_pixel: Derivation of point with respect to pixel
        """

        # Total ops: 297

        # Input arrays
        _self = self.data
        if pixel.shape == (2,):
            pixel = pixel.reshape((2, 1))
        elif pixel.shape != (2, 1):
            raise IndexError(
                "pixel is expected to have shape (2, 1) or (2,); instead had shape {}".format(
                    pixel.shape
                )
            )

        # Intermediate terms (111)
        _tmp0 = -_self[2] + pixel[0, 0]
        _tmp1 = 1 / _self[0]
        _tmp2 = -_self[3] + pixel[1, 0]
        _tmp3 = _tmp2**2
        _tmp4 = _self[1] ** (-2)
        _tmp5 = _tmp0**2
        _tmp6 = _self[0] ** (-2)
        _tmp7 = _tmp3 * _tmp4 + _tmp5 * _tmp6
        _tmp8 = _self[5] ** 2
        _tmp9 = -_tmp7 * _tmp8 + 1
        _tmp10 = _tmp9**2
        _tmp11 = 2 * _self[5]
        _tmp12 = _tmp11 - 1
        _tmp13 = -_tmp12 * _tmp7 + 1
        _tmp14 = math.sqrt(max(_tmp13, epsilon))
        _tmp15 = _self[5] * _tmp14 - _self[5] + 1
        _tmp16 = _tmp15 + epsilon * (
            2 * min(0, (0.0 if _tmp15 == 0 else math.copysign(1, _tmp15))) + 1
        )
        _tmp17 = _tmp16 ** (-2)
        _tmp18 = _tmp10 * _tmp17
        _tmp19 = _tmp18 + _tmp7
        _tmp20 = _tmp19 + epsilon * (
            2 * min(0, (0.0 if _tmp19 == 0 else math.copysign(1, _tmp19))) + 1
        )
        _tmp21 = 1 / _tmp20
        _tmp22 = 1 / _tmp16
        _tmp23 = _tmp22 * _tmp9
        _tmp24 = 1 - _self[4] ** 2
        _tmp25 = _tmp18 + _tmp24 * _tmp7
        _tmp26 = math.sqrt(max(_tmp25, epsilon))
        _tmp27 = _self[4] * _tmp23 + _tmp26
        _tmp28 = _tmp21 * _tmp27
        _tmp29 = _tmp1 * _tmp28
        _tmp30 = 1 / _self[1]
        _tmp31 = _tmp28 * _tmp30
        _tmp32 = _tmp0 * _tmp6
        _tmp33 = _tmp28 * _tmp32
        _tmp34 = _tmp0 * _tmp1
        _tmp35 = _tmp5 / _self[0] ** 3
        _tmp36 = 2 * _tmp35
        _tmp37 = _tmp17 * _tmp9
        _tmp38 = 4 * _tmp37
        _tmp39 = _tmp38 * _tmp8
        _tmp40 = _tmp10 / _tmp16**3
        _tmp41 = -epsilon
        _tmp42 = (
            _self[5]
            * ((0.0 if _tmp13 + _tmp41 == 0 else math.copysign(1, _tmp13 + _tmp41)) + 1)
            / _tmp14
        )
        _tmp43 = _tmp12 * _tmp42
        _tmp44 = _tmp40 * _tmp43
        _tmp45 = _tmp35 * _tmp39 - _tmp35 * _tmp44
        _tmp46 = _tmp27 / _tmp20**2
        _tmp47 = _tmp46 * (-_tmp36 + _tmp45)
        _tmp48 = (1.0 / 2.0) * _tmp37 * _tmp43
        _tmp49 = _self[4] * _tmp48
        _tmp50 = _self[4] * _tmp22
        _tmp51 = _tmp50 * _tmp8
        _tmp52 = ((0.0 if _tmp25 + _tmp41 == 0 else math.copysign(1, _tmp25 + _tmp41)) + 1) / _tmp26
        _tmp53 = (1.0 / 4.0) * _tmp52
        _tmp54 = -_tmp35 * _tmp49 + _tmp36 * _tmp51 + _tmp53 * (-_tmp24 * _tmp36 + _tmp45)
        _tmp55 = _tmp21 * _tmp54
        _tmp56 = _tmp2 * _tmp30
        _tmp57 = _tmp22 * _tmp28
        _tmp58 = _tmp57 * _tmp8
        _tmp59 = _tmp28 * _tmp48
        _tmp60 = _tmp21 * _tmp23
        _tmp61 = _tmp3 / _self[1] ** 3
        _tmp62 = 2 * _tmp61
        _tmp63 = _tmp39 * _tmp61 - _tmp44 * _tmp61
        _tmp64 = _tmp46 * (-_tmp62 + _tmp63)
        _tmp65 = -_tmp49 * _tmp61 + _tmp51 * _tmp62 + _tmp53 * (-_tmp24 * _tmp62 + _tmp63)
        _tmp66 = _tmp21 * _tmp65
        _tmp67 = _tmp2 * _tmp4
        _tmp68 = _tmp28 * _tmp67
        _tmp69 = _tmp32 * _tmp49
        _tmp70 = 2 * _tmp32
        _tmp71 = _tmp51 * _tmp70
        _tmp72 = _tmp24 * _tmp70
        _tmp73 = _tmp32 * _tmp39
        _tmp74 = _tmp32 * _tmp44
        _tmp75 = _tmp73 - _tmp74
        _tmp76 = _tmp53 * (-_tmp72 + _tmp75) - _tmp69 + _tmp71
        _tmp77 = _tmp21 * _tmp76
        _tmp78 = _tmp46 * (-_tmp70 + _tmp75)
        _tmp79 = _tmp58 * _tmp70
        _tmp80 = _tmp33 * _tmp48
        _tmp81 = 2 * _tmp67
        _tmp82 = _tmp51 * _tmp81
        _tmp83 = _tmp49 * _tmp67
        _tmp84 = _tmp24 * _tmp81
        _tmp85 = _tmp39 * _tmp67
        _tmp86 = _tmp44 * _tmp67
        _tmp87 = _tmp85 - _tmp86
        _tmp88 = _tmp53 * (-_tmp84 + _tmp87) + _tmp82 - _tmp83
        _tmp89 = _tmp21 * _tmp88
        _tmp90 = _tmp46 * (-_tmp81 + _tmp87)
        _tmp91 = _tmp48 * _tmp68
        _tmp92 = _tmp58 * _tmp81
        _tmp93 = (1.0 / 2.0) * _tmp7
        _tmp94 = -_self[4] * _tmp52 * _tmp93 + _tmp23
        _tmp95 = _tmp21 * _tmp94
        _tmp96 = _tmp14 - _tmp42 * _tmp93 - 1
        _tmp97 = -_self[5] * _tmp38 * _tmp7 - 2 * _tmp40 * _tmp96
        _tmp98 = _tmp46 * _tmp97
        _tmp99 = _tmp37 * _tmp96
        _tmp100 = _tmp11 * _tmp7
        _tmp101 = -_self[4] * _tmp99 - _tmp100 * _tmp50 + _tmp53 * _tmp97
        _tmp102 = _tmp101 * _tmp21
        _tmp103 = -_tmp73 + _tmp74
        _tmp104 = _tmp53 * (_tmp103 + _tmp72) + _tmp69 - _tmp71
        _tmp105 = _tmp104 * _tmp21
        _tmp106 = _tmp46 * (_tmp103 + _tmp70)
        _tmp107 = -_tmp85 + _tmp86
        _tmp108 = _tmp53 * (_tmp107 + _tmp84) - _tmp82 + _tmp83
        _tmp109 = _tmp108 * _tmp21
        _tmp110 = _tmp46 * (_tmp107 + _tmp81)

        # Output terms
        _camera_ray = numpy.zeros(3)
        _camera_ray[0] = _tmp0 * _tmp29
        _camera_ray[1] = _tmp2 * _tmp31
        _camera_ray[2] = -_self[4] + _tmp23 * _tmp28
        _is_valid = min(
            1 - max(0, -(0.0 if _tmp13 == 0 else math.copysign(1, _tmp13))),
            1 - max(0, -(0.0 if _tmp25 == 0 else math.copysign(1, _tmp25))),
        )
        _point_D_cal = numpy.zeros((3, 6))
        _point_D_cal[0, 0] = -_tmp33 - _tmp34 * _tmp47 + _tmp34 * _tmp55
        _point_D_cal[1, 0] = -_tmp47 * _tmp56 + _tmp55 * _tmp56
        _point_D_cal[2, 0] = -_tmp23 * _tmp47 - _tmp35 * _tmp59 + _tmp36 * _tmp58 + _tmp54 * _tmp60
        _point_D_cal[0, 1] = -_tmp34 * _tmp64 + _tmp34 * _tmp66
        _point_D_cal[1, 1] = -_tmp56 * _tmp64 + _tmp56 * _tmp66 - _tmp68
        _point_D_cal[2, 1] = -_tmp23 * _tmp64 + _tmp58 * _tmp62 - _tmp59 * _tmp61 + _tmp60 * _tmp65
        _point_D_cal[0, 2] = -_tmp29 + _tmp34 * _tmp77 - _tmp34 * _tmp78
        _point_D_cal[1, 2] = _tmp56 * _tmp77 - _tmp56 * _tmp78
        _point_D_cal[2, 2] = -_tmp23 * _tmp78 + _tmp60 * _tmp76 + _tmp79 - _tmp80
        _point_D_cal[0, 3] = _tmp34 * _tmp89 - _tmp34 * _tmp90
        _point_D_cal[1, 3] = -_tmp31 + _tmp56 * _tmp89 - _tmp56 * _tmp90
        _point_D_cal[2, 3] = -_tmp23 * _tmp90 + _tmp60 * _tmp88 - _tmp91 + _tmp92
        _point_D_cal[0, 4] = _tmp34 * _tmp95
        _point_D_cal[1, 4] = _tmp56 * _tmp95
        _point_D_cal[2, 4] = _tmp60 * _tmp94 - 1
        _point_D_cal[0, 5] = _tmp102 * _tmp34 - _tmp34 * _tmp98
        _point_D_cal[1, 5] = _tmp102 * _tmp56 - _tmp56 * _tmp98
        _point_D_cal[2, 5] = (
            -_tmp100 * _tmp57 + _tmp101 * _tmp60 - _tmp23 * _tmp98 - _tmp28 * _tmp99
        )
        _point_D_pixel = numpy.zeros((3, 2))
        _point_D_pixel[0, 0] = _tmp105 * _tmp34 - _tmp106 * _tmp34 + _tmp29
        _point_D_pixel[1, 0] = _tmp105 * _tmp56 - _tmp106 * _tmp56
        _point_D_pixel[2, 0] = _tmp104 * _tmp60 - _tmp106 * _tmp23 - _tmp79 + _tmp80
        _point_D_pixel[0, 1] = _tmp109 * _tmp34 - _tmp110 * _tmp34
        _point_D_pixel[1, 1] = _tmp109 * _tmp56 - _tmp110 * _tmp56 + _tmp31
        _point_D_pixel[2, 1] = _tmp108 * _tmp60 - _tmp110 * _tmp23 + _tmp91 - _tmp92
        return _camera_ray, _is_valid, _point_D_cal, _point_D_pixel
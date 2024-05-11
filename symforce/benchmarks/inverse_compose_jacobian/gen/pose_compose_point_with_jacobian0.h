// -----------------------------------------------------------------------------
// This file was autogenerated by symforce from template:
//     function/FUNCTION.h.jinja
// Do NOT modify by hand.
// -----------------------------------------------------------------------------

#pragma once

#include <Eigen/Core>

#include <sym/pose3.h>

namespace sym {

/**
 * This function was autogenerated from a symbolic function. Do not modify by hand.
 *
 * Symbolic function: pose_compose_point
 *
 * Args:
 *     pose: Pose3
 *     point: Matrix31
 *
 * Outputs:
 *     res: Matrix31
 *     res_D_pose: (3x6) jacobian of res (3) wrt arg pose (6)
 */
template <typename Scalar>
__attribute__((noinline)) Eigen::Matrix<Scalar, 3, 1> PoseComposePointWithJacobian0(
    const sym::Pose3<Scalar>& pose, const Eigen::Matrix<Scalar, 3, 1>& point,
    Eigen::Matrix<Scalar, 3, 6>* const res_D_pose = nullptr) {
  // Total ops: 85

  // Input arrays
  const Eigen::Matrix<Scalar, 7, 1>& _pose = pose.Data();

  // Intermediate terms (27)
  const Scalar _tmp0 = 2 * _pose[2];
  const Scalar _tmp1 = _pose[3] * _tmp0;
  const Scalar _tmp2 = 2 * _pose[1];
  const Scalar _tmp3 = _pose[0] * _tmp2;
  const Scalar _tmp4 = _tmp1 - _tmp3;
  const Scalar _tmp5 = _pose[3] * _tmp2;
  const Scalar _tmp6 = _pose[0] * _tmp0;
  const Scalar _tmp7 = _tmp5 + _tmp6;
  const Scalar _tmp8 = std::pow(_pose[1], Scalar(2));
  const Scalar _tmp9 = 2 * _tmp8;
  const Scalar _tmp10 = std::pow(_pose[2], Scalar(2));
  const Scalar _tmp11 = 2 * _tmp10 - 1;
  const Scalar _tmp12 = _tmp1 + _tmp3;
  const Scalar _tmp13 = 2 * _pose[0] * _pose[3];
  const Scalar _tmp14 = _pose[2] * _tmp2;
  const Scalar _tmp15 = _tmp13 - _tmp14;
  const Scalar _tmp16 = std::pow(_pose[0], Scalar(2));
  const Scalar _tmp17 = 2 * _tmp16;
  const Scalar _tmp18 = _tmp5 - _tmp6;
  const Scalar _tmp19 = _tmp13 + _tmp14;
  const Scalar _tmp20 = -_tmp10;
  const Scalar _tmp21 = std::pow(_pose[3], Scalar(2));
  const Scalar _tmp22 = -_tmp16 + _tmp21;
  const Scalar _tmp23 = _tmp20 + _tmp22 + _tmp8;
  const Scalar _tmp24 = -_tmp8;
  const Scalar _tmp25 = _tmp10 + _tmp22 + _tmp24;
  const Scalar _tmp26 = _tmp16 + _tmp20 + _tmp21 + _tmp24;

  // Output terms (2)
  Eigen::Matrix<Scalar, 3, 1> _res;

  _res(0, 0) =
      _pose[4] - _tmp4 * point(1, 0) + _tmp7 * point(2, 0) - point(0, 0) * (_tmp11 + _tmp9);
  _res(1, 0) =
      _pose[5] + _tmp12 * point(0, 0) - _tmp15 * point(2, 0) - point(1, 0) * (_tmp11 + _tmp17);
  _res(2, 0) =
      _pose[6] - _tmp18 * point(0, 0) + _tmp19 * point(1, 0) - point(2, 0) * (_tmp17 + _tmp9 - 1);

  if (res_D_pose != nullptr) {
    Eigen::Matrix<Scalar, 3, 6>& _res_D_pose = (*res_D_pose);

    _res_D_pose(0, 0) = _tmp4 * point(2, 0) + _tmp7 * point(1, 0);
    _res_D_pose(1, 0) = -_tmp15 * point(1, 0) - _tmp23 * point(2, 0);
    _res_D_pose(2, 0) = -_tmp19 * point(2, 0) + _tmp25 * point(1, 0);
    _res_D_pose(0, 1) = _tmp26 * point(2, 0) - _tmp7 * point(0, 0);
    _res_D_pose(1, 1) = _tmp12 * point(2, 0) + _tmp15 * point(0, 0);
    _res_D_pose(2, 1) = -_tmp18 * point(2, 0) - _tmp25 * point(0, 0);
    _res_D_pose(0, 2) = -_tmp26 * point(1, 0) - _tmp4 * point(0, 0);
    _res_D_pose(1, 2) = -_tmp12 * point(1, 0) + _tmp23 * point(0, 0);
    _res_D_pose(2, 2) = _tmp18 * point(1, 0) + _tmp19 * point(0, 0);
    _res_D_pose(0, 3) = 1;
    _res_D_pose(1, 3) = 0;
    _res_D_pose(2, 3) = 0;
    _res_D_pose(0, 4) = 0;
    _res_D_pose(1, 4) = 1;
    _res_D_pose(2, 4) = 0;
    _res_D_pose(0, 5) = 0;
    _res_D_pose(1, 5) = 0;
    _res_D_pose(2, 5) = 1;
  }

  return _res;
}  // NOLINT(readability/fn_size)

// NOLINTNEXTLINE(readability/fn_size)
}  // namespace sym
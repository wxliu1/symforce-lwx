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
 * Composition of two elements in the group.
 *
 * Returns:
 *     Element: a @ b
 *     res_D_a: (6x6) jacobian of res (6) wrt arg a (6)
 *     res_D_b: (6x6) jacobian of res (6) wrt arg b (6)
 */
template <typename Scalar>
sym::Pose3<Scalar> ComposePose3WithJacobians(const sym::Pose3<Scalar>& a,
                                             const sym::Pose3<Scalar>& b,
                                             Eigen::Matrix<Scalar, 6, 6>* const res_D_a = nullptr,
                                             Eigen::Matrix<Scalar, 6, 6>* const res_D_b = nullptr) {
  // Total ops: 331

  // Input arrays
  const Eigen::Matrix<Scalar, 7, 1>& _a = a.Data();
  const Eigen::Matrix<Scalar, 7, 1>& _b = b.Data();

  // Intermediate terms (102)
  const Scalar _tmp0 = _a[3] * _b[0];
  const Scalar _tmp1 = _a[0] * _b[3];
  const Scalar _tmp2 = _a[1] * _b[2];
  const Scalar _tmp3 = _a[2] * _b[1];
  const Scalar _tmp4 = _a[3] * _b[1];
  const Scalar _tmp5 = _a[1] * _b[3];
  const Scalar _tmp6 = _a[2] * _b[0];
  const Scalar _tmp7 = _a[0] * _b[2];
  const Scalar _tmp8 = _a[3] * _b[2];
  const Scalar _tmp9 = _a[0] * _b[1];
  const Scalar _tmp10 = _a[2] * _b[3];
  const Scalar _tmp11 = _a[1] * _b[0];
  const Scalar _tmp12 = _a[0] * _b[0];
  const Scalar _tmp13 = _a[1] * _b[1];
  const Scalar _tmp14 = _a[2] * _b[2];
  const Scalar _tmp15 = 2 * _a[3];
  const Scalar _tmp16 = _a[2] * _tmp15;
  const Scalar _tmp17 = 2 * _a[0];
  const Scalar _tmp18 = _a[1] * _tmp17;
  const Scalar _tmp19 = _tmp16 - _tmp18;
  const Scalar _tmp20 = -_tmp19;
  const Scalar _tmp21 = _a[1] * _tmp15;
  const Scalar _tmp22 = _a[2] * _tmp17;
  const Scalar _tmp23 = _tmp21 + _tmp22;
  const Scalar _tmp24 = std::pow(_a[1], Scalar(2));
  const Scalar _tmp25 = 2 * _tmp24;
  const Scalar _tmp26 = std::pow(_a[2], Scalar(2));
  const Scalar _tmp27 = 2 * _tmp26 - 1;
  const Scalar _tmp28 = -_tmp25 - _tmp27;
  const Scalar _tmp29 = _tmp16 + _tmp18;
  const Scalar _tmp30 = _a[0] * _tmp15;
  const Scalar _tmp31 = 2 * _a[1] * _a[2];
  const Scalar _tmp32 = _tmp30 - _tmp31;
  const Scalar _tmp33 = -_tmp32;
  const Scalar _tmp34 = std::pow(_a[0], Scalar(2));
  const Scalar _tmp35 = 2 * _tmp34;
  const Scalar _tmp36 = -_tmp27 - _tmp35;
  const Scalar _tmp37 = _tmp21 - _tmp22;
  const Scalar _tmp38 = -_tmp37;
  const Scalar _tmp39 = _tmp30 + _tmp31;
  const Scalar _tmp40 = -_tmp25 - _tmp35 + 1;
  const Scalar _tmp41 = (Scalar(1) / Scalar(2)) * _tmp13;
  const Scalar _tmp42 = (Scalar(1) / Scalar(2)) * _tmp12;
  const Scalar _tmp43 = (Scalar(1) / Scalar(2)) * _a[3] * _b[3];
  const Scalar _tmp44 = (Scalar(1) / Scalar(2)) * _tmp14;
  const Scalar _tmp45 = _tmp43 + _tmp44;
  const Scalar _tmp46 = _tmp41 - _tmp42 + _tmp45;
  const Scalar _tmp47 = 2 * _a[3] * _b[3] - 2 * _tmp12 - 2 * _tmp13 - 2 * _tmp14;
  const Scalar _tmp48 = -2 * _a[2] * _b[1] + 2 * _tmp0 + 2 * _tmp1 + 2 * _tmp2;
  const Scalar _tmp49 = -_tmp48;
  const Scalar _tmp50 = (Scalar(1) / Scalar(2)) * _tmp1;
  const Scalar _tmp51 = (Scalar(1) / Scalar(2)) * _tmp2;
  const Scalar _tmp52 = (Scalar(1) / Scalar(2)) * _tmp0;
  const Scalar _tmp53 = (Scalar(1) / Scalar(2)) * _tmp3;
  const Scalar _tmp54 = _tmp52 + _tmp53;
  const Scalar _tmp55 = -_tmp50 + _tmp51 - _tmp54;
  const Scalar _tmp56 = -2 * _a[0] * _b[2] + 2 * _tmp4 + 2 * _tmp5 + 2 * _tmp6;
  const Scalar _tmp57 = -_tmp56;
  const Scalar _tmp58 = (Scalar(1) / Scalar(2)) * _tmp6;
  const Scalar _tmp59 = (Scalar(1) / Scalar(2)) * _tmp4;
  const Scalar _tmp60 = (Scalar(1) / Scalar(2)) * _tmp7;
  const Scalar _tmp61 = (Scalar(1) / Scalar(2)) * _tmp5;
  const Scalar _tmp62 = _tmp60 + _tmp61;
  const Scalar _tmp63 = -_tmp58 + _tmp59 - _tmp62;
  const Scalar _tmp64 = (Scalar(1) / Scalar(2)) * _tmp8;
  const Scalar _tmp65 = (Scalar(1) / Scalar(2)) * _tmp10;
  const Scalar _tmp66 = (Scalar(1) / Scalar(2)) * _tmp9;
  const Scalar _tmp67 = (Scalar(1) / Scalar(2)) * _tmp11;
  const Scalar _tmp68 = _tmp66 + _tmp67;
  const Scalar _tmp69 = -_tmp64 + _tmp65 - _tmp68;
  const Scalar _tmp70 = 2 * _tmp10 - 2 * _tmp11 + 2 * _tmp8 + 2 * _tmp9;
  const Scalar _tmp71 = -_tmp70;
  const Scalar _tmp72 = -_tmp26;
  const Scalar _tmp73 = std::pow(_a[3], Scalar(2));
  const Scalar _tmp74 = -_tmp34 + _tmp73;
  const Scalar _tmp75 = _tmp24 + _tmp72 + _tmp74;
  const Scalar _tmp76 = -_tmp24;
  const Scalar _tmp77 = _tmp26 + _tmp74 + _tmp76;
  const Scalar _tmp78 = -_tmp41 + _tmp42 + _tmp45;
  const Scalar _tmp79 = _tmp64 - _tmp65 - _tmp68;
  const Scalar _tmp80 = _tmp58 - _tmp59 - _tmp62;
  const Scalar _tmp81 = _tmp50 - _tmp51 - _tmp54;
  const Scalar _tmp82 = _tmp34 + _tmp72 + _tmp73 + _tmp76;
  const Scalar _tmp83 = _tmp41 + _tmp42;
  const Scalar _tmp84 = _tmp43 - _tmp44 + _tmp83;
  const Scalar _tmp85 = _tmp58 + _tmp59;
  const Scalar _tmp86 = -_tmp60 + _tmp61 - _tmp85;
  const Scalar _tmp87 = _tmp64 + _tmp65;
  const Scalar _tmp88 = _tmp66 - _tmp67 - _tmp87;
  const Scalar _tmp89 = _tmp50 + _tmp51;
  const Scalar _tmp90 = _tmp52 - _tmp53 - _tmp89;
  const Scalar _tmp91 = _tmp66 - _tmp67 + _tmp87;
  const Scalar _tmp92 = _tmp52 - _tmp53 + _tmp89;
  const Scalar _tmp93 = -_tmp92;
  const Scalar _tmp94 = _tmp49 * _tmp93;
  const Scalar _tmp95 = (Scalar(1) / Scalar(2)) * _a[3] * _b[3] - _tmp44 - _tmp83;
  const Scalar _tmp96 = _tmp47 * _tmp95;
  const Scalar _tmp97 = -_tmp60 + _tmp61 + _tmp85;
  const Scalar _tmp98 = -_tmp97;
  const Scalar _tmp99 = _tmp57 * _tmp98 + _tmp96;
  const Scalar _tmp100 = -_tmp91;
  const Scalar _tmp101 = _tmp100 * _tmp71;

  // Output terms (3)
  Eigen::Matrix<Scalar, 7, 1> _res;

  _res[0] = _tmp0 + _tmp1 + _tmp2 - _tmp3;
  _res[1] = _tmp4 + _tmp5 + _tmp6 - _tmp7;
  _res[2] = _tmp10 - _tmp11 + _tmp8 + _tmp9;
  _res[3] = _a[3] * _b[3] - _tmp12 - _tmp13 - _tmp14;
  _res[4] = _a[4] + _b[4] * _tmp28 + _b[5] * _tmp20 + _b[6] * _tmp23;
  _res[5] = _a[5] + _b[4] * _tmp29 + _b[5] * _tmp36 + _b[6] * _tmp33;
  _res[6] = _a[6] + _b[4] * _tmp38 + _b[5] * _tmp39 + _b[6] * _tmp40;

  if (res_D_a != nullptr) {
    Eigen::Matrix<Scalar, 6, 6>& _res_D_a = (*res_D_a);

    _res_D_a(0, 0) = _tmp46 * _tmp47 + _tmp49 * _tmp55 + _tmp57 * _tmp63 + _tmp69 * _tmp70;
    _res_D_a(1, 0) = _tmp46 * _tmp71 + _tmp47 * _tmp69 + _tmp48 * _tmp63 + _tmp55 * _tmp57;
    _res_D_a(2, 0) = _tmp46 * _tmp56 + _tmp47 * _tmp63 + _tmp49 * _tmp69 + _tmp55 * _tmp71;
    _res_D_a(3, 0) = _b[5] * _tmp23 + _b[6] * _tmp19;
    _res_D_a(4, 0) = _b[5] * _tmp33 - _b[6] * _tmp75;
    _res_D_a(5, 0) = _b[5] * _tmp77 - _b[6] * _tmp39;
    _res_D_a(0, 1) = _tmp47 * _tmp79 + _tmp49 * _tmp80 + _tmp57 * _tmp81 + _tmp70 * _tmp78;
    _res_D_a(1, 1) = _tmp47 * _tmp78 + _tmp48 * _tmp81 + _tmp57 * _tmp80 + _tmp71 * _tmp79;
    _res_D_a(2, 1) = _tmp47 * _tmp81 + _tmp49 * _tmp78 + _tmp56 * _tmp79 + _tmp71 * _tmp80;
    _res_D_a(3, 1) = -_b[4] * _tmp23 + _b[6] * _tmp82;
    _res_D_a(4, 1) = _b[4] * _tmp32 + _b[6] * _tmp29;
    _res_D_a(5, 1) = -_b[4] * _tmp77 + _b[6] * _tmp38;
    _res_D_a(0, 2) = _tmp47 * _tmp86 + _tmp49 * _tmp88 + _tmp57 * _tmp84 + _tmp70 * _tmp90;
    _res_D_a(1, 2) = _tmp47 * _tmp90 + _tmp48 * _tmp84 + _tmp57 * _tmp88 + _tmp71 * _tmp86;
    _res_D_a(2, 2) = _tmp47 * _tmp84 + _tmp49 * _tmp90 + _tmp56 * _tmp86 + _tmp71 * _tmp88;
    _res_D_a(3, 2) = _b[4] * _tmp20 - _b[5] * _tmp82;
    _res_D_a(4, 2) = _b[4] * _tmp75 - _b[5] * _tmp29;
    _res_D_a(5, 2) = _b[4] * _tmp39 + _b[5] * _tmp37;
    _res_D_a(0, 3) = 0;
    _res_D_a(1, 3) = 0;
    _res_D_a(2, 3) = 0;
    _res_D_a(3, 3) = 1;
    _res_D_a(4, 3) = 0;
    _res_D_a(5, 3) = 0;
    _res_D_a(0, 4) = 0;
    _res_D_a(1, 4) = 0;
    _res_D_a(2, 4) = 0;
    _res_D_a(3, 4) = 0;
    _res_D_a(4, 4) = 1;
    _res_D_a(5, 4) = 0;
    _res_D_a(0, 5) = 0;
    _res_D_a(1, 5) = 0;
    _res_D_a(2, 5) = 0;
    _res_D_a(3, 5) = 0;
    _res_D_a(4, 5) = 0;
    _res_D_a(5, 5) = 1;
  }

  if (res_D_b != nullptr) {
    Eigen::Matrix<Scalar, 6, 6>& _res_D_b = (*res_D_b);

    _res_D_b(0, 0) = _tmp70 * _tmp91 + _tmp94 + _tmp99;
    _res_D_b(1, 0) = _tmp47 * _tmp91 + _tmp48 * _tmp98 + _tmp57 * _tmp93 + _tmp71 * _tmp95;
    _res_D_b(2, 0) = _tmp47 * _tmp98 + _tmp49 * _tmp91 + _tmp56 * _tmp95 + _tmp71 * _tmp93;
    _res_D_b(3, 0) = 0;
    _res_D_b(4, 0) = 0;
    _res_D_b(5, 0) = 0;
    _res_D_b(0, 1) = _tmp100 * _tmp47 + _tmp49 * _tmp98 + _tmp57 * _tmp92 + _tmp70 * _tmp95;
    _res_D_b(1, 1) = _tmp101 + _tmp48 * _tmp92 + _tmp99;
    _res_D_b(2, 1) = _tmp100 * _tmp56 + _tmp47 * _tmp92 + _tmp49 * _tmp95 + _tmp71 * _tmp98;
    _res_D_b(3, 1) = 0;
    _res_D_b(4, 1) = 0;
    _res_D_b(5, 1) = 0;
    _res_D_b(0, 2) = _tmp100 * _tmp49 + _tmp47 * _tmp97 + _tmp57 * _tmp95 + _tmp70 * _tmp93;
    _res_D_b(1, 2) = _tmp100 * _tmp57 + _tmp47 * _tmp93 + _tmp48 * _tmp95 + _tmp71 * _tmp97;
    _res_D_b(2, 2) = _tmp101 + _tmp56 * _tmp97 + _tmp94 + _tmp96;
    _res_D_b(3, 2) = 0;
    _res_D_b(4, 2) = 0;
    _res_D_b(5, 2) = 0;
    _res_D_b(0, 3) = 0;
    _res_D_b(1, 3) = 0;
    _res_D_b(2, 3) = 0;
    _res_D_b(3, 3) = _tmp28;
    _res_D_b(4, 3) = _tmp29;
    _res_D_b(5, 3) = _tmp38;
    _res_D_b(0, 4) = 0;
    _res_D_b(1, 4) = 0;
    _res_D_b(2, 4) = 0;
    _res_D_b(3, 4) = _tmp20;
    _res_D_b(4, 4) = _tmp36;
    _res_D_b(5, 4) = _tmp39;
    _res_D_b(0, 5) = 0;
    _res_D_b(1, 5) = 0;
    _res_D_b(2, 5) = 0;
    _res_D_b(3, 5) = _tmp23;
    _res_D_b(4, 5) = _tmp33;
    _res_D_b(5, 5) = _tmp40;
  }

  return sym::Pose3<Scalar>(_res);
}  // NOLINT(readability/fn_size)

// NOLINTNEXTLINE(readability/fn_size)
}  // namespace sym

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
 */
template <typename Scalar>
sym::Pose3<Scalar> ComposePose3WithJacobian0(const sym::Pose3<Scalar>& a,
                                             const sym::Pose3<Scalar>& b,
                                             Eigen::Matrix<Scalar, 6, 6>* const res_D_a = nullptr) {
  // Total ops: 259

  // Input arrays
  const Eigen::Matrix<Scalar, 7, 1>& _a = a.Data();
  const Eigen::Matrix<Scalar, 7, 1>& _b = b.Data();

  // Intermediate terms (97)
  const Scalar _tmp0 = _a[2] * _b[1];
  const Scalar _tmp1 = _a[0] * _b[3];
  const Scalar _tmp2 = _a[1] * _b[2];
  const Scalar _tmp3 = _a[3] * _b[0];
  const Scalar _tmp4 = -_tmp0 + _tmp1 + _tmp2 + _tmp3;
  const Scalar _tmp5 = _a[3] * _b[1];
  const Scalar _tmp6 = _a[1] * _b[3];
  const Scalar _tmp7 = _a[0] * _b[2];
  const Scalar _tmp8 = _a[2] * _b[0];
  const Scalar _tmp9 = _tmp5 + _tmp6 - _tmp7 + _tmp8;
  const Scalar _tmp10 = _a[0] * _b[1];
  const Scalar _tmp11 = _a[2] * _b[3];
  const Scalar _tmp12 = _a[3] * _b[2];
  const Scalar _tmp13 = _a[1] * _b[0];
  const Scalar _tmp14 = _tmp10 + _tmp11 + _tmp12 - _tmp13;
  const Scalar _tmp15 = _a[1] * _b[1];
  const Scalar _tmp16 = _a[2] * _b[2];
  const Scalar _tmp17 = _a[0] * _b[0];
  const Scalar _tmp18 = _a[3] * _b[3];
  const Scalar _tmp19 = -_tmp15 - _tmp16 - _tmp17 + _tmp18;
  const Scalar _tmp20 = std::pow(_a[2], Scalar(2));
  const Scalar _tmp21 = -2 * _tmp20;
  const Scalar _tmp22 = std::pow(_a[1], Scalar(2));
  const Scalar _tmp23 = -2 * _tmp22;
  const Scalar _tmp24 = 2 * _a[2];
  const Scalar _tmp25 = _a[0] * _tmp24;
  const Scalar _tmp26 = 2 * _a[1];
  const Scalar _tmp27 = _a[3] * _tmp26;
  const Scalar _tmp28 = _tmp25 + _tmp27;
  const Scalar _tmp29 = _a[0] * _tmp26;
  const Scalar _tmp30 = _a[3] * _tmp24;
  const Scalar _tmp31 = -_tmp30;
  const Scalar _tmp32 = _tmp29 + _tmp31;
  const Scalar _tmp33 = std::pow(_a[0], Scalar(2));
  const Scalar _tmp34 = 1 - 2 * _tmp33;
  const Scalar _tmp35 = _a[2] * _tmp26;
  const Scalar _tmp36 = 2 * _a[0] * _a[3];
  const Scalar _tmp37 = -_tmp36;
  const Scalar _tmp38 = _tmp35 + _tmp37;
  const Scalar _tmp39 = _tmp29 + _tmp30;
  const Scalar _tmp40 = _tmp35 + _tmp36;
  const Scalar _tmp41 = -_tmp27;
  const Scalar _tmp42 = _tmp25 + _tmp41;
  const Scalar _tmp43 = (Scalar(1) / Scalar(2)) * _tmp10;
  const Scalar _tmp44 = -_tmp43;
  const Scalar _tmp45 = (Scalar(1) / Scalar(2)) * _tmp11;
  const Scalar _tmp46 = (Scalar(1) / Scalar(2)) * _tmp12;
  const Scalar _tmp47 = -Scalar(1) / Scalar(2) * _tmp13;
  const Scalar _tmp48 = -_tmp46 + _tmp47;
  const Scalar _tmp49 = _tmp44 + _tmp45 + _tmp48;
  const Scalar _tmp50 = 2 * _tmp14;
  const Scalar _tmp51 = (Scalar(1) / Scalar(2)) * _tmp5;
  const Scalar _tmp52 = (Scalar(1) / Scalar(2)) * _tmp8;
  const Scalar _tmp53 = -_tmp52;
  const Scalar _tmp54 = (Scalar(1) / Scalar(2)) * _tmp6;
  const Scalar _tmp55 = -Scalar(1) / Scalar(2) * _tmp7;
  const Scalar _tmp56 = -_tmp54 + _tmp55;
  const Scalar _tmp57 = _tmp51 + _tmp53 + _tmp56;
  const Scalar _tmp58 = 2 * _tmp9;
  const Scalar _tmp59 = (Scalar(1) / Scalar(2)) * _tmp1;
  const Scalar _tmp60 = -_tmp59;
  const Scalar _tmp61 = (Scalar(1) / Scalar(2)) * _tmp2;
  const Scalar _tmp62 = -Scalar(1) / Scalar(2) * _tmp0;
  const Scalar _tmp63 = (Scalar(1) / Scalar(2)) * _tmp3;
  const Scalar _tmp64 = _tmp62 - _tmp63;
  const Scalar _tmp65 = _tmp60 + _tmp61 + _tmp64;
  const Scalar _tmp66 = 2 * _tmp4;
  const Scalar _tmp67 = (Scalar(1) / Scalar(2)) * _tmp16;
  const Scalar _tmp68 = (Scalar(1) / Scalar(2)) * _tmp17;
  const Scalar _tmp69 = (Scalar(1) / Scalar(2)) * _tmp15;
  const Scalar _tmp70 = (Scalar(1) / Scalar(2)) * _tmp18;
  const Scalar _tmp71 = _tmp69 + _tmp70;
  const Scalar _tmp72 = _tmp67 - _tmp68 + _tmp71;
  const Scalar _tmp73 = 2 * _tmp19;
  const Scalar _tmp74 = -_tmp29;
  const Scalar _tmp75 = std::pow(_a[3], Scalar(2));
  const Scalar _tmp76 = -_tmp75;
  const Scalar _tmp77 = _tmp20 + _tmp76;
  const Scalar _tmp78 = -_tmp22;
  const Scalar _tmp79 = _tmp33 + _tmp78;
  const Scalar _tmp80 = -_tmp33;
  const Scalar _tmp81 = -_tmp35;
  const Scalar _tmp82 = _tmp67 + _tmp68 - _tmp69 + _tmp70;
  const Scalar _tmp83 = -_tmp61;
  const Scalar _tmp84 = _tmp59 + _tmp64 + _tmp83;
  const Scalar _tmp85 = -_tmp51;
  const Scalar _tmp86 = 2 * _tmp52 + 2 * _tmp56 + 2 * _tmp85;
  const Scalar _tmp87 = -_tmp45;
  const Scalar _tmp88 = _tmp44 + _tmp46 + _tmp47 + _tmp87;
  const Scalar _tmp89 = -_tmp20;
  const Scalar _tmp90 = _tmp75 + _tmp89;
  const Scalar _tmp91 = -_tmp25;
  const Scalar _tmp92 = _tmp60 + _tmp62 + _tmp63 + _tmp83;
  const Scalar _tmp93 = -_tmp67 + _tmp68 + _tmp71;
  const Scalar _tmp94 = _tmp43 + _tmp48 + _tmp87;
  const Scalar _tmp95 = _tmp53 + _tmp54 + _tmp55 + _tmp85;
  const Scalar _tmp96 = _tmp22 + _tmp80;

  // Output terms (2)
  Eigen::Matrix<Scalar, 7, 1> _res;

  _res[0] = _tmp4;
  _res[1] = _tmp9;
  _res[2] = _tmp14;
  _res[3] = _tmp19;
  _res[4] = _a[4] + _b[4] * (_tmp21 + _tmp23 + 1) + _b[5] * _tmp32 + _b[6] * _tmp28;
  _res[5] = _a[5] + _b[4] * _tmp39 + _b[5] * (_tmp21 + _tmp34) + _b[6] * _tmp38;
  _res[6] = _a[6] + _b[4] * _tmp42 + _b[5] * _tmp40 + _b[6] * (_tmp23 + _tmp34);

  if (res_D_a != nullptr) {
    Eigen::Matrix<Scalar, 6, 6>& _res_D_a = (*res_D_a);

    _res_D_a(0, 0) = _tmp49 * _tmp50 - _tmp57 * _tmp58 - _tmp65 * _tmp66 + _tmp72 * _tmp73;
    _res_D_a(1, 0) = _tmp49 * _tmp73 - _tmp50 * _tmp72 + _tmp57 * _tmp66 - _tmp58 * _tmp65;
    _res_D_a(2, 0) = -_tmp49 * _tmp66 - _tmp50 * _tmp65 + _tmp57 * _tmp73 + _tmp58 * _tmp72;
    _res_D_a(3, 0) = _b[5] * _tmp28 + _b[6] * (_tmp30 + _tmp74);
    _res_D_a(4, 0) = _b[5] * _tmp38 + _b[6] * (_tmp77 + _tmp79);
    _res_D_a(5, 0) = _b[5] * (_tmp20 + _tmp75 + _tmp78 + _tmp80) + _b[6] * (_tmp37 + _tmp81);
    _res_D_a(0, 1) = -_tmp4 * _tmp86 + _tmp50 * _tmp82 - _tmp58 * _tmp84 + _tmp73 * _tmp88;
    _res_D_a(1, 1) = -_tmp50 * _tmp88 + _tmp66 * _tmp84 + _tmp73 * _tmp82 - _tmp86 * _tmp9;
    _res_D_a(2, 1) = -_tmp14 * _tmp86 + _tmp58 * _tmp88 - _tmp66 * _tmp82 + _tmp73 * _tmp84;
    _res_D_a(3, 1) = _b[4] * (_tmp41 + _tmp91) + _b[6] * (_tmp79 + _tmp90);
    _res_D_a(4, 1) = _b[4] * (_tmp36 + _tmp81) + _b[6] * _tmp39;
    _res_D_a(5, 1) = _b[4] * (_tmp22 + _tmp33 + _tmp76 + _tmp89) + _b[6] * _tmp42;
    _res_D_a(0, 2) = _tmp50 * _tmp92 - _tmp58 * _tmp93 - _tmp66 * _tmp94 + _tmp73 * _tmp95;
    _res_D_a(1, 2) = -_tmp50 * _tmp95 - _tmp58 * _tmp94 + _tmp66 * _tmp93 + _tmp73 * _tmp92;
    _res_D_a(2, 2) = -_tmp50 * _tmp94 + _tmp58 * _tmp95 - _tmp66 * _tmp92 + _tmp73 * _tmp93;
    _res_D_a(3, 2) = _b[4] * _tmp32 + _b[5] * (_tmp77 + _tmp96);
    _res_D_a(4, 2) = _b[4] * (_tmp90 + _tmp96) + _b[5] * (_tmp31 + _tmp74);
    _res_D_a(5, 2) = _b[4] * _tmp40 + _b[5] * (_tmp27 + _tmp91);
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

  return sym::Pose3<Scalar>(_res);
}  // NOLINT(readability/fn_size)

// NOLINTNEXTLINE(readability/fn_size)
}  // namespace sym

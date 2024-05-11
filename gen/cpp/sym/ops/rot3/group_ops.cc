// -----------------------------------------------------------------------------
// This file was autogenerated by symforce from template:
//     ops/CLASS/group_ops.cc.jinja
// Do NOT modify by hand.
// -----------------------------------------------------------------------------

#include "./group_ops.h"

#include <sym/rot3.h>

namespace sym {

/**
 *
 * This function was autogenerated from a symbolic function. Do not modify by hand.
 *
 * Symbolic function: <lambda>
 *
 * Args:
 *
 * Outputs:
 *     res: Rot3
 */
template <typename Scalar>
sym::Rot3<Scalar> GroupOps<Rot3<Scalar>>::Identity() {
  // Total ops: 0

  // Input arrays

  // Intermediate terms (0)

  // Output terms (1)
  Eigen::Matrix<Scalar, 4, 1> _res;

  _res[0] = 0;
  _res[1] = 0;
  _res[2] = 0;
  _res[3] = 1;

  return sym::Rot3<Scalar>(_res, /* normalize */ false);
}

/**
 *
 * Inverse of the element a.
 *
 * Returns:
 *     Element: b such that a @ b = identity
 */
template <typename Scalar>
sym::Rot3<Scalar> GroupOps<Rot3<Scalar>>::Inverse(const sym::Rot3<Scalar>& a) {
  // Total ops: 3

  // Input arrays
  const Eigen::Matrix<Scalar, 4, 1>& _a = a.Data();

  // Intermediate terms (0)

  // Output terms (1)
  Eigen::Matrix<Scalar, 4, 1> _res;

  _res[0] = -_a[0];
  _res[1] = -_a[1];
  _res[2] = -_a[2];
  _res[3] = _a[3];

  return sym::Rot3<Scalar>(_res);
}

/**
 *
 * Composition of two elements in the group.
 *
 * Returns:
 *     Element: a @ b
 */
template <typename Scalar>
sym::Rot3<Scalar> GroupOps<Rot3<Scalar>>::Compose(const sym::Rot3<Scalar>& a,
                                                  const sym::Rot3<Scalar>& b) {
  // Total ops: 28

  // Input arrays
  const Eigen::Matrix<Scalar, 4, 1>& _a = a.Data();
  const Eigen::Matrix<Scalar, 4, 1>& _b = b.Data();

  // Intermediate terms (0)

  // Output terms (1)
  Eigen::Matrix<Scalar, 4, 1> _res;

  _res[0] = _a[0] * _b[3] + _a[1] * _b[2] - _a[2] * _b[1] + _a[3] * _b[0];
  _res[1] = -_a[0] * _b[2] + _a[1] * _b[3] + _a[2] * _b[0] + _a[3] * _b[1];
  _res[2] = _a[0] * _b[1] - _a[1] * _b[0] + _a[2] * _b[3] + _a[3] * _b[2];
  _res[3] = -_a[0] * _b[0] - _a[1] * _b[1] - _a[2] * _b[2] + _a[3] * _b[3];

  return sym::Rot3<Scalar>(_res);
}

/**
 *
 * Returns the element that when composed with a produces b. For vector spaces it is b - a.
 *
 * Implementation is simply ``compose(inverse(a), b)``.
 *
 * Returns:
 *     Element: c such that a @ c = b
 */
template <typename Scalar>
sym::Rot3<Scalar> GroupOps<Rot3<Scalar>>::Between(const sym::Rot3<Scalar>& a,
                                                  const sym::Rot3<Scalar>& b) {
  // Total ops: 28

  // Input arrays
  const Eigen::Matrix<Scalar, 4, 1>& _a = a.Data();
  const Eigen::Matrix<Scalar, 4, 1>& _b = b.Data();

  // Intermediate terms (0)

  // Output terms (1)
  Eigen::Matrix<Scalar, 4, 1> _res;

  _res[0] = -_a[0] * _b[3] - _a[1] * _b[2] + _a[2] * _b[1] + _a[3] * _b[0];
  _res[1] = _a[0] * _b[2] - _a[1] * _b[3] - _a[2] * _b[0] + _a[3] * _b[1];
  _res[2] = -_a[0] * _b[1] + _a[1] * _b[0] - _a[2] * _b[3] + _a[3] * _b[2];
  _res[3] = _a[0] * _b[0] + _a[1] * _b[1] + _a[2] * _b[2] + _a[3] * _b[3];

  return sym::Rot3<Scalar>(_res);
}

/**
 *
 * Inverse of the element a.
 *
 * Returns:
 *     Element: b such that a @ b = identity
 *     res_D_a: (3x3) jacobian of res (3) wrt arg a (3)
 */
template <typename Scalar>
sym::Rot3<Scalar> GroupOps<Rot3<Scalar>>::InverseWithJacobian(const sym::Rot3<Scalar>& a,
                                                              SelfJacobian* const res_D_a) {
  // Total ops: 34

  // Input arrays
  const Eigen::Matrix<Scalar, 4, 1>& _a = a.Data();

  // Intermediate terms (13)
  const Scalar _tmp0 = std::pow(_a[2], Scalar(2));
  const Scalar _tmp1 = std::pow(_a[0], Scalar(2));
  const Scalar _tmp2 = -std::pow(_a[3], Scalar(2));
  const Scalar _tmp3 = std::pow(_a[1], Scalar(2));
  const Scalar _tmp4 = _tmp2 + _tmp3;
  const Scalar _tmp5 = 2 * _a[2];
  const Scalar _tmp6 = _a[3] * _tmp5;
  const Scalar _tmp7 = -2 * _a[0] * _a[1];
  const Scalar _tmp8 = 2 * _a[3];
  const Scalar _tmp9 = _a[1] * _tmp8;
  const Scalar _tmp10 = -_a[0] * _tmp5;
  const Scalar _tmp11 = _a[0] * _tmp8;
  const Scalar _tmp12 = -_a[1] * _tmp5;

  // Output terms (2)
  Eigen::Matrix<Scalar, 4, 1> _res;

  _res[0] = -_a[0];
  _res[1] = -_a[1];
  _res[2] = -_a[2];
  _res[3] = _a[3];

  if (res_D_a != nullptr) {
    Eigen::Matrix<Scalar, 3, 3>& _res_D_a = (*res_D_a);

    _res_D_a(0, 0) = _tmp0 - _tmp1 + _tmp4;
    _res_D_a(1, 0) = -_tmp6 + _tmp7;
    _res_D_a(2, 0) = _tmp10 + _tmp9;
    _res_D_a(0, 1) = _tmp6 + _tmp7;
    _res_D_a(1, 1) = _tmp0 + _tmp1 + _tmp2 - _tmp3;
    _res_D_a(2, 1) = -_tmp11 + _tmp12;
    _res_D_a(0, 2) = _tmp10 - _tmp9;
    _res_D_a(1, 2) = _tmp11 + _tmp12;
    _res_D_a(2, 2) = -_tmp0 + _tmp1 + _tmp4;
  }

  return sym::Rot3<Scalar>(_res);
}

/**
 *
 * Composition of two elements in the group.
 *
 * Returns:
 *     Element: a @ b
 *     res_D_a: (3x3) jacobian of res (3) wrt arg a (3)
 *     res_D_b: (3x3) jacobian of res (3) wrt arg b (3)
 */
template <typename Scalar>
sym::Rot3<Scalar> GroupOps<Rot3<Scalar>>::ComposeWithJacobians(const sym::Rot3<Scalar>& a,
                                                               const sym::Rot3<Scalar>& b,
                                                               SelfJacobian* const res_D_a,
                                                               SelfJacobian* const res_D_b) {
  // Total ops: 224

  // Input arrays
  const Eigen::Matrix<Scalar, 4, 1>& _a = a.Data();
  const Eigen::Matrix<Scalar, 4, 1>& _b = b.Data();

  // Intermediate terms (94)
  const Scalar _tmp0 = _a[3] * _b[0];
  const Scalar _tmp1 = _a[2] * _b[1];
  const Scalar _tmp2 = _a[0] * _b[3];
  const Scalar _tmp3 = _a[1] * _b[2];
  const Scalar _tmp4 = _tmp0 - _tmp1 + _tmp2 + _tmp3;
  const Scalar _tmp5 = _a[3] * _b[1];
  const Scalar _tmp6 = _a[2] * _b[0];
  const Scalar _tmp7 = _a[0] * _b[2];
  const Scalar _tmp8 = _a[1] * _b[3];
  const Scalar _tmp9 = _tmp5 + _tmp6 - _tmp7 + _tmp8;
  const Scalar _tmp10 = _a[3] * _b[2];
  const Scalar _tmp11 = _a[2] * _b[3];
  const Scalar _tmp12 = _a[0] * _b[1];
  const Scalar _tmp13 = _a[1] * _b[0];
  const Scalar _tmp14 = _tmp10 + _tmp11 + _tmp12 - _tmp13;
  const Scalar _tmp15 = _a[3] * _b[3];
  const Scalar _tmp16 = _a[2] * _b[2];
  const Scalar _tmp17 = _a[0] * _b[0];
  const Scalar _tmp18 = _a[1] * _b[1];
  const Scalar _tmp19 = _tmp15 - _tmp16 - _tmp17 - _tmp18;
  const Scalar _tmp20 = (Scalar(1) / Scalar(2)) * _tmp13;
  const Scalar _tmp21 = -_tmp20;
  const Scalar _tmp22 = (Scalar(1) / Scalar(2)) * _tmp11;
  const Scalar _tmp23 = _tmp21 + _tmp22;
  const Scalar _tmp24 = (Scalar(1) / Scalar(2)) * _tmp10;
  const Scalar _tmp25 = -_tmp24;
  const Scalar _tmp26 = (Scalar(1) / Scalar(2)) * _tmp12;
  const Scalar _tmp27 = -_tmp26;
  const Scalar _tmp28 = _tmp25 + _tmp27;
  const Scalar _tmp29 = _tmp23 + _tmp28;
  const Scalar _tmp30 = 2 * _tmp14;
  const Scalar _tmp31 = (Scalar(1) / Scalar(2)) * _tmp3;
  const Scalar _tmp32 = (Scalar(1) / Scalar(2)) * _tmp0;
  const Scalar _tmp33 = -_tmp32;
  const Scalar _tmp34 = (Scalar(1) / Scalar(2)) * _tmp1;
  const Scalar _tmp35 = -_tmp34;
  const Scalar _tmp36 = (Scalar(1) / Scalar(2)) * _tmp2;
  const Scalar _tmp37 = -_tmp36;
  const Scalar _tmp38 = _tmp35 + _tmp37;
  const Scalar _tmp39 = _tmp31 + _tmp33 + _tmp38;
  const Scalar _tmp40 = 2 * _tmp4;
  const Scalar _tmp41 = (Scalar(1) / Scalar(2)) * _tmp5;
  const Scalar _tmp42 = (Scalar(1) / Scalar(2)) * _tmp6;
  const Scalar _tmp43 = -_tmp42;
  const Scalar _tmp44 = (Scalar(1) / Scalar(2)) * _tmp7;
  const Scalar _tmp45 = -_tmp44;
  const Scalar _tmp46 = (Scalar(1) / Scalar(2)) * _tmp8;
  const Scalar _tmp47 = -_tmp46;
  const Scalar _tmp48 = _tmp45 + _tmp47;
  const Scalar _tmp49 = _tmp41 + _tmp43 + _tmp48;
  const Scalar _tmp50 = 2 * _tmp9;
  const Scalar _tmp51 = (Scalar(1) / Scalar(2)) * _tmp17;
  const Scalar _tmp52 = -_tmp51;
  const Scalar _tmp53 = (Scalar(1) / Scalar(2)) * _tmp16;
  const Scalar _tmp54 = (Scalar(1) / Scalar(2)) * _tmp15;
  const Scalar _tmp55 = (Scalar(1) / Scalar(2)) * _tmp18;
  const Scalar _tmp56 = _tmp54 + _tmp55;
  const Scalar _tmp57 = _tmp52 + _tmp53 + _tmp56;
  const Scalar _tmp58 = 2 * _tmp19;
  const Scalar _tmp59 = _tmp54 - _tmp55;
  const Scalar _tmp60 = _tmp51 + _tmp53 + _tmp59;
  const Scalar _tmp61 = -_tmp41;
  const Scalar _tmp62 = _tmp42 + _tmp48 + _tmp61;
  const Scalar _tmp63 = _tmp35 + _tmp36;
  const Scalar _tmp64 = -_tmp31;
  const Scalar _tmp65 = _tmp33 + _tmp64;
  const Scalar _tmp66 = _tmp63 + _tmp65;
  const Scalar _tmp67 = -_tmp22;
  const Scalar _tmp68 = _tmp21 + _tmp67;
  const Scalar _tmp69 = _tmp24 + _tmp27 + _tmp68;
  const Scalar _tmp70 = _tmp32 + _tmp38 + _tmp64;
  const Scalar _tmp71 = _tmp25 + _tmp26 + _tmp68;
  const Scalar _tmp72 = -_tmp53;
  const Scalar _tmp73 = _tmp51 + _tmp56 + _tmp72;
  const Scalar _tmp74 = _tmp45 + _tmp46;
  const Scalar _tmp75 = _tmp43 + _tmp61;
  const Scalar _tmp76 = _tmp74 + _tmp75;
  const Scalar _tmp77 = _tmp23 + _tmp24 + _tmp26;
  const Scalar _tmp78 = _tmp44 + _tmp47 + _tmp75;
  const Scalar _tmp79 = -_tmp50 * _tmp78;
  const Scalar _tmp80 = _tmp34 + _tmp37 + _tmp65;
  const Scalar _tmp81 = _tmp52 + _tmp59 + _tmp72;
  const Scalar _tmp82 = _tmp58 * _tmp81;
  const Scalar _tmp83 = -_tmp40 * _tmp80 + _tmp82;
  const Scalar _tmp84 = _tmp30 * _tmp81;
  const Scalar _tmp85 = _tmp40 * _tmp78;
  const Scalar _tmp86 = _tmp30 * _tmp80;
  const Scalar _tmp87 = _tmp50 * _tmp81;
  const Scalar _tmp88 = _tmp31 + _tmp32 + _tmp63;
  const Scalar _tmp89 = _tmp20 + _tmp28 + _tmp67;
  const Scalar _tmp90 = -_tmp30 * _tmp89;
  const Scalar _tmp91 = _tmp40 * _tmp81;
  const Scalar _tmp92 = _tmp50 * _tmp89;
  const Scalar _tmp93 = _tmp41 + _tmp42 + _tmp74;

  // Output terms (3)
  Eigen::Matrix<Scalar, 4, 1> _res;

  _res[0] = _tmp4;
  _res[1] = _tmp9;
  _res[2] = _tmp14;
  _res[3] = _tmp19;

  if (res_D_a != nullptr) {
    Eigen::Matrix<Scalar, 3, 3>& _res_D_a = (*res_D_a);

    _res_D_a(0, 0) = _tmp29 * _tmp30 - _tmp39 * _tmp40 - _tmp49 * _tmp50 + _tmp57 * _tmp58;
    _res_D_a(1, 0) = _tmp29 * _tmp58 - _tmp30 * _tmp57 - _tmp39 * _tmp50 + _tmp40 * _tmp49;
    _res_D_a(2, 0) = -_tmp29 * _tmp40 - _tmp30 * _tmp39 + _tmp49 * _tmp58 + _tmp50 * _tmp57;
    _res_D_a(0, 1) = _tmp30 * _tmp60 - _tmp40 * _tmp62 - _tmp50 * _tmp66 + _tmp58 * _tmp69;
    _res_D_a(1, 1) = -_tmp30 * _tmp69 + _tmp40 * _tmp66 - _tmp50 * _tmp62 + _tmp58 * _tmp60;
    _res_D_a(2, 1) = -_tmp30 * _tmp62 - _tmp40 * _tmp60 + _tmp50 * _tmp69 + _tmp58 * _tmp66;
    _res_D_a(0, 2) = _tmp30 * _tmp70 - _tmp40 * _tmp71 - _tmp50 * _tmp73 + _tmp58 * _tmp76;
    _res_D_a(1, 2) = -_tmp30 * _tmp76 + _tmp40 * _tmp73 - _tmp50 * _tmp71 + _tmp58 * _tmp70;
    _res_D_a(2, 2) = -_tmp30 * _tmp71 - _tmp40 * _tmp70 + _tmp50 * _tmp76 + _tmp58 * _tmp73;
  }

  if (res_D_b != nullptr) {
    Eigen::Matrix<Scalar, 3, 3>& _res_D_b = (*res_D_b);

    _res_D_b(0, 0) = _tmp30 * _tmp77 + _tmp79 + _tmp83;
    _res_D_b(1, 0) = -_tmp50 * _tmp80 + _tmp58 * _tmp77 - _tmp84 + _tmp85;
    _res_D_b(2, 0) = -_tmp40 * _tmp77 + _tmp58 * _tmp78 - _tmp86 + _tmp87;
    _res_D_b(0, 1) = -_tmp50 * _tmp88 + _tmp58 * _tmp89 + _tmp84 - _tmp85;
    _res_D_b(1, 1) = _tmp40 * _tmp88 + _tmp79 + _tmp82 + _tmp90;
    _res_D_b(2, 1) = -_tmp30 * _tmp78 + _tmp58 * _tmp88 - _tmp91 + _tmp92;
    _res_D_b(0, 2) = -_tmp40 * _tmp89 + _tmp58 * _tmp93 + _tmp86 - _tmp87;
    _res_D_b(1, 2) = -_tmp30 * _tmp93 + _tmp58 * _tmp80 + _tmp91 - _tmp92;
    _res_D_b(2, 2) = _tmp50 * _tmp93 + _tmp83 + _tmp90;
  }

  return sym::Rot3<Scalar>(_res);
}

/**
 *
 * Returns the element that when composed with a produces b. For vector spaces it is b - a.
 *
 * Implementation is simply ``compose(inverse(a), b)``.
 *
 * Returns:
 *     Element: c such that a @ c = b
 *     res_D_a: (3x3) jacobian of res (3) wrt arg a (3)
 *     res_D_b: (3x3) jacobian of res (3) wrt arg b (3)
 */
template <typename Scalar>
sym::Rot3<Scalar> GroupOps<Rot3<Scalar>>::BetweenWithJacobians(const sym::Rot3<Scalar>& a,
                                                               const sym::Rot3<Scalar>& b,
                                                               SelfJacobian* const res_D_a,
                                                               SelfJacobian* const res_D_b) {
  // Total ops: 161

  // Input arrays
  const Eigen::Matrix<Scalar, 4, 1>& _a = a.Data();
  const Eigen::Matrix<Scalar, 4, 1>& _b = b.Data();

  // Intermediate terms (78)
  const Scalar _tmp0 = _a[3] * _b[0];
  const Scalar _tmp1 = _a[2] * _b[1];
  const Scalar _tmp2 = _a[0] * _b[3];
  const Scalar _tmp3 = _a[1] * _b[2];
  const Scalar _tmp4 = _tmp0 + _tmp1 - _tmp2 - _tmp3;
  const Scalar _tmp5 = _a[3] * _b[1];
  const Scalar _tmp6 = _a[2] * _b[0];
  const Scalar _tmp7 = _a[0] * _b[2];
  const Scalar _tmp8 = _a[1] * _b[3];
  const Scalar _tmp9 = _tmp5 - _tmp6 + _tmp7 - _tmp8;
  const Scalar _tmp10 = _a[3] * _b[2];
  const Scalar _tmp11 = _a[2] * _b[3];
  const Scalar _tmp12 = _a[0] * _b[1];
  const Scalar _tmp13 = _a[1] * _b[0];
  const Scalar _tmp14 = _tmp10 - _tmp11 - _tmp12 + _tmp13;
  const Scalar _tmp15 = _a[3] * _b[3];
  const Scalar _tmp16 = _a[2] * _b[2];
  const Scalar _tmp17 = _a[0] * _b[0];
  const Scalar _tmp18 = _a[1] * _b[1];
  const Scalar _tmp19 = _tmp15 + _tmp16 + _tmp17 + _tmp18;
  const Scalar _tmp20 = (Scalar(1) / Scalar(2)) * _tmp15;
  const Scalar _tmp21 = (Scalar(1) / Scalar(2)) * _tmp16;
  const Scalar _tmp22 = (Scalar(1) / Scalar(2)) * _tmp17;
  const Scalar _tmp23 = (Scalar(1) / Scalar(2)) * _tmp18;
  const Scalar _tmp24 = -_tmp20 - _tmp21 - _tmp22 - _tmp23;
  const Scalar _tmp25 = 2 * _tmp19;
  const Scalar _tmp26 = _tmp24 * _tmp25;
  const Scalar _tmp27 = (Scalar(1) / Scalar(2)) * _tmp0;
  const Scalar _tmp28 = (Scalar(1) / Scalar(2)) * _tmp1;
  const Scalar _tmp29 = (Scalar(1) / Scalar(2)) * _tmp2;
  const Scalar _tmp30 = (Scalar(1) / Scalar(2)) * _tmp3;
  const Scalar _tmp31 = _tmp27 + _tmp28 - _tmp29 - _tmp30;
  const Scalar _tmp32 = 2 * _tmp4;
  const Scalar _tmp33 = _tmp31 * _tmp32;
  const Scalar _tmp34 = (Scalar(1) / Scalar(2)) * _tmp10;
  const Scalar _tmp35 = (Scalar(1) / Scalar(2)) * _tmp11;
  const Scalar _tmp36 = (Scalar(1) / Scalar(2)) * _tmp12;
  const Scalar _tmp37 = (Scalar(1) / Scalar(2)) * _tmp13;
  const Scalar _tmp38 = _tmp34 - _tmp35 - _tmp36 + _tmp37;
  const Scalar _tmp39 = 2 * _tmp14;
  const Scalar _tmp40 = _tmp38 * _tmp39;
  const Scalar _tmp41 = (Scalar(1) / Scalar(2)) * _tmp5;
  const Scalar _tmp42 = (Scalar(1) / Scalar(2)) * _tmp6;
  const Scalar _tmp43 = (Scalar(1) / Scalar(2)) * _tmp7;
  const Scalar _tmp44 = (Scalar(1) / Scalar(2)) * _tmp8;
  const Scalar _tmp45 = -_tmp41 + _tmp42 - _tmp43 + _tmp44;
  const Scalar _tmp46 = 2 * _tmp9;
  const Scalar _tmp47 = -_tmp45 * _tmp46;
  const Scalar _tmp48 = _tmp40 + _tmp47;
  const Scalar _tmp49 = -_tmp31 * _tmp46;
  const Scalar _tmp50 = 2 * _tmp24;
  const Scalar _tmp51 = _tmp14 * _tmp50;
  const Scalar _tmp52 = _tmp32 * _tmp45;
  const Scalar _tmp53 = _tmp25 * _tmp38 + _tmp52;
  const Scalar _tmp54 = _tmp50 * _tmp9;
  const Scalar _tmp55 = -_tmp32 * _tmp38;
  const Scalar _tmp56 = _tmp25 * _tmp45 + _tmp55;
  const Scalar _tmp57 = _tmp41 - _tmp42 + _tmp43 - _tmp44;
  const Scalar _tmp58 = -2 * _tmp34 + 2 * _tmp35 + 2 * _tmp36 - 2 * _tmp37;
  const Scalar _tmp59 = _tmp19 * _tmp58 + _tmp49;
  const Scalar _tmp60 = _tmp46 * _tmp57;
  const Scalar _tmp61 = -_tmp14 * _tmp58;
  const Scalar _tmp62 = _tmp33 + _tmp61;
  const Scalar _tmp63 = -_tmp39 * _tmp57;
  const Scalar _tmp64 = _tmp4 * _tmp50;
  const Scalar _tmp65 = _tmp58 * _tmp9;
  const Scalar _tmp66 = _tmp25 * _tmp31 + _tmp65;
  const Scalar _tmp67 = -_tmp27 - _tmp28 + _tmp29 + _tmp30;
  const Scalar _tmp68 = _tmp39 * _tmp67;
  const Scalar _tmp69 = _tmp25 * _tmp57 + _tmp68;
  const Scalar _tmp70 = _tmp25 * _tmp67 + _tmp63;
  const Scalar _tmp71 = -_tmp32 * _tmp67;
  const Scalar _tmp72 = _tmp20 + _tmp21 + _tmp22 + _tmp23;
  const Scalar _tmp73 = _tmp25 * _tmp72;
  const Scalar _tmp74 = _tmp71 + _tmp73;
  const Scalar _tmp75 = _tmp39 * _tmp72;
  const Scalar _tmp76 = _tmp46 * _tmp72;
  const Scalar _tmp77 = _tmp32 * _tmp72;

  // Output terms (3)
  Eigen::Matrix<Scalar, 4, 1> _res;

  _res[0] = _tmp4;
  _res[1] = _tmp9;
  _res[2] = _tmp14;
  _res[3] = _tmp19;

  if (res_D_a != nullptr) {
    Eigen::Matrix<Scalar, 3, 3>& _res_D_a = (*res_D_a);

    _res_D_a(0, 0) = _tmp26 - _tmp33 + _tmp48;
    _res_D_a(1, 0) = _tmp49 - _tmp51 + _tmp53;
    _res_D_a(2, 0) = -_tmp31 * _tmp39 + _tmp54 + _tmp56;
    _res_D_a(0, 1) = -_tmp32 * _tmp57 + _tmp51 + _tmp59;
    _res_D_a(1, 1) = _tmp26 - _tmp60 + _tmp62;
    _res_D_a(2, 1) = _tmp63 - _tmp64 + _tmp66;
    _res_D_a(0, 2) = -_tmp54 + _tmp55 + _tmp69;
    _res_D_a(1, 2) = -_tmp38 * _tmp46 + _tmp64 + _tmp70;
    _res_D_a(2, 2) = _tmp26 - _tmp40 + _tmp60 + _tmp71;
  }

  if (res_D_b != nullptr) {
    Eigen::Matrix<Scalar, 3, 3>& _res_D_b = (*res_D_b);

    _res_D_b(0, 0) = _tmp48 + _tmp74;
    _res_D_b(1, 0) = -_tmp46 * _tmp67 + _tmp53 - _tmp75;
    _res_D_b(2, 0) = _tmp56 - _tmp68 + _tmp76;
    _res_D_b(0, 1) = -_tmp52 + _tmp59 + _tmp75;
    _res_D_b(1, 1) = _tmp47 + _tmp62 + _tmp73;
    _res_D_b(2, 1) = -_tmp39 * _tmp45 + _tmp66 - _tmp77;
    _res_D_b(0, 2) = -_tmp4 * _tmp58 + _tmp69 - _tmp76;
    _res_D_b(1, 2) = -_tmp65 + _tmp70 + _tmp77;
    _res_D_b(2, 2) = _tmp60 + _tmp61 + _tmp74;
  }

  return sym::Rot3<Scalar>(_res);
}

}  // namespace sym

// Explicit instantiation
template struct sym::GroupOps<sym::Rot3<double>>;
template struct sym::GroupOps<sym::Rot3<float>>;
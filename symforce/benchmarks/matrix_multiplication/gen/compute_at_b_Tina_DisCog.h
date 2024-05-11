// -----------------------------------------------------------------------------
// This file was autogenerated by symforce from template:
//     function/FUNCTION.h.jinja
// Do NOT modify by hand.
// -----------------------------------------------------------------------------

#pragma once


#include <Eigen/Core>



namespace sym {


/**
 * This function was autogenerated. Do not modify by hand.
 *
 * Args:
 *     x0: Scalar
 *     x1: Scalar
 *     x2: Scalar
 *     x3: Scalar
 *     x4: Scalar
 *
 * Outputs:
 *     result: Matrix11_11
 */
template <typename Scalar>
__attribute__((noinline))
Eigen::Matrix<Scalar, 11, 11> ComputeAtBTinaDiscog(const Scalar x0, const Scalar x1, const Scalar x2, const Scalar x3, const Scalar x4) {

    // Total ops: 558

    // Input arrays

    // Intermediate terms (133)
    const Scalar _tmp0 = 2*x0;
    const Scalar _tmp1 = _tmp0 - 2;
    const Scalar _tmp2 = _tmp1 + x1;
    const Scalar _tmp3 = x2*x4;
    const Scalar _tmp4 = 2*x2;
    const Scalar _tmp5 = _tmp4 + 6;
    const Scalar _tmp6 = std::pow(x0, Scalar(2));
    const Scalar _tmp7 = -_tmp6 + 2 + Scalar(1.0) / (x0);
    const Scalar _tmp8 = -x4;
    const Scalar _tmp9 = _tmp8 + x2;
    const Scalar _tmp10 = _tmp9/x3;
    const Scalar _tmp11 = 2*x1;
    const Scalar _tmp12 = x0 + 2;
    const Scalar _tmp13 = 2*x4;
    const Scalar _tmp14 = _tmp11 - _tmp12*_tmp13;
    const Scalar _tmp15 = -x1;
    const Scalar _tmp16 = _tmp12 + _tmp15;
    const Scalar _tmp17 = -x3;
    const Scalar _tmp18 = _tmp17 + x0;
    const Scalar _tmp19 = _tmp18 + Scalar(1.0);
    const Scalar _tmp20 = _tmp0 - 6;
    const Scalar _tmp21 = -x2;
    const Scalar _tmp22 = x1*x3;
    const Scalar _tmp23 = _tmp17 + _tmp22;
    const Scalar _tmp24 = Scalar(1.0) / (x4);
    const Scalar _tmp25 = _tmp24*x0;
    const Scalar _tmp26 = _tmp21 + _tmp23*_tmp25;
    const Scalar _tmp27 = -_tmp26;
    const Scalar _tmp28 = x0 + x1;
    const Scalar _tmp29 = _tmp28*(x1 + x3);
    const Scalar _tmp30 = _tmp29*x4;
    const Scalar _tmp31 = x1 + 3;
    const Scalar _tmp32 = _tmp17 + _tmp31;
    const Scalar _tmp33 = Scalar(2.0)*x0;
    const Scalar _tmp34 = std::pow(x4, Scalar(2));
    const Scalar _tmp35 = _tmp34*x0;
    const Scalar _tmp36 = _tmp15 + x0;
    const Scalar _tmp37 = _tmp36 + 6;
    const Scalar _tmp38 = x4 - 2;
    const Scalar _tmp39 = x1 - 4;
    const Scalar _tmp40 = _tmp13 - 3;
    const Scalar _tmp41 = x0*x4;
    const Scalar _tmp42 = x2 + 2;
    const Scalar _tmp43 = x3*x4;
    const Scalar _tmp44 = _tmp43 + x1 + 2;
    const Scalar _tmp45 = x0*x1;
    const Scalar _tmp46 = _tmp15 + _tmp45 + 2;
    const Scalar _tmp47 = _tmp11 + x3;
    const Scalar _tmp48 = _tmp21 + x1;
    const Scalar _tmp49 = _tmp48*_tmp9;
    const Scalar _tmp50 = Scalar(1.0) / (x2);
    const Scalar _tmp51 = _tmp50*x3;
    const Scalar _tmp52 = _tmp42*x3;
    const Scalar _tmp53 = x1 + x4;
    const Scalar _tmp54 = std::pow(x2, Scalar(2));
    const Scalar _tmp55 = _tmp53*_tmp54;
    const Scalar _tmp56 = x3 - 2;
    const Scalar _tmp57 = x1 + 4;
    const Scalar _tmp58 = _tmp52*x2;
    const Scalar _tmp59 = x1 - 1;
    const Scalar _tmp60 = _tmp13*_tmp59;
    const Scalar _tmp61 = _tmp21 - _tmp31*x1;
    const Scalar _tmp62 = _tmp1*_tmp50;
    const Scalar _tmp63 = _tmp48 + 1;
    const Scalar _tmp64 = Scalar(1.0)*x0;
    const Scalar _tmp65 = -_tmp24*_tmp4 + 4*x0 + 4;
    const Scalar _tmp66 = -x0 + (Scalar(1)/Scalar(2))*x3 + 1;
    const Scalar _tmp67 = x3 + 4;
    const Scalar _tmp68 = _tmp11 + 1;
    const Scalar _tmp69 = _tmp21 + x4 + 4;
    const Scalar _tmp70 = x2 + 1;
    const Scalar _tmp71 = _tmp45*_tmp70;
    const Scalar _tmp72 = _tmp0 + 1;
    const Scalar _tmp73 = (Scalar(1)/Scalar(2))*x4;
    const Scalar _tmp74 = _tmp73 - Scalar(1)/Scalar(2)*x0 + x3;
    const Scalar _tmp75 = x0 - 3;
    const Scalar _tmp76 = _tmp18 + _tmp70;
    const Scalar _tmp77 = _tmp38 + 3*x0;
    const Scalar _tmp78 = std::pow(x1, Scalar(2));
    const Scalar _tmp79 = 4*_tmp78;
    const Scalar _tmp80 = x3 + 1;
    const Scalar _tmp81 = _tmp36*x0;
    const Scalar _tmp82 = _tmp13 + x0;
    const Scalar _tmp83 = x1*x2;
    const Scalar _tmp84 = _tmp83 + x3;
    const Scalar _tmp85 = x0 + 1;
    const Scalar _tmp86 = _tmp78*_tmp85;
    const Scalar _tmp87 = _tmp4*x3;
    const Scalar _tmp88 = _tmp4 + x3;
    const Scalar _tmp89 = Scalar(1.0) / (x2 - 2);
    const Scalar _tmp90 = _tmp89*x3;
    const Scalar _tmp91 = _tmp79*x2;
    const Scalar _tmp92 = _tmp28 + 1;
    const Scalar _tmp93 = _tmp1*x1;
    const Scalar _tmp94 = _tmp0*x4;
    const Scalar _tmp95 = _tmp54*_tmp94;
    const Scalar _tmp96 = _tmp15 + _tmp95;
    const Scalar _tmp97 = _tmp17 + x1;
    const Scalar _tmp98 = _tmp25*_tmp97 - 1;
    const Scalar _tmp99 = -_tmp98;
    const Scalar _tmp100 = x3/(_tmp15 + x4 + 1);
    const Scalar _tmp101 = -_tmp24*x2 + x2 + x3;
    const Scalar _tmp102 = -_tmp3 + _tmp94;
    const Scalar _tmp103 = x1*x4;
    const Scalar _tmp104 = _tmp0*_tmp103;
    const Scalar _tmp105 = x0 + 3;
    const Scalar _tmp106 = _tmp23 + 2;
    const Scalar _tmp107 = _tmp83 - 2;
    const Scalar _tmp108 = _tmp85*(_tmp8 + x0);
    const Scalar _tmp109 = x0*(_tmp103 + x4);
    const Scalar _tmp110 = _tmp0*x3;
    const Scalar _tmp111 = _tmp110 + 2;
    const Scalar _tmp112 = 6*x1;
    const Scalar _tmp113 = _tmp68*x3;
    const Scalar _tmp114 = _tmp56 + x2;
    const Scalar _tmp115 = _tmp16*x4;
    const Scalar _tmp116 = _tmp17 + _tmp4 + _tmp85;
    const Scalar _tmp117 = _tmp53*x2;
    const Scalar _tmp118 = x3 + 2;
    const Scalar _tmp119 = _tmp118 + x4;
    const Scalar _tmp120 = 6*x0;
    const Scalar _tmp121 = 3*x3;
    const Scalar _tmp122 = x0 + x2;
    const Scalar _tmp123 = _tmp38 + x0;
    const Scalar _tmp124 = _tmp73*x2 + Scalar(-1)/Scalar(2);
    const Scalar _tmp125 = 2 - 2/(x1 + 1);
    const Scalar _tmp126 = _tmp118/_tmp70;
    const Scalar _tmp127 = _tmp126*x2;
    const Scalar _tmp128 = _tmp120 - 3;
    const Scalar _tmp129 = _tmp8 + _tmp85;
    const Scalar _tmp130 = _tmp4*x0;
    const Scalar _tmp131 = _tmp57 + x2;
    const Scalar _tmp132 = x1 + x2;

    // Output terms (1)
    Eigen::Matrix<Scalar, 11, 11> _result;


    _result(0, 0) = _tmp2*_tmp3 + _tmp5*_tmp7;
    _result(1, 0) = -_tmp10*_tmp2 - _tmp14*_tmp5;
    _result(2, 0) = -_tmp16*_tmp5;
    _result(3, 0) = _tmp19*_tmp2;
    _result(4, 0) = -_tmp2*_tmp20 + _tmp27*_tmp5;
    _result(5, 0) = -_tmp2*_tmp30;
    _result(6, 0) = -_tmp2*(-_tmp31*x1 - x2);
    _result(7, 0) = -_tmp2*_tmp32;
    _result(8, 0) = 0;
    _result(9, 0) = -_tmp33 - Scalar(1.0)*x1 + Scalar(2.0);
    _result(10, 0) = 0;
    _result(0, 1) = _tmp35*x2 - _tmp37*_tmp7;
    _result(1, 1) = -_tmp0*_tmp38 - _tmp10*_tmp41 + _tmp14*_tmp37 - _tmp16*_tmp42*_tmp43*x2 + _tmp39*_tmp40 + _tmp44*_tmp46;
    _result(2, 1) = -_tmp0*_tmp47 + _tmp16*_tmp37 - _tmp44*_tmp49;
    _result(3, 1) = _tmp19*_tmp41 - _tmp40*_tmp51 - _tmp52*_tmp55;
    _result(4, 1) = -_tmp20*_tmp41 - _tmp27*_tmp37;
    _result(5, 1) = _tmp0*_tmp56 - _tmp29*_tmp35 - _tmp40*_tmp57;
    _result(6, 1) = -_tmp40*_tmp60 - _tmp41*_tmp61 - _tmp44*_tmp62 + _tmp58;
    _result(7, 1) = -_tmp32*_tmp41 - _tmp44*_tmp63;
    _result(8, 1) = 0;
    _result(9, 1) = _tmp0*_tmp65 + _tmp40*_tmp66 - _tmp64*x4;
    _result(10, 1) = -_tmp44*_tmp68 + _tmp58*_tmp67;
    _result(0, 2) = -_tmp69*_tmp7;
    _result(1, 2) = _tmp14*_tmp69 - _tmp38*_tmp71 - _tmp46*_tmp72;
    _result(2, 2) = _tmp16*_tmp69 - _tmp47*_tmp71 + _tmp49*_tmp72;
    _result(3, 2) = 0;
    _result(4, 2) = _tmp26*_tmp69;
    _result(5, 2) = _tmp56*_tmp71;
    _result(6, 2) = _tmp62*_tmp72;
    _result(7, 2) = _tmp63*_tmp72;
    _result(8, 2) = 0;
    _result(9, 2) = _tmp65*_tmp71;
    _result(10, 2) = _tmp68*_tmp72;
    _result(0, 3) = _tmp3*_tmp74;
    _result(1, 3) = -_tmp10*_tmp74 + _tmp16*_tmp3*_tmp79 - _tmp39*_tmp77 + _tmp75*_tmp76;
    _result(2, 3) = 0;
    _result(3, 3) = _tmp19*_tmp74 + _tmp51*_tmp77 + _tmp55*_tmp79 - _tmp76*_tmp86 - _tmp80*_tmp81 - _tmp82*_tmp84;
    _result(4, 3) = -_tmp20*_tmp74 + _tmp76*_tmp87;
    _result(5, 3) = -_tmp30*_tmp74 + _tmp48*_tmp90 + _tmp57*_tmp77 - _tmp80*_tmp88;
    _result(6, 3) = _tmp11*_tmp76 + _tmp60*_tmp77 - _tmp61*_tmp74 - _tmp91;
    _result(7, 3) = -_tmp32*_tmp74 - _tmp48*_tmp96 + _tmp76*_tmp99 - 2*_tmp80*_tmp92*x3 - _tmp82*_tmp93;
    _result(8, 3) = _tmp100*_tmp82;
    _result(9, 3) = -_tmp66*_tmp77 + Scalar(0.5)*x0 - Scalar(1.0)*x3 - Scalar(0.5)*x4;
    _result(10, 3) = _tmp101*_tmp82 - _tmp67*_tmp91;
    _result(0, 4) = -_tmp102*_tmp3 + _tmp104*_tmp7;
    _result(1, 4) = _tmp10*_tmp102 - _tmp104*_tmp14 - _tmp105*_tmp75;
    _result(2, 4) = -_tmp104*_tmp16;
    _result(3, 4) = -_tmp102*_tmp19 + _tmp105*_tmp86;
    _result(4, 4) = _tmp102*_tmp20 - _tmp104*_tmp26 - _tmp105*_tmp87;
    _result(5, 4) = _tmp102*_tmp30;
    _result(6, 4) = _tmp102*_tmp61 - _tmp105*_tmp11;
    _result(7, 4) = _tmp102*_tmp32 - _tmp105*_tmp99;
    _result(8, 4) = 0;
    _result(9, 4) = -Scalar(1.0)*_tmp3 + _tmp33*x4;
    _result(10, 4) = 0;
    _result(0, 5) = _tmp106*_tmp3;
    _result(1, 5) = -_tmp10*_tmp106 + _tmp107*_tmp38 - _tmp108*_tmp39;
    _result(2, 5) = _tmp107*_tmp47;
    _result(3, 5) = _tmp106*_tmp19 + _tmp108*_tmp51 + _tmp81*_tmp97;
    _result(4, 5) = -_tmp106*_tmp20;
    _result(5, 5) = -_tmp106*_tmp30 - _tmp107*_tmp56 + _tmp108*_tmp57 + _tmp109*_tmp90 + _tmp88*_tmp97;
    _result(6, 5) = -_tmp106*_tmp61 + _tmp108*_tmp60;
    _result(7, 5) = -_tmp106*_tmp32 - _tmp109*_tmp96 + 2*_tmp92*_tmp97*x3;
    _result(8, 5) = 0;
    _result(9, 5) = -_tmp107*_tmp65 - _tmp108*_tmp66 - Scalar(1.0)*_tmp22 + Scalar(1.0)*x3 + Scalar(-2.0);
    _result(10, 5) = 0;
    _result(0, 6) = _tmp111*_tmp3;
    _result(1, 6) = -_tmp10*_tmp111 + _tmp112 - _tmp113*_tmp75 - _tmp114*_tmp115 - _tmp116*_tmp46 - 24;
    _result(2, 6) = _tmp116*_tmp49;
    _result(3, 6) = _tmp111*_tmp19 + _tmp113*_tmp86 - _tmp114*_tmp117 - 6*_tmp51;
    _result(4, 6) = -_tmp111*_tmp20 - _tmp4*_tmp68*std::pow(x3, Scalar(2));
    _result(5, 6) = -_tmp111*_tmp30 - _tmp112 - 24;
    _result(6, 6) = _tmp1*_tmp116*_tmp50 - _tmp11*_tmp113 - _tmp111*_tmp61 - _tmp17 - _tmp21 - 12*_tmp59*x4 - 2;
    _result(7, 6) = -_tmp111*_tmp32 - _tmp113*_tmp99 + _tmp116*_tmp63 + _tmp119*(_tmp110 + _tmp17);
    _result(8, 6) = 0;
    _result(9, 6) = -_tmp120 + _tmp121 - _tmp33*x3 + Scalar(4.0);
    _result(10, 6) = _tmp114*_tmp67 + _tmp116*_tmp68;
    _result(0, 7) = -_tmp122*_tmp3;
    _result(1, 7) = _tmp10*_tmp122 - _tmp123*_tmp75 + _tmp124*_tmp39 - _tmp125*_tmp46;
    _result(2, 7) = _tmp125*_tmp49;
    _result(3, 7) = -_tmp122*_tmp19 + _tmp123*_tmp86 - _tmp124*_tmp51 - _tmp127*_tmp84 + _tmp24*_tmp36*_tmp6;
    _result(4, 7) = _tmp122*_tmp20 - _tmp123*_tmp87;
    _result(5, 7) = -_tmp121*_tmp89 + _tmp122*_tmp30 - _tmp124*_tmp57 + _tmp25*_tmp88;
    _result(6, 7) = -_tmp11*_tmp123 + _tmp122*_tmp61 - _tmp124*_tmp60 + _tmp125*_tmp62;
    _result(7, 7) = -_tmp1*_tmp126*_tmp83 + _tmp110*_tmp24*_tmp92 - _tmp119*_tmp88 + _tmp120*_tmp54*x4 + _tmp122*_tmp32 + _tmp123*_tmp98 + _tmp125*_tmp63 - 3*x1;
    _result(8, 7) = _tmp100*_tmp127;
    _result(9, 7) = _tmp124*_tmp66 + _tmp64 + Scalar(1.0)*x2;
    _result(10, 7) = _tmp101*_tmp127 + _tmp125*_tmp68;
    _result(0, 8) = 0;
    _result(1, 8) = 0;
    _result(2, 8) = 0;
    _result(3, 8) = -_tmp128*_tmp84;
    _result(4, 8) = 0;
    _result(5, 8) = 0;
    _result(6, 8) = 0;
    _result(7, 8) = -_tmp128*_tmp93;
    _result(8, 8) = _tmp100*_tmp128;
    _result(9, 8) = 0;
    _result(10, 8) = _tmp101*_tmp128;
    _result(0, 9) = _tmp95;
    _result(1, 9) = -_tmp10*_tmp130 - _tmp103*_tmp39 - _tmp129*_tmp38;
    _result(2, 9) = -_tmp129*_tmp47;
    _result(3, 9) = _tmp130*_tmp19 + _tmp22*_tmp50*x4;
    _result(4, 9) = -_tmp130*_tmp20;
    _result(5, 9) = _tmp103*_tmp57 + _tmp129*_tmp56 - _tmp29*_tmp4*_tmp41;
    _result(6, 9) = _tmp11*_tmp34*_tmp59 - _tmp130*_tmp61;
    _result(7, 9) = -_tmp130*_tmp32;
    _result(8, 9) = 0;
    _result(9, 9) = -_tmp103*_tmp66 + _tmp129*_tmp65 - _tmp33*x2;
    _result(10, 9) = 0;
    _result(0, 10) = 0;
    _result(1, 10) = _tmp115 + _tmp131*_tmp46;
    _result(2, 10) = -_tmp131*_tmp49;
    _result(3, 10) = _tmp117 - _tmp132*_tmp84;
    _result(4, 10) = 0;
    _result(5, 10) = 0;
    _result(6, 10) = -_tmp131*_tmp62 - 1;
    _result(7, 10) = -_tmp131*_tmp63 - _tmp132*_tmp93;
    _result(8, 10) = _tmp100*_tmp132;
    _result(9, 10) = 0;
    _result(10, 10) = _tmp101*_tmp132 - _tmp131*_tmp68 - _tmp67;

    return _result;
}  // NOLINT(readability/fn_size)

// NOLINTNEXTLINE(readability/fn_size)
}  // namespace sym
// -----------------------------------------------------------------------------
// This file was autogenerated by symforce from template:
//     function/FUNCTION.h.jinja
// Do NOT modify by hand.
// -----------------------------------------------------------------------------

#pragma once


#include <Eigen/Core>
#include <Eigen/SparseCore>



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
 *     result: Matrix36_84
 */
template <typename Scalar>
Eigen::SparseMatrix<Scalar> ComputeABibd93(const Scalar x0, const Scalar x1, const Scalar x2, const Scalar x3, const Scalar x4) {

    // Total ops: 454

    // Input arrays

    // Intermediate terms (92)
    const Scalar _tmp0 = x4 + 1;
    const Scalar _tmp1 = x0*x1;
    const Scalar _tmp2 = -x2;
    const Scalar _tmp3 = _tmp2 + x0;
    const Scalar _tmp4 = x1*x3;
    const Scalar _tmp5 = 2*x3;
    const Scalar _tmp6 = x3 + x4;
    const Scalar _tmp7 = (Scalar(1)/Scalar(2))*x4;
    const Scalar _tmp8 = -x0;
    const Scalar _tmp9 = Scalar(1.0) / (_tmp0);
    const Scalar _tmp10 = x2 + 2;
    const Scalar _tmp11 = 2*x1;
    const Scalar _tmp12 = 2*x2;
    const Scalar _tmp13 = _tmp12 - 2;
    const Scalar _tmp14 = _tmp10 + _tmp8;
    const Scalar _tmp15 = x0 + 1;
    const Scalar _tmp16 = x0 + x3;
    const Scalar _tmp17 = Scalar(1.0) / (x2);
    const Scalar _tmp18 = x3 - 1;
    const Scalar _tmp19 = _tmp18*x0;
    const Scalar _tmp20 = Scalar(1.0) / (x3);
    const Scalar _tmp21 = -x3;
    const Scalar _tmp22 = _tmp21 + x2;
    const Scalar _tmp23 = (Scalar(1)/Scalar(2))*x3;
    const Scalar _tmp24 = 2*x0;
    const Scalar _tmp25 = x0 + x1;
    const Scalar _tmp26 = x2 - 2;
    const Scalar _tmp27 = std::pow(x1, Scalar(2));
    const Scalar _tmp28 = 2*_tmp27;
    const Scalar _tmp29 = Scalar(1.0) / (x1);
    const Scalar _tmp30 = -x4;
    const Scalar _tmp31 = 4*x2;
    const Scalar _tmp32 = -_tmp31;
    const Scalar _tmp33 = x4 - 1;
    const Scalar _tmp34 = 2*_tmp17;
    const Scalar _tmp35 = x3 + 1;
    const Scalar _tmp36 = x0*x3;
    const Scalar _tmp37 = x2*x4;
    const Scalar _tmp38 = -x1;
    const Scalar _tmp39 = x4 + 2;
    const Scalar _tmp40 = _tmp11*x4;
    const Scalar _tmp41 = _tmp17*x1;
    const Scalar _tmp42 = x1 + x4;
    const Scalar _tmp43 = _tmp42*x2;
    const Scalar _tmp44 = x0 - 4;
    const Scalar _tmp45 = _tmp18*x1;
    const Scalar _tmp46 = Scalar(1.0) / (x0);
    const Scalar _tmp47 = x1*x4;
    const Scalar _tmp48 = _tmp12 + 2;
    const Scalar _tmp49 = x1 + 2;
    const Scalar _tmp50 = x0 + x4;
    const Scalar _tmp51 = std::pow(x4, Scalar(2));
    const Scalar _tmp52 = _tmp30 + x2;
    const Scalar _tmp53 = _tmp52*x2;
    const Scalar _tmp54 = x1 + 1;
    const Scalar _tmp55 = x2 + 1;
    const Scalar _tmp56 = x2 + x3;
    const Scalar _tmp57 = Scalar(1.0) / (x4);
    const Scalar _tmp58 = _tmp57*x2;
    const Scalar _tmp59 = _tmp21 + x1;
    const Scalar _tmp60 = x3 + 2;
    const Scalar _tmp61 = std::pow(x0, Scalar(2));
    const Scalar _tmp62 = x1 - 2;
    const Scalar _tmp63 = x1*x2;
    const Scalar _tmp64 = x3 - 2;
    const Scalar _tmp65 = x0 + 3;
    const Scalar _tmp66 = -_tmp34;
    const Scalar _tmp67 = _tmp24 - 2;
    const Scalar _tmp68 = _tmp66 + x4;
    const Scalar _tmp69 = 2*x4;
    const Scalar _tmp70 = _tmp69 - 4;
    const Scalar _tmp71 = x2*x3;
    const Scalar _tmp72 = x0*x4;
    const Scalar _tmp73 = x3*x4;
    const Scalar _tmp74 = x1 + x2;
    const Scalar _tmp75 = x4 - 2;
    const Scalar _tmp76 = x2 - 1;
    const Scalar _tmp77 = -_tmp24;
    const Scalar _tmp78 = _tmp48 + _tmp77;
    const Scalar _tmp79 = x0 - 2;
    const Scalar _tmp80 = -_tmp69;
    const Scalar _tmp81 = std::pow(x2, Scalar(2));
    const Scalar _tmp82 = x0 - 1;
    const Scalar _tmp83 = _tmp24 + 2;
    const Scalar _tmp84 = x1 - 1;
    const Scalar _tmp85 = -_tmp12;
    const Scalar _tmp86 = _tmp30 + x0;
    const Scalar _tmp87 = _tmp69 + 4;
    const Scalar _tmp88 = _tmp4*x0;
    const Scalar _tmp89 = x0*x2;
    const Scalar _tmp90 = 4*x4;
    const Scalar _tmp91 = _tmp24*x1;

    // Output terms (1)
    static constexpr int kRows_result = 36;
    static constexpr int kCols_result = 84;
    static constexpr int kNumNonZero_result = 250;
    static constexpr int kColPtrs_result[] = {0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60, 63, 66, 69, 72, 75, 78, 81, 84, 86, 89, 92, 95, 98, 101, 104, 107, 110, 113, 116, 119, 122, 125, 128, 131, 134, 137, 140, 143, 146, 149, 152, 155, 158, 161, 164, 167, 170, 172, 175, 178, 181, 184, 187, 190, 193, 196, 199, 202, 205, 208, 211, 214, 217, 220, 223, 226, 229, 232, 235, 238, 241, 244, 247, 250};
    static constexpr int kRowIndices_result[] = {0, 1, 8, 0, 2, 9, 0, 3, 10, 0, 4, 11, 0, 5, 12, 0, 6, 13, 0, 7, 14, 1, 2, 15, 1, 3, 16, 1, 4, 17, 1, 5, 18, 1, 6, 19, 1, 7, 20, 2, 3, 21, 2, 4, 22, 2, 5, 23, 2, 6, 24, 2, 7, 25, 3, 4, 26, 3, 5, 27, 3, 6, 28, 3, 7, 29, 4, 5, 30, 4, 6, 31, 4, 7, 32, 5, 6, 33, 5, 7, 34, 6, 7, 35, 8, 9, 8, 10, 16, 8, 11, 17, 8, 12, 18, 8, 13, 19, 8, 14, 20, 9, 10, 21, 9, 11, 22, 9, 12, 23, 9, 13, 24, 9, 14, 25, 10, 11, 26, 10, 12, 27, 10, 13, 28, 10, 14, 29, 11, 12, 30, 11, 13, 31, 11, 14, 32, 12, 13, 33, 12, 14, 34, 13, 14, 35, 15, 16, 21, 15, 17, 22, 15, 18, 23, 15, 19, 24, 15, 20, 25, 16, 17, 26, 16, 18, 27, 16, 19, 28, 20, 29, 17, 18, 30, 17, 19, 31, 17, 20, 32, 18, 19, 33, 18, 20, 34, 19, 20, 35, 21, 22, 26, 21, 23, 27, 21, 24, 28, 21, 25, 29, 22, 23, 30, 22, 24, 31, 22, 25, 32, 23, 24, 33, 23, 25, 34, 24, 25, 35, 26, 27, 30, 26, 28, 31, 26, 29, 32, 27, 28, 33, 27, 29, 34, 28, 29, 35, 30, 31, 33, 30, 32, 34, 31, 32, 35, 33, 34, 35};
    Scalar result_empty_value_ptr[250];
    Eigen::SparseMatrix<Scalar> result = Eigen::Map<const Eigen::SparseMatrix<Scalar>>(
        kRows_result,
        kCols_result,
        kNumNonZero_result,
        kColPtrs_result,
        kRowIndices_result,
        result_empty_value_ptr
    );
    Scalar* result_value_ptr = result.valuePtr();


    result_value_ptr[0] = _tmp0;
    result_value_ptr[1] = -Scalar(1)/Scalar(4)*_tmp1;
    result_value_ptr[2] = -_tmp3 - 4;
    result_value_ptr[3] = _tmp4;
    result_value_ptr[4] = -_tmp5*(x3 - 3) + 1;
    result_value_ptr[5] = -x3*(_tmp2 + _tmp6);
    result_value_ptr[6] = -x4*(_tmp7 + (Scalar(1)/Scalar(2))*x2);
    result_value_ptr[7] = _tmp5 + _tmp8 + 2;
    result_value_ptr[8] = _tmp9*x0 + x3;
    result_value_ptr[9] = -_tmp10*_tmp11 + _tmp10;
    result_value_ptr[10] = _tmp13 + x3;
    result_value_ptr[11] = -_tmp14;
    result_value_ptr[12] = -_tmp15/_tmp16;
    result_value_ptr[13] = _tmp5 + 6*x2 - 4;
    result_value_ptr[14] = 8*x0;
    result_value_ptr[15] = -x4*(-_tmp17 + x1) + 1;
    result_value_ptr[16] = -_tmp19;
    result_value_ptr[17] = _tmp20*(_tmp22 + 3);
    result_value_ptr[18] = -_tmp23*x0 + _tmp7;
    result_value_ptr[19] = 6 - _tmp24;
    result_value_ptr[20] = _tmp25*x2 + x0;
    result_value_ptr[21] = _tmp26;
    result_value_ptr[22] = -_tmp28 - 4;
    result_value_ptr[23] = -_tmp11 - 2;
    result_value_ptr[24] = _tmp5*x0 - 2;
    result_value_ptr[25] = _tmp29 + x1;
    result_value_ptr[26] = -_tmp23 - _tmp30;
    result_value_ptr[27] = x2*(_tmp6 + x2) + x4;
    result_value_ptr[28] = _tmp32;
    result_value_ptr[29] = 2;
    result_value_ptr[30] = _tmp0;
    result_value_ptr[31] = -_tmp33*x2;
    result_value_ptr[32] = _tmp10 + _tmp34 + _tmp5;
    result_value_ptr[33] = -_tmp35;
    result_value_ptr[34] = x0 - x1*(_tmp7 + Scalar(-1)/Scalar(2));
    result_value_ptr[35] = _tmp35*_tmp36;
    result_value_ptr[36] = _tmp30 + _tmp37;
    result_value_ptr[37] = -_tmp3*x1;
    result_value_ptr[38] = -_tmp13;
    result_value_ptr[39] = -x2*(_tmp21 + _tmp25 - 2);
    result_value_ptr[40] = x4;
    result_value_ptr[41] = -_tmp37*(_tmp38 + _tmp39);
    result_value_ptr[42] = -_tmp6 - 2;
    result_value_ptr[43] = -_tmp17*_tmp40;
    result_value_ptr[44] = -_tmp2 - Scalar(1)/Scalar(2)*_tmp41;
    result_value_ptr[45] = _tmp43;
    result_value_ptr[46] = _tmp30 + _tmp44*x3;
    result_value_ptr[47] = _tmp2 + _tmp21 + _tmp42;
    result_value_ptr[48] = _tmp11;
    result_value_ptr[49] = _tmp35*x1;
    result_value_ptr[50] = _tmp45;
    result_value_ptr[51] = -_tmp46*_tmp47 - 2;
    result_value_ptr[52] = -_tmp16*x3;
    result_value_ptr[53] = x1*(_tmp47 + x1);
    result_value_ptr[54] = -_tmp48*x4;
    result_value_ptr[55] = -_tmp49*_tmp9 + 1;
    result_value_ptr[56] = -_tmp6;
    result_value_ptr[57] = x2/(_tmp50 + 2);
    result_value_ptr[58] = _tmp51;
    result_value_ptr[59] = _tmp27*x3 + x2;
    result_value_ptr[60] = -_tmp0;
    result_value_ptr[61] = 1;
    result_value_ptr[62] = -_tmp41 - 1;
    result_value_ptr[63] = -_tmp53;
    result_value_ptr[64] = Scalar(2.0);
    result_value_ptr[65] = _tmp36 + _tmp54;
    result_value_ptr[66] = Scalar(1.0)*_tmp55*x0 + 2;
    result_value_ptr[67] = -_tmp11 - _tmp32;
    result_value_ptr[68] = -_tmp12*x4;
    result_value_ptr[69] = x3*(_tmp42 - 2);
    result_value_ptr[70] = -_tmp56*_tmp58;
    result_value_ptr[71] = _tmp59*x1 + x1;
    result_value_ptr[72] = _tmp37;
    result_value_ptr[73] = _tmp60 + _tmp8;
    result_value_ptr[74] = x0 - 3*x1 + 2;
    result_value_ptr[75] = x1*(_tmp10 + x0);
    result_value_ptr[76] = -_tmp22;
    result_value_ptr[77] = -_tmp5*_tmp60;
    result_value_ptr[78] = -_tmp45*_tmp61;
    result_value_ptr[79] = _tmp37*_tmp46;
    result_value_ptr[80] = -x2*(_tmp24 + x4);
    result_value_ptr[81] = _tmp62;
    result_value_ptr[82] = -_tmp38 - _tmp63;
    result_value_ptr[83] = (Scalar(1)/Scalar(2))*x0;
    result_value_ptr[84] = _tmp64*x3 + x1;
    result_value_ptr[85] = -_tmp65*x1 + x0;
    result_value_ptr[86] = _tmp2 + _tmp25 - 3;
    result_value_ptr[87] = _tmp66 + 1;
    result_value_ptr[88] = x4;
    result_value_ptr[89] = -_tmp20*_tmp67;
    result_value_ptr[90] = x1*(_tmp15 + x2);
    result_value_ptr[91] = _tmp68;
    result_value_ptr[92] = _tmp24 + _tmp5 - 3;
    result_value_ptr[93] = _tmp24 + _tmp70;
    result_value_ptr[94] = -_tmp21 - _tmp71;
    result_value_ptr[95] = _tmp44;
    result_value_ptr[96] = _tmp27;
    result_value_ptr[97] = -_tmp12 - x3;
    result_value_ptr[98] = -_tmp10*x0;
    result_value_ptr[99] = _tmp11 + x2 - 3;
    result_value_ptr[100] = -_tmp60;
    result_value_ptr[101] = _tmp40;
    result_value_ptr[102] = _tmp50 - _tmp72;
    result_value_ptr[103] = 4*_tmp36;
    result_value_ptr[104] = -_tmp30 - _tmp73 - 2;
    result_value_ptr[105] = _tmp47;
    result_value_ptr[106] = x2;
    result_value_ptr[107] = -_tmp20*_tmp72*_tmp74;
    result_value_ptr[108] = x3 - x4/_tmp49;
    result_value_ptr[109] = _tmp30 + _tmp5;
    result_value_ptr[110] = _tmp17*_tmp24*_tmp47;
    result_value_ptr[111] = x2/(x0 - 2*std::pow(x3, Scalar(2)));
    result_value_ptr[112] = _tmp66 + 2;
    result_value_ptr[113] = _tmp35 + x2;
    result_value_ptr[114] = -_tmp5*x4 - x0;
    result_value_ptr[115] = -_tmp50;
    result_value_ptr[116] = _tmp20*(_tmp0 + 2*_tmp20);
    result_value_ptr[117] = Scalar(1.0)/_tmp75;
    result_value_ptr[118] = _tmp75*x4;
    result_value_ptr[119] = _tmp31 - 1;
    result_value_ptr[120] = -_tmp12*(_tmp12 - 1);
    result_value_ptr[121] = _tmp53 + 2;
    result_value_ptr[122] = -_tmp52 - _tmp65;
    result_value_ptr[123] = _tmp67;
    result_value_ptr[124] = _tmp1 + _tmp11 + _tmp30;
    result_value_ptr[125] = _tmp64;
    result_value_ptr[126] = -_tmp69 - 2;
    result_value_ptr[127] = -_tmp76;
    result_value_ptr[128] = -_tmp49;
    result_value_ptr[129] = -_tmp47;
    result_value_ptr[130] = _tmp30;
    result_value_ptr[131] = _tmp57*x1;
    result_value_ptr[132] = -_tmp25 - 2;
    result_value_ptr[133] = _tmp50*x3 - _tmp54;
    result_value_ptr[134] = _tmp78;
    result_value_ptr[135] = -_tmp35 + _tmp36 - x1;
    result_value_ptr[136] = _tmp52*x1;
    result_value_ptr[137] = _tmp76*x4 + _tmp79;
    result_value_ptr[138] = _tmp26;
    result_value_ptr[139] = _tmp78;
    result_value_ptr[140] = _tmp47;
    result_value_ptr[141] = _tmp26 + x1;
    result_value_ptr[142] = -x3*(_tmp56 + x1);
    result_value_ptr[143] = -_tmp5;
    result_value_ptr[144] = _tmp20*_tmp47;
    result_value_ptr[145] = _tmp12*_tmp14 + 4;
    result_value_ptr[146] = _tmp52;
    result_value_ptr[147] = -x0/(_tmp12 - 4);
    result_value_ptr[148] = -_tmp69*_tmp81 - _tmp80;
    result_value_ptr[149] = x4*(_tmp10 + x3);
    result_value_ptr[150] = _tmp55;
    result_value_ptr[151] = -_tmp15*x4;
    result_value_ptr[152] = -_tmp82;
    result_value_ptr[153] = _tmp56 + _tmp8;
    result_value_ptr[154] = -_tmp5 - 6;
    result_value_ptr[155] = -_tmp26;
    result_value_ptr[156] = _tmp83;
    result_value_ptr[157] = -_tmp39*x3 + x2;
    result_value_ptr[158] = _tmp73 + _tmp84;
    result_value_ptr[159] = -_tmp49 - _tmp85;
    result_value_ptr[160] = -_tmp86;
    result_value_ptr[161] = -_tmp29*_tmp36*_tmp86 - 1;
    result_value_ptr[162] = _tmp60*_tmp84;
    result_value_ptr[163] = _tmp33*_tmp72;
    result_value_ptr[164] = 4;
    result_value_ptr[165] = _tmp1;
    result_value_ptr[166] = -2;
    result_value_ptr[167] = _tmp79*x4;
    result_value_ptr[168] = -_tmp67;
    result_value_ptr[169] = -_tmp29*_tmp5;
    result_value_ptr[170] = -_tmp24*_tmp64 + 1;
    result_value_ptr[171] = _tmp11 + _tmp80;
    result_value_ptr[172] = -_tmp33*_tmp46 + x2;
    result_value_ptr[173] = -_tmp39*(x3 + Scalar(-1.0));
    result_value_ptr[174] = _tmp25;
    result_value_ptr[175] = -_tmp29*_tmp57*x0;
    result_value_ptr[176] = _tmp63 + 1;
    result_value_ptr[177] = -x0/_tmp62;
    result_value_ptr[178] = _tmp19 + 1;
    result_value_ptr[179] = -_tmp70;
    result_value_ptr[180] = -_tmp59*x0*(x0 + x2);
    result_value_ptr[181] = -2;
    result_value_ptr[182] = _tmp12*_tmp72;
    result_value_ptr[183] = -3;
    result_value_ptr[184] = -_tmp39*x4;
    result_value_ptr[185] = -_tmp77 - _tmp87;
    result_value_ptr[186] = _tmp12*_tmp47;
    result_value_ptr[187] = _tmp88;
    result_value_ptr[188] = _tmp6 - 2;
    result_value_ptr[189] = _tmp81 - _tmp89 + x3;
    result_value_ptr[190] = -_tmp12 - x4;
    result_value_ptr[191] = 3 - _tmp63;
    result_value_ptr[192] = x1;
    result_value_ptr[193] = _tmp27*_tmp87;
    result_value_ptr[194] = x2*(_tmp2 + x1);
    result_value_ptr[195] = _tmp11*(x2 + x4);
    result_value_ptr[196] = _tmp89;
    result_value_ptr[197] = -x3/(_tmp69 + 1);
    result_value_ptr[198] = -_tmp81*x1;
    result_value_ptr[199] = _tmp24 - 1;
    result_value_ptr[200] = _tmp88 + x3;
    result_value_ptr[201] = -_tmp11;
    result_value_ptr[202] = _tmp80 + _tmp85 + x1 + 4;
    result_value_ptr[203] = _tmp74;
    result_value_ptr[204] = _tmp35;
    result_value_ptr[205] = -2*_tmp57 + x4;
    result_value_ptr[206] = -_tmp90;
    result_value_ptr[207] = x1 + 3;
    result_value_ptr[208] = 3*x2 + 4;
    result_value_ptr[209] = _tmp24 - 4;
    result_value_ptr[210] = _tmp2 + _tmp83;
    result_value_ptr[211] = -_tmp68*x3 - 1;
    result_value_ptr[212] = -x4*(_tmp76*x3 + 1);
    result_value_ptr[213] = _tmp34/_tmp64;
    result_value_ptr[214] = -_tmp18*_tmp4;
    result_value_ptr[215] = _tmp6 - _tmp63*x0;
    result_value_ptr[216] = _tmp43;
    result_value_ptr[217] = _tmp70;
    result_value_ptr[218] = -_tmp11 - _tmp8 - x4;
    result_value_ptr[219] = _tmp21 + _tmp91;
    result_value_ptr[220] = _tmp33 + x2;
    result_value_ptr[221] = _tmp38 + _tmp6;
    result_value_ptr[222] = Scalar(-0.5);
    result_value_ptr[223] = -6;
    result_value_ptr[224] = _tmp60*x2;
    result_value_ptr[225] = -_tmp20*_tmp26 - x3;
    result_value_ptr[226] = -_tmp10*_tmp46 - x1;
    result_value_ptr[227] = x2*(x0 - 12);
    result_value_ptr[228] = _tmp91;
    result_value_ptr[229] = -x4/(_tmp37*x1 + x2);
    result_value_ptr[230] = -_tmp47*_tmp82;
    result_value_ptr[231] = _tmp90 - 8;
    result_value_ptr[232] = -_tmp40 + x0;
    result_value_ptr[233] = Scalar(-1.5);
    result_value_ptr[234] = _tmp38;
    result_value_ptr[235] = -_tmp34*_tmp61;
    result_value_ptr[236] = _tmp11;
    result_value_ptr[237] = -_tmp21 - _tmp28;
    result_value_ptr[238] = -x2 - 1/(_tmp51 + 1);
    result_value_ptr[239] = _tmp52;
    result_value_ptr[240] = _tmp10;
    result_value_ptr[241] = _tmp58 - 1;
    result_value_ptr[242] = -_tmp0*x0 - _tmp2;
    result_value_ptr[243] = _tmp89 + x1 + x3;
    result_value_ptr[244] = -_tmp4*(_tmp16 + 2);
    result_value_ptr[245] = Scalar(1.0) / (_tmp33);
    result_value_ptr[246] = -_tmp74*x3;
    result_value_ptr[247] = _tmp18*_tmp71;
    result_value_ptr[248] = _tmp66;
    result_value_ptr[249] = _tmp17*_tmp73;

    return result;
}  // NOLINT(readability/fn_size)

// NOLINTNEXTLINE(readability/fn_size)
}  // namespace sym

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
 *     result: Matrix20_15
 */
template <typename Scalar>
Eigen::SparseMatrix<Scalar> ComputeAN3C4B2(const Scalar x0, const Scalar x1, const Scalar x2, const Scalar x3, const Scalar x4) {

    // Total ops: 133

    // Input arrays

    // Intermediate terms (23)
    const Scalar _tmp0 = x1*x2;
    const Scalar _tmp1 = x1 + 2;
    const Scalar _tmp2 = -x1;
    const Scalar _tmp3 = _tmp2 + x4;
    const Scalar _tmp4 = 4*x4;
    const Scalar _tmp5 = -x4;
    const Scalar _tmp6 = std::pow(x2, Scalar(2));
    const Scalar _tmp7 = 2*_tmp6;
    const Scalar _tmp8 = 2*x0;
    const Scalar _tmp9 = x3*x4;
    const Scalar _tmp10 = x1 - 2;
    const Scalar _tmp11 = -x3;
    const Scalar _tmp12 = 4*x1;
    const Scalar _tmp13 = x0*x3;
    const Scalar _tmp14 = (Scalar(1)/Scalar(2))*x0;
    const Scalar _tmp15 = 2*x1;
    const Scalar _tmp16 = x3 + x4;
    const Scalar _tmp17 = x2 - 1;
    const Scalar _tmp18 = 2*x4;
    const Scalar _tmp19 = x1 - 1;
    const Scalar _tmp20 = _tmp8 + x2;
    const Scalar _tmp21 = -_tmp18;
    const Scalar _tmp22 = x1*x3;

    // Output terms (1)
    static constexpr int kRows_result = 20;
    static constexpr int kCols_result = 15;
    static constexpr int kNumNonZero_result = 60;
    static constexpr int kColPtrs_result[] = {0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60};
    static constexpr int kRowIndices_result[] = {9, 15, 18, 19, 8, 14, 17, 19, 7, 13, 16, 19, 6, 12, 17, 18, 5, 11, 16, 18, 4, 10, 16, 17, 3, 12, 14, 15, 2, 11, 13, 15, 1, 10, 13, 14, 0, 10, 11, 12, 3, 6, 8, 9, 2, 5, 7, 9, 1, 4, 7, 8, 0, 4, 5, 6, 0, 1, 2, 3};
    Scalar result_empty_value_ptr[60];
    Eigen::SparseMatrix<Scalar> result = Eigen::Map<const Eigen::SparseMatrix<Scalar>>(
        kRows_result,
        kCols_result,
        kNumNonZero_result,
        kColPtrs_result,
        kRowIndices_result,
        result_empty_value_ptr
    );
    Scalar* result_value_ptr = result.valuePtr();


    result_value_ptr[0] = -_tmp0;
    result_value_ptr[1] = -x4 - 6;
    result_value_ptr[2] = -_tmp1*x4;
    result_value_ptr[3] = x3*(x3 - 1) - 1;
    result_value_ptr[4] = 2;
    result_value_ptr[5] = -_tmp3 - 2;
    result_value_ptr[6] = x0*std::pow(x3, Scalar(2))*(x0 - 2);
    result_value_ptr[7] = -2/x1;
    result_value_ptr[8] = _tmp4;
    result_value_ptr[9] = -_tmp5 - x1/x2;
    result_value_ptr[10] = x2 - 2;
    result_value_ptr[11] = -_tmp5 - x0;
    result_value_ptr[12] = -_tmp7*x4;
    result_value_ptr[13] = x1 - 4;
    result_value_ptr[14] = x2 + 3;
    result_value_ptr[15] = -_tmp8*x3 + 1;
    result_value_ptr[16] = _tmp9*(x0 + x1);
    result_value_ptr[17] = _tmp7;
    result_value_ptr[18] = x3*(x0 + 7);
    result_value_ptr[19] = -_tmp10*(_tmp11 + x1);
    result_value_ptr[20] = -_tmp1;
    result_value_ptr[21] = -_tmp8 - 1;
    result_value_ptr[22] = _tmp12 + x4;
    result_value_ptr[23] = -_tmp13 - x0 - x3;
    result_value_ptr[24] = _tmp9*(1 - x2/x0);
    result_value_ptr[25] = _tmp14 - _tmp15;
    result_value_ptr[26] = -x0*(_tmp16 + 2) + 2;
    result_value_ptr[27] = -_tmp12 - _tmp4;
    result_value_ptr[28] = 2 - _tmp9/_tmp17;
    result_value_ptr[29] = -_tmp15*x3 - 2;
    result_value_ptr[30] = x0;
    result_value_ptr[31] = -_tmp10 - x3;
    result_value_ptr[32] = x3*(-_tmp18*x2 + x1);
    result_value_ptr[33] = -_tmp19*x3 - x0;
    result_value_ptr[34] = -_tmp11 - x2 - 2;
    result_value_ptr[35] = -4/(_tmp8 - 2);
    result_value_ptr[36] = -_tmp13 - 2;
    result_value_ptr[37] = _tmp14;
    result_value_ptr[38] = _tmp11 + _tmp20;
    result_value_ptr[39] = -x3*(_tmp2 + x2 + 1);
    result_value_ptr[40] = -_tmp8*(x0 + x2) - 2;
    result_value_ptr[41] = _tmp21 + 2*x2 + 2;
    result_value_ptr[42] = -_tmp16;
    result_value_ptr[43] = x1 + x2 - 3;
    result_value_ptr[44] = 2 - _tmp15;
    result_value_ptr[45] = _tmp13*x1;
    result_value_ptr[46] = -2*_tmp0*(x1 + 1);
    result_value_ptr[47] = 3;
    result_value_ptr[48] = -_tmp6*(x3 - 2);
    result_value_ptr[49] = -_tmp19;
    result_value_ptr[50] = _tmp22;
    result_value_ptr[51] = -_tmp15*_tmp9;
    result_value_ptr[52] = 1 - x4;
    result_value_ptr[53] = -_tmp3 - _tmp9*x2;
    result_value_ptr[54] = _tmp17 + x4;
    result_value_ptr[55] = -_tmp20;
    result_value_ptr[56] = -_tmp22;
    result_value_ptr[57] = _tmp19;
    result_value_ptr[58] = _tmp0*x0 + _tmp8;
    result_value_ptr[59] = -_tmp21 - x1;

    return result;
}  // NOLINT(readability/fn_size)

// NOLINTNEXTLINE(readability/fn_size)
}  // namespace sym
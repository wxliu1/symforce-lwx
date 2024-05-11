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
 *     result: Matrix11_11
 */
template <typename Scalar>
Eigen::SparseMatrix<Scalar> ComputeBTinaDiscog(const Scalar x0, const Scalar x1, const Scalar x2, const Scalar x3, const Scalar x4) {

    // Total ops: 95

    // Input arrays

    // Intermediate terms (13)
    const Scalar _tmp0 = 2*x0;
    const Scalar _tmp1 = 2*x2;
    const Scalar _tmp2 = 2*x4;
    const Scalar _tmp3 = x2 + 1;
    const Scalar _tmp4 = -x2;
    const Scalar _tmp5 = (Scalar(1)/Scalar(2))*x4;
    const Scalar _tmp6 = x4 - 2;
    const Scalar _tmp7 = -x3;
    const Scalar _tmp8 = x1*x4;
    const Scalar _tmp9 = x0 + 1;
    const Scalar _tmp10 = -x4;
    const Scalar _tmp11 = _tmp0*x3;
    const Scalar _tmp12 = x1 + x2;

    // Output terms (1)
    static constexpr int kRows_result = 11;
    static constexpr int kCols_result = 11;
    static constexpr int kNumNonZero_result = 47;
    static constexpr int kColPtrs_result[] = {0, 2, 8, 11, 18, 21, 26, 32, 40, 41, 44, 47};
    static constexpr int kRowIndices_result[] = {2, 9, 0, 2, 4, 7, 8, 9, 0, 8, 9, 1, 2, 4, 5, 6, 7, 10, 2, 5, 9, 0, 1, 2, 4, 10, 2, 3, 4, 5, 7, 8, 1, 2, 3, 4, 5, 6, 8, 10, 6, 0, 2, 4, 6, 7, 8};
    Scalar result_empty_value_ptr[47];
    Eigen::SparseMatrix<Scalar> result = Eigen::Map<const Eigen::SparseMatrix<Scalar>>(
        kRows_result,
        kCols_result,
        kNumNonZero_result,
        kColPtrs_result,
        kRowIndices_result,
        result_empty_value_ptr
    );
    Scalar* result_value_ptr = result.valuePtr();


    result_value_ptr[0] = _tmp0 + x1 - 2;
    result_value_ptr[1] = _tmp1 + 6;
    result_value_ptr[2] = _tmp0;
    result_value_ptr[3] = x0*x4;
    result_value_ptr[4] = 3 - _tmp2;
    result_value_ptr[5] = x2*x3*(x2 + 2);
    result_value_ptr[6] = -x1 - x3*x4 - 2;
    result_value_ptr[7] = -x0 + x1 - 6;
    result_value_ptr[8] = _tmp3*x0*x1;
    result_value_ptr[9] = _tmp0 + 1;
    result_value_ptr[10] = -_tmp4 - x4 - 4;
    result_value_ptr[11] = _tmp4 + x1;
    result_value_ptr[12] = _tmp5 - Scalar(1)/Scalar(2)*x0 + x3;
    result_value_ptr[13] = _tmp6 + 3*x0;
    result_value_ptr[14] = _tmp3 + _tmp7 + x0;
    result_value_ptr[15] = _tmp2 + x0;
    result_value_ptr[16] = -4*std::pow(x1, Scalar(2))*x2;
    result_value_ptr[17] = -x3 - 1;
    result_value_ptr[18] = -_tmp0*x4 + x2*x4;
    result_value_ptr[19] = -x0 - 3;
    result_value_ptr[20] = _tmp0*_tmp8;
    result_value_ptr[21] = -x1*x2 + 2;
    result_value_ptr[22] = x0*(_tmp8 + x4);
    result_value_ptr[23] = _tmp7 + x1*x3 + 2;
    result_value_ptr[24] = _tmp9*(_tmp10 + x0);
    result_value_ptr[25] = _tmp7 + x1;
    result_value_ptr[26] = _tmp11 + 2;
    result_value_ptr[27] = -_tmp11 - _tmp7;
    result_value_ptr[28] = -6;
    result_value_ptr[29] = -x3*(2*x1 + 1);
    result_value_ptr[30] = x2 + x3 - 2;
    result_value_ptr[31] = _tmp1 + _tmp7 + _tmp9;
    result_value_ptr[32] = -3;
    result_value_ptr[33] = -x0 - x2;
    result_value_ptr[34] = _tmp1 + x3;
    result_value_ptr[35] = -_tmp5*x2 + Scalar(1)/Scalar(2);
    result_value_ptr[36] = -_tmp6 - x0;
    result_value_ptr[37] = x2*(x3 + 2)/_tmp3;
    result_value_ptr[38] = 2 - 2/(x1 + 1);
    result_value_ptr[39] = x0/x4;
    result_value_ptr[40] = 6*x0 - 3;
    result_value_ptr[41] = _tmp10 + _tmp9;
    result_value_ptr[42] = _tmp0*x2;
    result_value_ptr[43] = _tmp8;
    result_value_ptr[44] = _tmp12;
    result_value_ptr[45] = -1;
    result_value_ptr[46] = -_tmp12 - 4;

    return result;
}  // NOLINT(readability/fn_size)

// NOLINTNEXTLINE(readability/fn_size)
}  // namespace sym
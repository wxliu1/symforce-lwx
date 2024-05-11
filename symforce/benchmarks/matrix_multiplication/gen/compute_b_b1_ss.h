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
 *     result: Matrix77
 */
template <typename Scalar>
Eigen::SparseMatrix<Scalar> ComputeBB1Ss(const Scalar x0, const Scalar x1, const Scalar x2, const Scalar x3, const Scalar x4) {

    // Total ops: 32

    // Input arrays

    // Intermediate terms (2)
    const Scalar _tmp0 = x0 + 2;
    const Scalar _tmp1 = 2*x0;

    // Output terms (1)
    static constexpr int kRows_result = 7;
    static constexpr int kCols_result = 7;
    static constexpr int kNumNonZero_result = 14;
    static constexpr int kColPtrs_result[] = {0, 3, 5, 7, 9, 10, 12, 14};
    static constexpr int kRowIndices_result[] = {4, 5, 6, 0, 1, 0, 2, 0, 3, 1, 2, 5, 3, 6};
    Scalar result_empty_value_ptr[14];
    Eigen::SparseMatrix<Scalar> result = Eigen::Map<const Eigen::SparseMatrix<Scalar>>(
        kRows_result,
        kCols_result,
        kNumNonZero_result,
        kColPtrs_result,
        kRowIndices_result,
        result_empty_value_ptr
    );
    Scalar* result_value_ptr = result.valuePtr();


    result_value_ptr[0] = x0*x2;
    result_value_ptr[1] = x1 + std::pow(x2, Scalar(2));
    result_value_ptr[2] = -2*(x0 - 2)*(x1 + 1);
    result_value_ptr[3] = 1 + x2/x0;
    result_value_ptr[4] = _tmp0 - x3;
    result_value_ptr[5] = x0*(x2 + 3);
    result_value_ptr[6] = -_tmp1*x3 - x3 - 2;
    result_value_ptr[7] = x2 + x4;
    result_value_ptr[8] = -_tmp0 - x2;
    result_value_ptr[9] = -1/(x3 + 4);
    result_value_ptr[10] = _tmp1 + 2;
    result_value_ptr[11] = -x1/x4;
    result_value_ptr[12] = std::pow(x0, Scalar(2))*x2;
    result_value_ptr[13] = -[&]() { const Scalar base = x1; return base * base * base; }();

    return result;
}  // NOLINT(readability/fn_size)

// NOLINTNEXTLINE(readability/fn_size)
}  // namespace sym
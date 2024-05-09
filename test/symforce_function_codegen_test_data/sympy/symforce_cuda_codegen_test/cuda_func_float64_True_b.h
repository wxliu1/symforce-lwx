// -----------------------------------------------------------------------------
// This file was autogenerated by symforce from template:
//     function/FUNCTION.h.jinja
// Do NOT modify by hand.
// -----------------------------------------------------------------------------

#pragma once

namespace sym {

/**
 * This function was autogenerated from a symbolic function. Do not modify by hand.
 *
 * Symbolic function: cuda_func
 *
 * Args:
 *     a: Scalar
 *     b: Matrix11
 *     c: Matrix31
 *     d: Matrix22
 *     e: Matrix51
 *     f: Matrix66
 *     g: DataBuffer
 *
 * Outputs:
 *     a_out: Scalar
 *     b_out: Matrix11
 *     c_out: Matrix31
 *     d_out: Matrix22
 *     e_out: Matrix51
 *     f_out: Matrix66
 */
inline __host__ __device__ void CudaFuncFloat64TrueB(
    const double a, const double1& b, const double* const __restrict__ c,
    const double* const __restrict__ d, const double* const __restrict__ e,
    const double* const __restrict__ f, const double* const __restrict__ g,
    double* const __restrict__ a_out = nullptr, double* const __restrict__ b_out = nullptr,
    double* const __restrict__ c_out = nullptr, double* const __restrict__ d_out = nullptr,
    double* const __restrict__ e_out = nullptr, double* const __restrict__ f_out = nullptr) {
  // Total ops: 36

  // Intermediate terms (0)

  // Output terms (6)
  if (a_out != nullptr) {
    *a_out = a;
  }

  if (b_out != nullptr) {
    b_out[0] = b.x;
  }

  if (c_out != nullptr) {
    c_out[0] = c[0];
    c_out[1] = c[1];
    c_out[2] = c[2];
  }

  if (d_out != nullptr) {
    d_out[0] = d[0];
    d_out[2] = d[2];
    d_out[1] = d[1];
    d_out[3] = d[3];
  }

  if (e_out != nullptr) {
    e_out[0] = e[0];
    e_out[1] = e[1];
    e_out[2] = e[2];
    e_out[3] = e[3];
    e_out[4] = e[4];
  }

  if (f_out != nullptr) {
    f_out[0] = f[0] + g[static_cast<size_t>(0)];
    f_out[6] = f[6] + g[static_cast<size_t>(0)];
    f_out[12] = f[12] + g[static_cast<size_t>(0)];
    f_out[18] = f[18] + g[static_cast<size_t>(0)];
    f_out[24] = f[24] + g[static_cast<size_t>(0)];
    f_out[30] = f[30] + g[static_cast<size_t>(0)];
    f_out[1] = f[1] + g[static_cast<size_t>(0)];
    f_out[7] = f[7] + g[static_cast<size_t>(0)];
    f_out[13] = f[13] + g[static_cast<size_t>(0)];
    f_out[19] = f[19] + g[static_cast<size_t>(0)];
    f_out[25] = f[25] + g[static_cast<size_t>(0)];
    f_out[31] = f[31] + g[static_cast<size_t>(0)];
    f_out[2] = f[2] + g[static_cast<size_t>(0)];
    f_out[8] = f[8] + g[static_cast<size_t>(0)];
    f_out[14] = f[14] + g[static_cast<size_t>(0)];
    f_out[20] = f[20] + g[static_cast<size_t>(0)];
    f_out[26] = f[26] + g[static_cast<size_t>(0)];
    f_out[32] = f[32] + g[static_cast<size_t>(0)];
    f_out[3] = f[3] + g[static_cast<size_t>(0)];
    f_out[9] = f[9] + g[static_cast<size_t>(0)];
    f_out[15] = f[15] + g[static_cast<size_t>(0)];
    f_out[21] = f[21] + g[static_cast<size_t>(0)];
    f_out[27] = f[27] + g[static_cast<size_t>(0)];
    f_out[33] = f[33] + g[static_cast<size_t>(0)];
    f_out[4] = f[4] + g[static_cast<size_t>(0)];
    f_out[10] = f[10] + g[static_cast<size_t>(0)];
    f_out[16] = f[16] + g[static_cast<size_t>(0)];
    f_out[22] = f[22] + g[static_cast<size_t>(0)];
    f_out[28] = f[28] + g[static_cast<size_t>(0)];
    f_out[34] = f[34] + g[static_cast<size_t>(0)];
    f_out[5] = f[5] + g[static_cast<size_t>(0)];
    f_out[11] = f[11] + g[static_cast<size_t>(0)];
    f_out[17] = f[17] + g[static_cast<size_t>(0)];
    f_out[23] = f[23] + g[static_cast<size_t>(0)];
    f_out[29] = f[29] + g[static_cast<size_t>(0)];
    f_out[35] = f[35] + g[static_cast<size_t>(0)];
  }
}

}  // namespace sym

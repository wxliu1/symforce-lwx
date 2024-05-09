// -----------------------------------------------------------------------------
// This file was autogenerated by symforce from template:
//     cam_package/CLASS.h.jinja
// Do NOT modify by hand.
// -----------------------------------------------------------------------------

#pragma once

#include <ostream>
#include <random>
#include <vector>

#include <Eigen/Core>

#include <sym/ops/storage_ops.h>

namespace sym {

/**
 * Autogenerated C++ implementation of `symforce.cam.spherical_camera_cal.SphericalCameraCal`.
 *
 * Kannala-Brandt camera model, where radial distortion is modeled relative to the 3D angle theta
 * off the optical axis as opposed to radius within the image plane (i.e. ATANCamera)
 *
 * I.e. the radius in the image plane as a function of the angle theta from the camera z-axis is
 * assumed to be given by::
 *
 *     r(theta) = theta + d[0] * theta^3 + d[1] * theta^5 + d[2] * theta^7 + d[3] * theta^9
 *
 * With no tangential coefficients, this model is over-parameterized in that we may scale all the
 * distortion coefficients by a constant, and the focal length by the inverse of that constant. To
 * fix this issue, we peg the first coefficient at 1. So while the distortion dimension is '4',
 * the actual total number of coeffs is 5.
 *
 * Additionally, the storage for this class includes the critical theta, the maximum angle from the
 * optical axis where projection is invertible; although the critical theta is a function of the
 * other parameters, this function requires polynomial root finding, so it should be computed
 * externally at runtime and set to the computed value.
 *
 * Paper::
 *
 *     A generic camera model and calibration method for conventional, wide-angle, and fish-eye
 * lenses Kannala, Juho; Brandt, Sami S. PAMI 2006
 *
 * This is the simpler "P9" model without any non-radially-symmetric distortion params.
 *
 * The storage for this class is:
 *
 *     [ fx fy cx cy critical_theta d0 d1 d2 d3 ]
 */
template <typename ScalarType>
class SphericalCameraCal {
 public:
  // Typedefs
  using Scalar = ScalarType;
  using Self = SphericalCameraCal<Scalar>;
  using DataVec = Eigen::Matrix<Scalar, 9, 1>;

  // Construct from focal_length, principal_point, critical_theta, and distortion_coeffs.
  SphericalCameraCal(const Eigen::Matrix<Scalar, 2, 1>& focal_length,
                     const Eigen::Matrix<Scalar, 2, 1>& principal_point,
                     const Scalar critical_theta,
                     const Eigen::Matrix<Scalar, 4, 1>& distortion_coeffs)
      : SphericalCameraCal(
            (Eigen::Matrix<Scalar, sym::StorageOps<Self>::StorageDim(), 1>() << focal_length,
             principal_point, critical_theta, distortion_coeffs)
                .finished()) {}

  /**
   * Construct from data vec
   *
   * @param normalize Project to the manifold on construction.  This ensures numerical stability as
   *     this constructor is called after each codegen operation.  Constructing from a normalized
   *     vector may be faster, e.g. with `FromStorage`.
   */
  explicit SphericalCameraCal(const DataVec& data, bool normalize = true) : data_(data) {
    (void)normalize;
  }

  // Access underlying storage as const
  inline const DataVec& Data() const {
    return data_;
  }

  // --------------------------------------------------------------------------
  // StorageOps concept
  // --------------------------------------------------------------------------

  static constexpr int32_t StorageDim() {
    return sym::StorageOps<Self>::StorageDim();
  }

  void ToStorage(Scalar* const vec) const {
    return sym::StorageOps<Self>::ToStorage(*this, vec);
  }

  static SphericalCameraCal FromStorage(const Scalar* const vec) {
    return sym::StorageOps<Self>::FromStorage(vec);
  }

  // --------------------------------------------------------------------------
  // Camera model methods
  // --------------------------------------------------------------------------

  /**
   * Return the focal length.
   */
  Eigen::Matrix<Scalar, 2, 1> FocalLength() const;

  /**
   * Return the principal point.
   */
  Eigen::Matrix<Scalar, 2, 1> PrincipalPoint() const;

  /**
   * Project a 3D point in the camera frame into 2D pixel coordinates.
   *
   * Returns:
   *     pixel: (x, y) coordinate in pixels if valid
   *     is_valid: 1 if the operation is within bounds else 0
   */
  Eigen::Matrix<Scalar, 2, 1> PixelFromCameraPoint(const Eigen::Matrix<Scalar, 3, 1>& point,
                                                   const Scalar epsilon,
                                                   Scalar* const is_valid = nullptr) const;

  /**
   * Project a 3D point in the camera frame into 2D pixel coordinates.
   *
   * Returns:
   *     pixel: (x, y) coordinate in pixels if valid
   *     is_valid: 1 if the operation is within bounds else 0
   *     pixel_D_cal: Derivative of pixel with respect to intrinsic calibration parameters
   *     pixel_D_point: Derivative of pixel with respect to point
   */
  Eigen::Matrix<Scalar, 2, 1> PixelFromCameraPointWithJacobians(
      const Eigen::Matrix<Scalar, 3, 1>& point, const Scalar epsilon,
      Scalar* const is_valid = nullptr, Eigen::Matrix<Scalar, 2, 8>* const pixel_D_cal = nullptr,
      Eigen::Matrix<Scalar, 2, 3>* const pixel_D_point = nullptr) const;

  // --------------------------------------------------------------------------
  // General Helpers
  // --------------------------------------------------------------------------

  bool IsApprox(const Self& b, const Scalar tol) const {
    // isApprox is multiplicative so we check the norm for the exact zero case
    // https://eigen.tuxfamily.org/dox/classEigen_1_1DenseBase.html#ae8443357b808cd393be1b51974213f9c
    if (b.Data() == DataVec::Zero()) {
      return Data().norm() < tol;
    }

    return Data().isApprox(b.Data(), tol);
  }

  template <typename ToScalar>
  SphericalCameraCal<ToScalar> Cast() const {
    return SphericalCameraCal<ToScalar>(Data().template cast<ToScalar>());
  }

  bool operator==(const SphericalCameraCal& rhs) const {
    return data_ == rhs.Data();
  }

 protected:
  DataVec data_;
};

// Shorthand for scalar types
using SphericalCameraCald = SphericalCameraCal<double>;
using SphericalCameraCalf = SphericalCameraCal<float>;

// Print definitions
std::ostream& operator<<(std::ostream& os, const SphericalCameraCal<double>& a);
std::ostream& operator<<(std::ostream& os, const SphericalCameraCal<float>& a);

}  // namespace sym

// Externs to reduce duplicate instantiation
extern template class sym::SphericalCameraCal<double>;
extern template class sym::SphericalCameraCal<float>;

// Concept implementations for this class
#include "./ops/spherical_camera_cal/group_ops.h"
#include "./ops/spherical_camera_cal/lie_group_ops.h"
#include "./ops/spherical_camera_cal/storage_ops.h"

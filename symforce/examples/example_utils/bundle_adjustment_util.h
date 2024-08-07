/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#pragma once

#include <Eigen/Core>

#include <lcmtypes/sym/optimizer_params_t.hpp>

#include <sym/linear_camera_cal.h>
#include <sym/posed_camera.h>
#include <sym/util/typedefs.h>
#include <symforce/opt/assert.h>

namespace sym {
namespace example_utils {

template <typename Scalar>
struct Correspondence {
  Eigen::Vector2d source_uv;
  Eigen::Vector2d target_uv;
  Scalar is_valid;
};

// Generate correspondences in the target camera given observations in the source camera.
// 给定源相机中的观测，生成目标相机里的correspondences
// 在源相机中给定的观测结果，在目标相机中生成对应关系。
template <typename Scalar>
std::vector<Correspondence<Scalar>> GenerateCorrespondences(
    const sym::PosedCamera<sym::LinearCameraCal<Scalar>>& source_cam,
    const sym::PosedCamera<sym::LinearCameraCal<Scalar>>& target_cam,
    const std::vector<std::pair<Eigen::Matrix<Scalar, 2, 1>, Scalar>> source_observations,
    std::mt19937& gen, const Scalar epsilon = 1e-12, const Scalar noise_px = 0,
    const size_t num_outliers = 0) {
  // Create correspondence for each sample
  std::vector<Correspondence<Scalar>> correspondences;
  for (const auto& observation : source_observations) {
    const auto& source_uv = observation.first;
    const auto& inverse_range = observation.second;

    Correspondence<Scalar> correspondence;
    correspondence.source_uv = source_uv;

    if (correspondences.size() < num_outliers) { // 如果num_outliers>0 生成一些outliers
      std::uniform_real_distribution<Scalar> uniform_dist_u(0, source_cam.ImageSize()(1));
      std::uniform_real_distribution<Scalar> uniform_dist_v(0, source_cam.ImageSize()(0));

      correspondence.target_uv =
          Eigen::Matrix<Scalar, 2, 1>(uniform_dist_u(gen), uniform_dist_v(gen));
      // 判断点是否在图像边界以内，返回1(in)或者0(out)    
      correspondence.is_valid = target_cam.InView(correspondence.target_uv, target_cam.ImageSize());
    } else {
      // Warp the point to the target
      // 变换点到目标图像
      const Eigen::Matrix<Scalar, 2, 1> target_uv_perfect = source_cam.WarpPixel(
          source_uv, inverse_range, target_cam, epsilon, &correspondence.is_valid);

      correspondence.target_uv = target_uv_perfect + noise_px * Random<Vector2<Scalar>>(gen);
    }

    correspondences.push_back(correspondence);
  }

  return correspondences;
}

// Sample pixel coords at the center of buckets with the given width, aligned with (0, 0).
template <typename Scalar>
std::vector<Eigen::Matrix<Scalar, 2, 1>> SampleRegularGrid(const Eigen::Vector2i& img_shape,
                                                           const int bucket_width_px) {
  std::vector<Eigen::Matrix<Scalar, 2, 1>> coords;

  int row = bucket_width_px / 2;
  while (row < img_shape(0)) {
    int col = bucket_width_px / 2;
    while (col < img_shape(1)) {
      coords.push_back(Eigen::Vector2i(row, col).cast<Scalar>());
      col += bucket_width_px;
    }
    row += bucket_width_px;
  }

  return coords;
}

// Given an image shape, pick a sparse but distributed set of points by
// generating a grid and then shuffling and randomly selecting from that set.
// 给定一个图像形状，通过生成一个网格来选择一个稀疏但分布的点集合，然后进行变换并从该集合中随机选择。
// 生成一个网格，然后洗牌并从该集合中随机选择。
template <typename Scalar>
std::vector<Eigen::Matrix<Scalar, 2, 1>> PickSourceCoordsRandomlyFromGrid(
    const Eigen::Vector2i& image_shape, const int num_coords, std::mt19937& gen,
    const Scalar bucket_width_px = 100) {
  // Sample some pixel coordinates in a grid
  std::vector<Eigen::Matrix<Scalar, 2, 1>> uv_samples =
      SampleRegularGrid<Scalar>(image_shape, bucket_width_px);

  // Pick randomly the required number of source pixels
  SYM_ASSERT(static_cast<int>(uv_samples.size()) >= num_coords);
  std::shuffle(uv_samples.begin(), uv_samples.end(), gen);

  std::vector<Eigen::Matrix<Scalar, 2, 1>> subset_uv_samples;
  std::copy_n(uv_samples.begin(), num_coords, std::back_inserter(subset_uv_samples));

  return subset_uv_samples;
}

// Sample a number of inverse ranges with a uniform distribution in range
// between the given values.
// 在给定值之间的范围内以均匀分布的若干逆范围进行抽样。
template <typename Scalar>
std::vector<Scalar> SampleInverseRanges(const size_t num, std::mt19937& gen,
                                        const bool at_infinity = false, const Scalar close_m = 2.5,
                                        const Scalar far_m = 30.0) {
  std::uniform_real_distribution<Scalar> dist =
      at_infinity ? std::uniform_real_distribution<Scalar>(0, 0)
                  : std::uniform_real_distribution<Scalar>(close_m, far_m);

  std::vector<Scalar> inverse_ranges;
  for (size_t i = 0; i < num; i++) {
    // 运算符()：此函数返回给定范围内的随机值
    // 均匀分布dist(gen)产生一个位于区间[close_m, far_m)的随机数，随机数是均匀分布的
    // 谈到分布，产生的都是一个位于定义域的随机变量
    inverse_ranges.push_back(1 / dist(gen));
  }

  return inverse_ranges;
}

inline optimizer_params_t OptimizerParams() {
  optimizer_params_t params{};
  params.iterations = 50;
  params.verbose = true;
  params.initial_lambda = 1.0;
  params.lambda_update_type = sym::lambda_update_type_t::STATIC;
  params.lambda_up_factor = 10.0;
  params.lambda_down_factor = 1 / 10.0;
  params.lambda_lower_bound = 1.0e-8;
  params.lambda_upper_bound = 1000000.0;
  params.early_exit_min_reduction = 1.0e-6;
  params.use_unit_damping = true;
  params.use_diagonal_damping = false;
  params.keep_max_diagonal_damping = false;
  params.diagonal_damping_min = 1e-6;
  params.enable_bold_updates = false;
  return params;
}

}  // namespace example_utils
}  // namespace sym

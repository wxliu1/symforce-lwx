/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#include "./build_example_state.h"

#include <sym/linear_camera_cal.h>
#include <sym/posed_camera.h>
#include <sym/util/typedefs.h>
#include <symforce/examples/example_utils/bundle_adjustment_util.h>
#include <symforce/opt/assert.h>
#include <symforce/opt/util.h>

namespace bundle_adjustment {

namespace {

// std::mt19937是C++标准库中的一个伪随机数生成器类
// 一个有着19937位状态大小的能够生成32位数的梅森旋转伪随机生成器

/*
 * 关于正态分布函数的用法: std::normal_distribution<type> distribution
  // With normal distribution, we can create events like "has a great chance
  // to get medium result, and has a little chance to get bad or too good result"
  
  std::default_random_engine generator;
  std::normal_distribution<double> distribution(5.0,2.0);
  generator.seed(clock()); // std pesudo also needs seed to avoid generating fixed random value.

  double number = distribution(generator);
*/    

/**
 * Build two posed cameras and put them into values, with similar but not identical poses looking
 * into the same general area and identical calibrations.  The ground truth posed cameras are
 * returned; the first view is inserted into the Values with its ground truth value, and the second
 * is inserted with some noise on the pose.
 */
/*
 * 建立两个PosedCamera对象，并存储于Values容器，它们具有类似但不相等的位姿，相同的区域(分辨率)和内参。
 * 该函数返回posed cameras的ground truth.
 * 第一个插入Valuesd的view, 位姿具有ground truth 真值；
 * 第二个插入Values的view，位姿带有噪声。
 */
std::vector<sym::PosedCamera<sym::LinearCameraCald>> AddViews(
    const BundleAdjustmentProblemParams& params, std::mt19937& gen, sym::Valuesd* const values) {
  // 随机生成相机位姿
  const sym::Pose3d view0 = sym::Random<sym::Pose3d>(gen);
  // 给定一个扰动
  sym::Vector6d perturbation;
  perturbation << 0.1, -0.2, 0.1, 2.1, 0.4, -0.2;
  // 用给定的vector对view0进行扰动得到view1
  // 其中扰动vector为扰动小量乘以服从正态分布的随机值，正态分布（高斯分布）的期望是0，标准差为params.pose_difference_std
  // 因为期望是0，这意味着大概率产生一个服从高斯分布的小量（在0附近），
  // 因而view1大概率是view0经过一个小的扰动得到的
  const sym::Pose3d view1 = view0.Retract(
      perturbation * std::normal_distribution<double>(0, params.pose_difference_std)(gen));

  // sym::LinearCameraCald为sym::LinearCameraCal<double>类型
  // 针孔相机模型Standard pinhole camera w/ four parameters [fx, fy, cx, cy].
  // 内参: (fx, fy) representing focal length; (cx, cy) representing principal point.
  // 对应的函数原型inline LinearCameraCal(const Eigen::Matrix<Scalar, 2, 1> &focal_length, const Eigen::Matrix<Scalar, 2, 1> &principal_point)
  const sym::LinearCameraCald camera_cal(Eigen::Vector2d(740, 740), Eigen::Vector2d(639.5, 359.5));
  // sym::PosedCamera是一个相机类: Camera with a given pose, camera calibration, and an optionally specified image size.
  const sym::PosedCamera<sym::LinearCameraCald> cam0(view0, camera_cal, params.image_shape);
  const sym::PosedCamera<sym::LinearCameraCald> cam1(view1, camera_cal, params.image_shape);

  // 函数原型: std::enable_if_t<!kIsEigenType<T>, bool> Set(const Key &key, const T &value)
  // 添加或者更新值: Add or update a value by key. Returns true if added, false if updated.
  values->Set({Var::VIEW, 0}, cam0.Pose());
  // cam0.Calibration()返回的是sym::LinearCameraCald类型的对象
  values->Set({Var::CALIBRATION, 0}, cam0.Calibration());

  values->Set({Var::VIEW, 1},
              cam1.Pose().Retract(params.pose_noise * sym::Random<sym::Vector6d>(gen)));
  values->Set({Var::CALIBRATION, 1}, cam1.Calibration());

  return {cam0, cam1};
}

/**
 * Add Gaussian priors on the relative poses between each pair of views.  Most of the priors have 0
 * weight in this example, and therefore have no effect.  The priors between sequential views are
 * set to be the actual transform between the ground truth poses, plus some noise.
 */
// 添加高斯先验到每对视图(view)之间的相对位姿上. 在本例中大部分先验有0权重，因此没有作用。
// 在连续的views之间的先验被设置为ground truth 位姿之间的实际变换，添加一些噪声
// 序列视图之间的先验被设置为地面真实姿态之间的实际变换，再加上一些噪声。 
void AddPosePriors(const std::vector<sym::PosedCamera<sym::LinearCameraCald>>& cams,
                   const BundleAdjustmentProblemParams& params, std::mt19937& gen,
                   sym::Valuesd* const values) {
  // First, create the 0-weight priors:
  // 首先，创建0权重先验
  for (int i = 0; i < params.num_views; i++) {
    for (int j = 0; j < params.num_views; j++) {
      // 设置先验位姿的变换：单位阵加平移(0, 0, 0)
      values->Set({Var::POSE_PRIOR_T, i, j}, sym::Pose3d());
      // 设置信息矩阵的平方根，为 6*6 的0矩阵，此即为权重
      values->Set({Var::POSE_PRIOR_SQRT_INFO, i, j}, sym::Matrix66d::Zero());
    }
  }

  // Now, the actual priors between sequential views:
  // 现在，序列视图之间的实际先验：
  for (int i = 0; i < params.num_views - 1; i++) {
    // 取出连续两帧的位姿
    const auto& first_view = cams[i].Pose();
    const auto& second_view = cams[i + 1].Pose();
    // 设置位姿先验变换：first_view到second_view的增量四元数，由 q_2 = q_1 * delta_q , q_2 = q_1 * q_{12}
    // 可知：delta_q = q_{12}
    // 然后再加入噪声，对位姿增量做一个扰动
    values->Set({Var::POSE_PRIOR_T, i, i + 1},
                first_view.Between(second_view)
                    .Retract(params.pose_prior_noise * sym::Random<sym::Vector6d>(gen)));
    // 设置先验信息矩阵：单位阵除以噪声
    values->Set({Var::POSE_PRIOR_SQRT_INFO, i, i + 1},
                sym::Matrix66d::Identity() / params.pose_prior_noise);
  }
}

/**
 * Add randomly sampled correspondences and their corresponding inverse range landmarks.  For each
 * correspondence, we have the pixel coordinates in both the source and target images, as well as a
 * weight to apply to that correspondence's cost.  Each landmark has its inverse range in the source
 * view, as well as a Gaussian inverse range prior with mean and sigma.
 */
void AddCorrespondences(const std::vector<sym::PosedCamera<sym::LinearCameraCald>>& cams,
                        const BundleAdjustmentProblemParams& params, std::mt19937& gen,
                        sym::Valuesd* const values) {
  // Sample random correspondences
  const std::vector<Eigen::Vector2d> source_coords =
      sym::example_utils::PickSourceCoordsRandomlyFromGrid<double>(params.image_shape,
                                                                   params.num_landmarks, gen);
  const std::vector<double> source_inverse_ranges =
      sym::example_utils::SampleInverseRanges<double>(params.num_landmarks, gen);

  std::vector<std::pair<Eigen::Matrix<double, 2, 1>, double>> source_observations;
  std::transform(
      source_coords.begin(), source_coords.end(), source_inverse_ranges.begin(),
      std::back_inserter(source_observations),
      [](const auto& uv, const double inverse_range) { return std::make_pair(uv, inverse_range); });
  const std::vector<sym::example_utils::Correspondence<double>> correspondences =
      sym::example_utils::GenerateCorrespondences(cams[0], cams[1], source_observations, gen,
                                                  params.epsilon, params.noise_px,
                                                  params.num_outliers);

  // Fill matches and landmarks for each correspondence
  std::normal_distribution<double> range_normal_dist(0, params.landmark_relative_range_noise);
  for (int i = 0; i < params.num_landmarks; i++) {
    const double source_range = 1 / source_inverse_ranges[i];
    const double range_perturbation = sym::Clamp(1 + range_normal_dist(gen), 0.5, 2.0);
    values->Set({Var::LANDMARK, i}, 1 / (source_range * range_perturbation));

    const auto& correspondence = correspondences[i];

    values->Set({Var::MATCH_SOURCE_COORDS, 1, i}, correspondence.source_uv);
    values->Set({Var::MATCH_TARGET_COORDS, 1, i}, correspondence.target_uv);
    values->Set({Var::MATCH_WEIGHT, 1, i}, correspondence.is_valid);
    values->Set({Var::LANDMARK_PRIOR, 1, i}, source_inverse_ranges[i]);
    values->Set({Var::LANDMARK_PRIOR_SIGMA, 1, i}, 100.0);
  }
}

}  // namespace

/**
 * Build the Values for a small bundle adjustment problem.  Generates multiple posed cameras, with
 * Gaussian priors on their relative poses, as well as noisy correspondences and landmarks.
 */
sym::Valuesd BuildValues(std::mt19937& gen, const BundleAdjustmentProblemParams& params) {
  sym::Valuesd values;

  values.Set({Var::EPSILON}, params.epsilon);

  // The factors we use have variable convexity, for use as a tunable robust cost or in an iterative
  // Graduated Non-Convexity (GNC) optimization.  See the GncFactor docstring for more
  // information.
  values.Set({Var::GNC_SCALE}, params.reprojection_error_gnc_scale);
  values.Set(Var::GNC_MU, 0.0);

  const auto cams = AddViews(params, gen, &values);
  AddPosePriors(cams, params, gen, &values);
  AddCorrespondences(cams, params, gen, &values);

  return values;
}

}  // namespace bundle_adjustment

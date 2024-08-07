/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#include <spdlog/spdlog.h>

#include <sym/factors/between_factor_pose3.h>
#include <sym/factors/inverse_range_landmark_linear_gnc_factor.h>
#include <sym/factors/inverse_range_landmark_prior_factor.h>
#include <symforce/examples/example_utils/bundle_adjustment_util.h>
#include <symforce/opt/assert.h>
#include <symforce/opt/factor.h>
#include <symforce/opt/optimizer.h>

#include "./build_example_state.h"

namespace bundle_adjustment {

/**
 * Creates a factor for a prior on the relative pose between view i and view j
 */
// 为视图i和j之间的相对位姿先验，创建一个因子
sym::Factord CreateRelativePosePriorFactor(const int i, const int j) {
  return sym::Factord::Hessian(sym::BetweenFactorPose3<double>,
                               // 输入参数keys: 视图i和视图j, 视图i和j之间的相对位姿先验，以及相对位姿先验的权重信息矩阵的平方根
                               {{Var::VIEW, i},
                                {Var::VIEW, j},
                                {Var::POSE_PRIOR_T, i, j},
                                {Var::POSE_PRIOR_SQRT_INFO, i, j},
                                Var::EPSILON},
                               // 优化参数的keys: 这里指示优化视图i和视图j的位姿
                               {
                                   {Var::VIEW, i},
                                   {Var::VIEW, j},
                               });
}

/**
 * Creates a factor for a prior on the inverse range of landmark landmark_idx based on its initial
 * triangulation between view 0 and view i
 */
// 为基于视图0和i之间的初始三角测量的路标landmark_idx的逆范围先验，创建一个因子
sym::Factord CreateInverseRangeLandmarkPriorFactor(const int i, const int landmark_idx) {
  return sym::Factord::Hessian(sym::InverseRangeLandmarkPriorFactor<double>,
                               {{Var::LANDMARK, landmark_idx},
                                {Var::LANDMARK_PRIOR, i, landmark_idx},
                                {Var::MATCH_WEIGHT, i, landmark_idx},
                                {Var::LANDMARK_PRIOR_SIGMA, i, landmark_idx},
                                Var::EPSILON},
                               // 优化参数的key: 指示优化路标的逆深度 
                               {{Var::LANDMARK, landmark_idx}});
}

/**
 * Creates a factor for a reprojection error residual of landmark landmark_idx projected into view i
 */
// 为投影到视图i的路标landmark_idx的重投影残差，创建一个因子
sym::Factord CreateInverseRangeLandmarkGncFactor(const int i, const int landmark_idx) {
  return sym::Factord::Hessian(sym::InverseRangeLandmarkLinearGncFactor<double>,
                               {{Var::VIEW, 0},
                                {Var::CALIBRATION, 0},
                                {Var::VIEW, i},
                                {Var::CALIBRATION, i},
                                {Var::LANDMARK, landmark_idx},
                                {Var::MATCH_SOURCE_COORDS, i, landmark_idx},
                                {Var::MATCH_TARGET_COORDS, i, landmark_idx},
                                {Var::MATCH_WEIGHT, i, landmark_idx},
                                Var::GNC_MU,
                                Var::GNC_SCALE,
                                Var::EPSILON},
                               // 以下为keys to optimize, 指示优化的状态量为：视图i的位姿，视图j的位姿，以及路标点landmark_idx的逆深度
                               {
                                   {Var::VIEW, 0},
                                   {Var::VIEW, i},
                                   {Var::LANDMARK, landmark_idx},
                               });
}

// 残差总共72维，充分说明：一个残差对应一个因子，或者说一个因子对应一个残差
std::vector<sym::Factord> BuildFactors(const BundleAdjustmentProblemParams& params) {
  std::vector<sym::Factord> factors;

  // Relative pose priors
  // 相对位姿先验的残差维数：6 * 2 = 12
  for (int i = 0; i < params.num_views; i++) {
    for (int j = 0; j < params.num_views; j++) {
      if (i == j) {
        continue;
      }

      // 创建相对位姿先验因子
      factors.push_back(CreateRelativePosePriorFactor(i, j));
    }
  }

  // Inverse range priors
  // 逆范围先验的残差维数：1 * 20 = 20
  for (int i = 1; i < params.num_views; i++) { // 注意视图序号是从1开始的，主要是因为只有2帧
    for (int landmark_idx = 0; landmark_idx < params.num_landmarks; landmark_idx++) {
      // 创建逆范围先验因子
      factors.push_back(CreateInverseRangeLandmarkPriorFactor(i, landmark_idx));
    }
  }

  // Reprojection errors
  // 重投影误差的维数：2 * 20 = 40
  for (int i = 1; i < params.num_views; i++) {
    for (int landmark_idx = 0; landmark_idx < params.num_landmarks; landmark_idx++) {
      // 创建重投影残差因子
      factors.push_back(CreateInverseRangeLandmarkGncFactor(i, landmark_idx));
    }
  }

  return factors;
}

/**
 * Compute the keys in the Values which are optimized (as opposed to fixed)
 *
 * We fix the pose of view 0 so that the whole problem is constrained; alternatively, we could add a
 * prior on the pose of view 0 and leave it optimized
 */
std::vector<sym::Key> ComputeKeysToOptimizeWithoutView0(const std::vector<sym::Factord>& factors) {
  std::vector<sym::Key> keys_to_optimize;

  // ComputeKeysToOptimize will return all of the keys touched by all of the factors we've
  // created (specifically the optimized keys for those factors, i.e. the keys for which the factors
  // have derivatives, as opposed to other factor parameters like weights or epsilon)
  for (const auto& key : ComputeKeysToOptimize(factors)) {
    // Don't optimize view 0
    if (key == sym::Key(Var::VIEW, 0)) {
      continue;
    }

    keys_to_optimize.push_back(key);
  }

  return keys_to_optimize;
}

void RunBundleAdjustment() {
  // Create initial state
  std::mt19937 gen(42);
  const auto params = BundleAdjustmentProblemParams();
  // step1 为最小二乘问题建立Values
  sym::Valuesd values = BuildValues(gen, params);

  spdlog::info("Initial State:");
  for (int i = 0; i < params.num_views; i++) {
    spdlog::info("Pose {}: {}", i, values.At<sym::Pose3d>({Var::VIEW, i}));
  }
  spdlog::info("Landmarks:");
  for (int i = 0; i < params.num_landmarks; i++) {
    spdlog::info("{} ", values.At<double>({Var::LANDMARK, i}));
  }

  // Create and set up Optimizer
  // step2 建立因子图
  const std::vector<sym::Factord> factors = BuildFactors(params);
  // 计算除了view0之外的optimized keys
  const std::vector<sym::Key> optimized_keys = ComputeKeysToOptimizeWithoutView0(factors);

  const sym::optimizer_params_t optimizer_params = sym::example_utils::OptimizerParams();

  // 构造优化器
  sym::Optimizerd optimizer(optimizer_params, factors, "BundleAdjustmentOptimizer", optimized_keys,
                            params.epsilon);

  // Optimize
  // step3 执行BA
  const sym::Optimizerd::Stats stats = optimizer.Optimize(values);

  // Print out results
  spdlog::info("Optimized State:");
  for (int i = 0; i < params.num_views; i++) {
    spdlog::info("Pose {}: {}", i, values.At<sym::Pose3d>({Var::VIEW, i}));
  }
  spdlog::info("Landmarks:");
  for (int i = 0; i < params.num_landmarks; i++) {
    spdlog::info("{} ", values.At<double>({Var::LANDMARK, i}));
  }

  const auto& iteration_stats = stats.iterations;
  const auto& first_iter = iteration_stats.front();
  const auto& last_iter = iteration_stats.back();

  // Note that the best iteration (with the best error, and representing the Values that gives that
  // error) may not be the last iteration, if later steps did not successfully decrease the cost
  const auto& best_iter = iteration_stats[stats.best_index];

  spdlog::info("Iterations: {}", last_iter.iteration);
  spdlog::info("Lambda: {}", last_iter.current_lambda);
  spdlog::info("Initial error: {}", first_iter.new_error);
  spdlog::info("Final error: {}", best_iter.new_error);
  spdlog::info("Status: {}", stats.status);

  // Check successful convergence
  SYM_ASSERT(best_iter.new_error < 10);
  SYM_ASSERT(stats.status == sym::optimization_status_t::SUCCESS);
}

}  // namespace bundle_adjustment

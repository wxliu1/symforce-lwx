/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#include <iostream>

#include <sym/pose3.h>
#include <symforce/examples/example_utils/bundle_adjustment_util.h>
#include <symforce/opt/assert.h>
#include <symforce/opt/optimizer.h>

#include "./build_example_state.h"
#include "symforce/bundle_adjustment_fixed_size/linearization.h"

/*
 *
 * add comments by wxliu:
 * 
 * This is the C++ file that actually runs the optimization. 
 * It builds up the Values for the problem and builds a factor graph. 
 * In this example, the C++ optimization consists of one sym::Factor, 
 * with a single generated linearization function that contains all of the symbolic residuals.
 *
 * This particular example is set up so that the number of cameras and landmarks is set at code generation time;
 * in contrast, the Bundle Adjustment example shows how to make them configurable at runtime.
 * 
 * 在代码生成时（python代码生成c++代码）固定最小二乘问题的大小（相机和路标点的个数固定），能产生更有效率的线性化函数
 * 因为常见的子表达式消除可以应用于多个因子。
 * 例如，将不同的特征重新投影到同一台相机中的多个因子通常会共享计算量。
 * 比如相对位姿，就可以提前计算好，作为子表达式固定下来
 * 
 */

namespace bundle_adjustment_fixed_size {

// 在本例中，C++优化由一个sym::Factor因子组成，带有 一个包含所有符号残差的单一生成的线性化函数。 
// 固定size的BA和非固定BA, 区别在于前者，相机位姿个数，路标点个数，匹配的特征（correspondences）个数都是固定的
sym::Factord BuildFactor() {
  const std::vector<sym::Key> factor_keys = {{Var::CALIBRATION, 0},
                                             {Var::VIEW, 0},
                                             {Var::CALIBRATION, 1},
                                             {Var::VIEW, 1},
                                             {Var::POSE_PRIOR_T, 0, 0},
                                             {Var::POSE_PRIOR_SQRT_INFO, 0, 0},
                                             {Var::POSE_PRIOR_T, 0, 1},
                                             {Var::POSE_PRIOR_SQRT_INFO, 0, 1},
                                             {Var::POSE_PRIOR_T, 1, 0},
                                             {Var::POSE_PRIOR_SQRT_INFO, 1, 0},
                                             {Var::POSE_PRIOR_T, 1, 1},
                                             {Var::POSE_PRIOR_SQRT_INFO, 1, 1},
                                             {Var::MATCH_SOURCE_COORDS, 1, 0}, // 这里的1代表view 1, 而0代表landmark 0
                                             {Var::MATCH_TARGET_COORDS, 1, 0},
                                             {Var::MATCH_WEIGHT, 1, 0},
                                             {Var::LANDMARK_PRIOR, 1, 0},
                                             {Var::LANDMARK_PRIOR_SIGMA, 1, 0},
                                             {Var::MATCH_SOURCE_COORDS, 1, 1},
                                             {Var::MATCH_TARGET_COORDS, 1, 1},
                                             {Var::MATCH_WEIGHT, 1, 1},
                                             {Var::LANDMARK_PRIOR, 1, 1},
                                             {Var::LANDMARK_PRIOR_SIGMA, 1, 1},
                                             {Var::MATCH_SOURCE_COORDS, 1, 2},
                                             {Var::MATCH_TARGET_COORDS, 1, 2},
                                             {Var::MATCH_WEIGHT, 1, 2},
                                             {Var::LANDMARK_PRIOR, 1, 2},
                                             {Var::LANDMARK_PRIOR_SIGMA, 1, 2},
                                             {Var::MATCH_SOURCE_COORDS, 1, 3},
                                             {Var::MATCH_TARGET_COORDS, 1, 3},
                                             {Var::MATCH_WEIGHT, 1, 3},
                                             {Var::LANDMARK_PRIOR, 1, 3},
                                             {Var::LANDMARK_PRIOR_SIGMA, 1, 3},
                                             {Var::MATCH_SOURCE_COORDS, 1, 4},
                                             {Var::MATCH_TARGET_COORDS, 1, 4},
                                             {Var::MATCH_WEIGHT, 1, 4},
                                             {Var::LANDMARK_PRIOR, 1, 4},
                                             {Var::LANDMARK_PRIOR_SIGMA, 1, 4},
                                             {Var::MATCH_SOURCE_COORDS, 1, 5},
                                             {Var::MATCH_TARGET_COORDS, 1, 5},
                                             {Var::MATCH_WEIGHT, 1, 5},
                                             {Var::LANDMARK_PRIOR, 1, 5},
                                             {Var::LANDMARK_PRIOR_SIGMA, 1, 5},
                                             {Var::MATCH_SOURCE_COORDS, 1, 6},
                                             {Var::MATCH_TARGET_COORDS, 1, 6},
                                             {Var::MATCH_WEIGHT, 1, 6},
                                             {Var::LANDMARK_PRIOR, 1, 6},
                                             {Var::LANDMARK_PRIOR_SIGMA, 1, 6},
                                             {Var::MATCH_SOURCE_COORDS, 1, 7},
                                             {Var::MATCH_TARGET_COORDS, 1, 7},
                                             {Var::MATCH_WEIGHT, 1, 7},
                                             {Var::LANDMARK_PRIOR, 1, 7},
                                             {Var::LANDMARK_PRIOR_SIGMA, 1, 7},
                                             {Var::MATCH_SOURCE_COORDS, 1, 8},
                                             {Var::MATCH_TARGET_COORDS, 1, 8},
                                             {Var::MATCH_WEIGHT, 1, 8},
                                             {Var::LANDMARK_PRIOR, 1, 8},
                                             {Var::LANDMARK_PRIOR_SIGMA, 1, 8},
                                             {Var::MATCH_SOURCE_COORDS, 1, 9},
                                             {Var::MATCH_TARGET_COORDS, 1, 9},
                                             {Var::MATCH_WEIGHT, 1, 9},
                                             {Var::LANDMARK_PRIOR, 1, 9},
                                             {Var::LANDMARK_PRIOR_SIGMA, 1, 9},
                                             {Var::MATCH_SOURCE_COORDS, 1, 10},
                                             {Var::MATCH_TARGET_COORDS, 1, 10},
                                             {Var::MATCH_WEIGHT, 1, 10},
                                             {Var::LANDMARK_PRIOR, 1, 10},
                                             {Var::LANDMARK_PRIOR_SIGMA, 1, 10},
                                             {Var::MATCH_SOURCE_COORDS, 1, 11},
                                             {Var::MATCH_TARGET_COORDS, 1, 11},
                                             {Var::MATCH_WEIGHT, 1, 11},
                                             {Var::LANDMARK_PRIOR, 1, 11},
                                             {Var::LANDMARK_PRIOR_SIGMA, 1, 11},
                                             {Var::MATCH_SOURCE_COORDS, 1, 12},
                                             {Var::MATCH_TARGET_COORDS, 1, 12},
                                             {Var::MATCH_WEIGHT, 1, 12},
                                             {Var::LANDMARK_PRIOR, 1, 12},
                                             {Var::LANDMARK_PRIOR_SIGMA, 1, 12},
                                             {Var::MATCH_SOURCE_COORDS, 1, 13},
                                             {Var::MATCH_TARGET_COORDS, 1, 13},
                                             {Var::MATCH_WEIGHT, 1, 13},
                                             {Var::LANDMARK_PRIOR, 1, 13},
                                             {Var::LANDMARK_PRIOR_SIGMA, 1, 13},
                                             {Var::MATCH_SOURCE_COORDS, 1, 14},
                                             {Var::MATCH_TARGET_COORDS, 1, 14},
                                             {Var::MATCH_WEIGHT, 1, 14},
                                             {Var::LANDMARK_PRIOR, 1, 14},
                                             {Var::LANDMARK_PRIOR_SIGMA, 1, 14},
                                             {Var::MATCH_SOURCE_COORDS, 1, 15},
                                             {Var::MATCH_TARGET_COORDS, 1, 15},
                                             {Var::MATCH_WEIGHT, 1, 15},
                                             {Var::LANDMARK_PRIOR, 1, 15},
                                             {Var::LANDMARK_PRIOR_SIGMA, 1, 15},
                                             {Var::MATCH_SOURCE_COORDS, 1, 16},
                                             {Var::MATCH_TARGET_COORDS, 1, 16},
                                             {Var::MATCH_WEIGHT, 1, 16},
                                             {Var::LANDMARK_PRIOR, 1, 16},
                                             {Var::LANDMARK_PRIOR_SIGMA, 1, 16},
                                             {Var::MATCH_SOURCE_COORDS, 1, 17},
                                             {Var::MATCH_TARGET_COORDS, 1, 17},
                                             {Var::MATCH_WEIGHT, 1, 17},
                                             {Var::LANDMARK_PRIOR, 1, 17},
                                             {Var::LANDMARK_PRIOR_SIGMA, 1, 17},
                                             {Var::MATCH_SOURCE_COORDS, 1, 18},
                                             {Var::MATCH_TARGET_COORDS, 1, 18},
                                             {Var::MATCH_WEIGHT, 1, 18},
                                             {Var::LANDMARK_PRIOR, 1, 18},
                                             {Var::LANDMARK_PRIOR_SIGMA, 1, 18},
                                             {Var::MATCH_SOURCE_COORDS, 1, 19},
                                             {Var::MATCH_TARGET_COORDS, 1, 19},
                                             {Var::MATCH_WEIGHT, 1, 19},
                                             {Var::LANDMARK_PRIOR, 1, 19},
                                             {Var::LANDMARK_PRIOR_SIGMA, 1, 19},
                                             {Var::LANDMARK, 0},
                                             {Var::LANDMARK, 1},
                                             {Var::LANDMARK, 2},
                                             {Var::LANDMARK, 3},
                                             {Var::LANDMARK, 4},
                                             {Var::LANDMARK, 5},
                                             {Var::LANDMARK, 6},
                                             {Var::LANDMARK, 7},
                                             {Var::LANDMARK, 8},
                                             {Var::LANDMARK, 9},
                                             {Var::LANDMARK, 10},
                                             {Var::LANDMARK, 11},
                                             {Var::LANDMARK, 12},
                                             {Var::LANDMARK, 13},
                                             {Var::LANDMARK, 14},
                                             {Var::LANDMARK, 15},
                                             {Var::LANDMARK, 16},
                                             {Var::LANDMARK, 17},
                                             {Var::LANDMARK, 18},
                                             {Var::LANDMARK, 19},
                                             {Var::GNC_SCALE},
                                             {Var::GNC_MU},
                                             {Var::EPSILON}};

  const std::vector<sym::Key> optimized_keys = {
      {Var::VIEW, 1},      {Var::LANDMARK, 0},  {Var::LANDMARK, 1},  {Var::LANDMARK, 2},
      {Var::LANDMARK, 3},  {Var::LANDMARK, 4},  {Var::LANDMARK, 5},  {Var::LANDMARK, 6},
      {Var::LANDMARK, 7},  {Var::LANDMARK, 8},  {Var::LANDMARK, 9},  {Var::LANDMARK, 10},
      {Var::LANDMARK, 11}, {Var::LANDMARK, 12}, {Var::LANDMARK, 13}, {Var::LANDMARK, 14},
      {Var::LANDMARK, 15}, {Var::LANDMARK, 16}, {Var::LANDMARK, 17}, {Var::LANDMARK, 18},
      {Var::LANDMARK, 19}};

  // 分别构造的factor keys和optimized keys 用来构造因子图
  // 符号表达式生成的Linearization函数在这里用上了
  return sym::Factord::Hessian(bundle_adjustment_fixed_size::Linearization<double>, factor_keys,
                               optimized_keys);
}

void RunBundleAdjustment() {
  // 优化三步骤几无差别，详见bundle_adjustment
  // Create initial state
  std::mt19937 gen(42);
  const auto params = BundleAdjustmentProblemParams();
  //- 这里build values用于最小二乘问题优化，是真正需要优化的状态量
  //- 而python脚本里面的build values用于输出线性化符号表达式，用于因子图构造
  //- 作用完全不一样，需要特别地注意！
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
  const sym::optimizer_params_t optimizer_params = sym::example_utils::OptimizerParams();

  sym::Optimizerd optimizer(optimizer_params, {BuildFactor()}, "BundleAdjustmentOptimizer", {},
                            params.epsilon);

  // Optimize
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

}  // namespace bundle_adjustment_fixed_size

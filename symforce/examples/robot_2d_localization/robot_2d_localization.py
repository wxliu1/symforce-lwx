# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

"""
below sections was extracted by lwx on 2024-5-8:

This tutorial shows the central workflow in SymForce for creating symbolic expressions, generating code, and optimizing. 
This approach works well for a wide range of complex problems in robotics, computer vision, and applied science.

However, each piece may also be used independently. 
The optimization machinery can work with handwritten functions, and the symbolic math and code generation is useful outside of any optimization context.

本教程展示了创建符号表达式、生成代码和优化的中心工作流。
这种方法适用于机器人技术、计算机视觉和应用科学中的各种复杂问题。

然而，每个部件也可以独立使用。优化机制可以与手写函数一起工作，而符号数学和代码生成在任何优化上下文之外都很有用。
"""


"""
Demonstrates solving a 2D localization problem with SymForce. The goal is for a robot
in a 2D plane to compute its trajectory given distance measurements from wheel odometry
and relative bearing angle measurements to known landmarks in the environment.
"""
"""
演示了使用SymForce求解一个2D定位问题。目标是根据从轮速计得到的距离测量和环境中已知路标点的相对方位角度测量，
来计算机器人在2D平面的轨迹
"""
# -----------------------------------------------------------------------------
# Set the default epsilon to a symbol
# -----------------------------------------------------------------------------
import symforce

symforce.set_epsilon_to_symbol()

# -----------------------------------------------------------------------------
# Create initial Values
# -----------------------------------------------------------------------------
import numpy as np

from symforce import typing as T # 导入symforce包里面的typing模块到当前模块的命名空间，并重命名为T
from symforce.values import Values # 导入symforce.values子包里面的Values类到当前命名空间


def build_initial_values() -> T.Tuple[Values, int, int]:
    """
    Creates a Values with numerical values for the constants in the problem, and initial guesses
    for the optimized variables
    """
    num_poses = 3
    num_landmarks = 3
    # 关键字参数是指在函数调用时，通过参数名指定参数值，这样可以不用担心参数的顺序。
    initial_values = Values(
        poses=[sf.Pose2.identity()] * num_poses, # world系下的位姿
        landmarks=[sf.V2(-2, 2), sf.V2(1, -3), sf.V2(5, 2)], # world系下的坐标
        distances=[1.7, 1.4], # "0, 1"帧和"1, 2"帧的相对距离
        angles=np.deg2rad([[55, 245, -35], [95, 220, -20], [125, 220, -20]]).tolist(), # 3个帧的3个路标点的角度测量值
        epsilon=sf.numeric_epsilon,
    )

    return initial_values, num_poses, num_landmarks


# -----------------------------------------------------------------------------
# Define residual functions
# -----------------------------------------------------------------------------
import symforce.symbolic as sf


def bearing_residual(
    pose: sf.Pose2, landmark: sf.V2, angle: sf.Scalar, epsilon: sf.Scalar
) -> sf.V1:
    """
    Residual from a relative bearing measurement of a 2D pose to a landmark.
    """
    # 一个路标点的2D位姿的相对方位测量的残差

    # 把路标点从world系转到body系
    t_body = pose.inverse() * landmark
    # 根据路标点的2D坐标计算相对方位角度(relative bearing angle):反正切计算公式atan2(y, x)
    predicted_angle = sf.atan2(t_body[1], t_body[0], epsilon=epsilon)
    # 预测值减去测量值
    return sf.V1(sf.wrap_angle(predicted_angle - angle))


def odometry_residual(
    pose_a: sf.Pose2, pose_b: sf.Pose2, dist: sf.Scalar, epsilon: sf.Scalar
) -> sf.V1:
    """
    Residual from the scalar distance between two poses.
    """
    # 两帧位姿的平移相减求2范数，得到相对距离，再减去测量距离
    return sf.V1((pose_b.t - pose_a.t).norm(epsilon=epsilon) - dist)


# -----------------------------------------------------------------------------
# Create a set of factors to represent the full problem
# -----------------------------------------------------------------------------
from symforce.opt.factor import Factor # 导入symforce.opt.factor模块里面的Factor类到当前命名空间


def build_factors(num_poses: int, num_landmarks: int) -> T.Iterator[Factor]:
    """
    Build factors for a problem of the given dimensionality.
    """
    # 创建给定维数的问题的因子

    # yield是Python中的一个关键字，用于在函数内部定义一个生成器（generator），这使得函数能够以迭代的方式执行
    '''
    yield 是 Python 的一个关键字，用于从一个函数中返回一个生成器（generator）。
    生成器是一种特殊类型的迭代器，它允许你延迟计算结果，这在处理大数据或者创建复杂数据结构时特别有用，因为你不需要一次性将所有的数据都存储在内存中。
    一个使用 yield 的函数会被称为生成器函数。
    这种函数并不直接返回一个值，而是生成一系列的值。
    每次调用这个生成器函数，它会从上次离开的地方继续执行，并且可以产生许多结果，而不是单个值。
    '''

    # 里程计因子，函数参数为相邻两帧位姿以及相对距离测量值
    for i in range(num_poses - 1):
        yield Factor(
            residual=odometry_residual,
            keys=[f"poses[{i}]", f"poses[{i + 1}]", f"distances[{i}]", "epsilon"],
        )

    # 方位因子，函数参数为每一帧的位姿以及对应的每个路标点和角度测量值
    for i in range(num_poses):
        for j in range(num_landmarks):
            yield Factor(
                residual=bearing_residual,
                keys=[f"poses[{i}]", f"landmarks[{j}]", f"angles[{i}][{j}]", "epsilon"],
            )


# -----------------------------------------------------------------------------
# Instantiate, optimize, and visualize
# -----------------------------------------------------------------------------
from symforce.opt.optimizer import Optimizer # 导入symforce.opt.optimizer模块里面的Optimizer类到当前命名空间


def main() -> None:
    # Create a problem setup and initial guess 创建问题的初始猜测值
    initial_values, num_poses, num_landmarks = build_initial_values()

    # Create factors 创建因子
    factors = build_factors(num_poses=num_poses, num_landmarks=num_landmarks)

    # Select the keys to optimize - the rest will be held constant
    # 选择要优化的状态量-其余的将保持不变
    # 只优化位姿：我们的目标是找到使因子图的残差最小化的位姿
    optimized_keys = [f"poses[{i}]" for i in range(num_poses)]

    # Create the optimizer
    optimizer = Optimizer(
        factors=factors,
        optimized_keys=optimized_keys,
        debug_stats=True,  # Return problem stats for every iteration
        params=Optimizer.Params(verbose=True),  # Customize optimizer behavior
    )

    # Solve and return the result
    '''
    执行优化
    Now run the optimization! 
    This returns an Optimizer.Result object that contains the optimized values, error statistics, and per-iteration debug stats (if enabled).
    '''
    result = optimizer.optimize(initial_values)

    # assert result.status == Optimizer.Status.SUCCESS # 可以加一个断言

    # Print some values
    print(f"Num iterations: {len(result.iterations) - 1}")
    print(f"Final error: {result.error():.6f}")
    print(f"Status: {result.status}")

    for i, pose in enumerate(result.optimized_values["poses"]):
        print(f"Pose {i}: t = {pose.position()}, heading = {pose.rotation().to_tangent()[0]}")

    # Plot the result
    # TODO(hayk): mypy gives the below error, but a relative import also doesn't work.
    # Skipping analyzing "symforce.examples.robot_2d_localization.plotting":
    #     found module but no type hints or library stubs
    from symforce.examples.robot_2d_localization.plotting import plot_solution # 导入plotting模块的plot_solution函数到当前命名空间

    plot_solution(optimizer, result)


import shutil
from pathlib import Path

# -----------------------------------------------------------------------------
# (Optional) Generate C++ functions for residuals with on-manifold jacobians
# -----------------------------------------------------------------------------
from symforce.codegen import Codegen # 导入symforce.codegen子包的Codegen类到当前命名空间
from symforce.codegen import CppConfig # 导入CppConfig子类到当前命名空间

# The Codegen class is the central tool for generating runtime code from symbolic expressions.
# Codegen类是从符号表达式生成运行时代码的中心工具。

# C++代码只依赖于Eigen，并在一个共享所有公共子表达式的平面函数中计算结果

def generate_bearing_residual_code(
    output_dir: T.Optional[Path] = None, print_code: bool = False
) -> None:
    """
    Generate C++ code for the bearing residual function. A C++ Factor can then be
    constructed and optimized from this function without any Python dependency.
    """
    # Create a Codegen object for the symbolic residual function, targeted at C++
    codegen = Codegen.function(bearing_residual, config=CppConfig())

    # Generate the function and print the code
    metadata = codegen.generate_function(output_dir=output_dir, skip_directory_nesting=True)
    if print_code:
        print(metadata.generated_files[0].read_text())

    if output_dir is None:
        shutil.rmtree(metadata.output_dir)

    # Create a Codegen object that computes a linearization from the residual Codegen object,
    # by introspecting and symbolically differentiating the given arguments
    codegen_with_linearization = codegen.with_linearization(which_args=["pose"])

    # Generate the function and print the code
    metadata = codegen_with_linearization.generate_function(
        output_dir=output_dir, skip_directory_nesting=True
    )
    if print_code:
        print(metadata.generated_files[0].read_text())

    if output_dir is None:
        shutil.rmtree(metadata.output_dir)


def generate_odometry_residual_code(
    output_dir: T.Optional[Path] = None, print_code: bool = False
) -> None:
    """
    Generate C++ code for the odometry residual function. A C++ Factor can then be
    constructed and optimized from this function without any Python dependency.
    """
    # Create a Codegen object for the symbolic residual function, targeted at C++
    codegen = Codegen.function(odometry_residual, config=CppConfig())

    # Generate the function and print the code
    metadata = codegen.generate_function(output_dir=output_dir, skip_directory_nesting=True)
    if print_code:
        print(metadata.generated_files[0].read_text())

    if output_dir is None:
        shutil.rmtree(metadata.output_dir)

    # Create a Codegen object that computes a linearization from the residual Codegen object,
    # by introspecting and symbolically differentiating the given arguments
    codegen_with_linearization = codegen.with_linearization(which_args=["pose_a", "pose_b"])

    # Generate the function and print the code
    metadata = codegen_with_linearization.generate_function(
        output_dir=output_dir, skip_directory_nesting=True
    )
    if print_code:
        print(metadata.generated_files[0].read_text())

    if output_dir is None:
        shutil.rmtree(metadata.output_dir)


if __name__ == "__main__":
    main()

    # Uncomment this to print generated C++ code
    # generate_bearing_residual_code(print_code=True)
    # generate_odometry_residual_code(print_code=True)

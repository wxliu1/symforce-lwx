# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

"""
Symbolic factor and codegen for the Bundle-Adjustment-in-the-Large problem
"""

from pathlib import Path

import symforce.symbolic as sf
from symforce import codegen
from symforce.codegen import values_codegen
from symforce.values import Values


def snavely_reprojection_residual(
    cam_T_world: sf.Pose3,
    intrinsics: sf.V3,
    point: sf.V3,
    pixel: sf.V2,
    epsilon: sf.Scalar,
) -> sf.V2:
    """
    Reprojection residual for the camera model used in the Bundle-Adjustment-in-the-Large dataset, a
    polynomial camera with two distortion coefficients, cx == cy == 0, and fx == fy

    See https://grail.cs.washington.edu/projects/bal/ for more information

    Args:
        cam_T_world: The (inverse) pose of the camera
        intrinsics: Camera intrinsics (f, k1, k2)
        point: The world point to be projected
        pixel: The measured pixel in the camera (with (0, 0) == center of image)

    Returns:
        residual: The reprojection residual
    """
    # 具有2个畸变系数的多项式畸变相机模型, cx==cy==0, fx==fy
    focal_length, k1, k2 = intrinsics

    # Here we're writing the projection ourselves because this isn't a camera model provided by
    # SymForce.  For cameras in `symforce.cam` we could just create a `sf.PosedCamera` and call
    # `camera.pixel_from_global_point` instead, or we could create a subclass of `sf.CameraCal` and
    # do that.
    # 转换为相机系3d坐标
    point_cam = cam_T_world * point

    # 转换为归一化平面的2d坐标
    # point_cam[:2]表示取前2维
    p = sf.V2(point_cam[:2]) / sf.Max(-point_cam[2], epsilon)

    # 径向畸变多项式: 1 + k_1r^2 + k_2r^4
    r = 1 + k1 * p.squared_norm() + k2 * p.squared_norm() ** 2

    # r * p 是畸变后的归一化平面2d坐标，再乘以焦距，得到在图像上的正确坐标
    pixel_projected = focal_length * r * p

    # 预测值减去实际值：投影后的正确像素平面坐标 减去测量坐标
    return pixel_projected - pixel


# 生成代码时，经常出现rhs, 其表示right-hand side, 表示的是GN方程式J^T{\delta}x=-J^Tr的右侧部分
# Gauss-Newton rhs 表示高斯牛顿线性方程的右侧部分
def generate(output_dir: Path) -> None:
    """
    Generates the snavely_reprojection_factor into C++, as well as a set of Keys to help construct
    the optimization problem in C++, and puts them into `output_dir`.  This is called by
    `symforce/test/symforce_examples_bundle_adjustment_in_the_large_codegen_test.py` to generate the
    contents of the `gen` folder inside this directory.
    """

    # Generate the residual function (see `gen/snavely_reprojection_factor.h`)
    # which_args列表里的参数是待优化的状态量，也是线性化时残差函数对这些状态量求偏导
    codegen.Codegen.function(snavely_reprojection_residual, codegen.CppConfig()).with_linearization(
        which_args=["cam_T_world", "intrinsics", "point"]
    ).generate_function(output_dir=output_dir, skip_directory_nesting=True)

    # Make a `Values` with variables used in the C++ problem, and generate C++ Keys for them (see
    # `gen/keys.h`)
    values = Values(
        cam_T_world=sf.Pose3(),
        intrinsics=sf.V3(),
        point=sf.V3(),
        pixel=sf.V2(),
        epsilon=sf.Scalar(),
    )

    # 为每个变量生成key
    values_codegen.generate_values_keys(
        values, output_dir, config=codegen.CppConfig(), skip_directory_nesting=True
    )

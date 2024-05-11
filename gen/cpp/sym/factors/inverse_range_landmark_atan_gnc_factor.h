// -----------------------------------------------------------------------------
// This file was autogenerated by symforce from template:
//     function/FUNCTION.h.jinja
// Do NOT modify by hand.
// -----------------------------------------------------------------------------

#pragma once

#include <Eigen/Core>

#include <sym/atan_camera_cal.h>
#include <sym/pose3.h>

namespace sym {

/**
 * Return the 2dof residual of reprojecting the landmark into the target camera and comparing
 * against the correspondence in the target camera.
 *
 * The landmark is specified as a pixel in the source camera and an inverse range; this means the
 * landmark is fixed in the source camera and always has residual 0 there (this 0 residual is not
 * returned, only the residual in the target camera is returned).
 *
 * The norm of the residual is whitened using the
 * :class:`BarronNoiseModel <symforce.opt.noise_models.BarronNoiseModel>`.  Whitening each
 * component of the reprojection error separately would result in rejecting individual components
 * as outliers. Instead, we minimize the whitened norm of the full reprojection error for each
 * point.  See
 * :meth:`ScalarNoiseModel.whiten_norm <symforce.opt.noise_models.ScalarNoiseModel.whiten_norm>`
 * for more information on this, and
 * :class:`BarronNoiseModel <symforce.opt.noise_models.BarronNoiseModel>` for more information on
 * the noise model.
 *
 * Args:
 *     source_pose: The pose of the source camera
 *     source_calibration: The source camera calibration
 *     target_pose: The pose of the target camera
 *     target_calibration: The target camera calibration
 *     source_inverse_range: The inverse range of the landmark in the source camera
 *     source_pixel: The location of the landmark in the source camera
 *     target_pixel: The location of the correspondence in the target camera
 *     weight: The weight of the factor
 *     gnc_mu: The mu convexity parameter for the
 *         :class:`BarronNoiseModel <symforce.opt.noise_models.BarronNoiseModel>`
 *     gnc_scale: The scale parameter for the
 *         :class:`BarronNoiseModel <symforce.opt.noise_models.BarronNoiseModel>`
 *     epsilon: Small positive value
 *
 * Outputs:
 *     res: 2dof residual of the reprojection
 *     jacobian: (2x13) jacobian of res wrt args source_pose (6), target_pose (6),
 *               source_inverse_range (1)
 *     hessian: (13x13) Gauss-Newton hessian for args source_pose (6), target_pose (6),
 *              source_inverse_range (1)
 *     rhs: (13x1) Gauss-Newton rhs for args source_pose (6), target_pose (6), source_inverse_range
 *          (1)
 */
template <typename Scalar>
void InverseRangeLandmarkAtanGncFactor(
    const sym::Pose3<Scalar>& source_pose, const sym::ATANCameraCal<Scalar>& source_calibration,
    const sym::Pose3<Scalar>& target_pose, const sym::ATANCameraCal<Scalar>& target_calibration,
    const Scalar source_inverse_range, const Eigen::Matrix<Scalar, 2, 1>& source_pixel,
    const Eigen::Matrix<Scalar, 2, 1>& target_pixel, const Scalar weight, const Scalar gnc_mu,
    const Scalar gnc_scale, const Scalar epsilon, Eigen::Matrix<Scalar, 2, 1>* const res = nullptr,
    Eigen::Matrix<Scalar, 2, 13>* const jacobian = nullptr,
    Eigen::Matrix<Scalar, 13, 13>* const hessian = nullptr,
    Eigen::Matrix<Scalar, 13, 1>* const rhs = nullptr) {
  // Total ops: 1231

  // Input arrays
  const Eigen::Matrix<Scalar, 7, 1>& _source_pose = source_pose.Data();
  const Eigen::Matrix<Scalar, 5, 1>& _source_calibration = source_calibration.Data();
  const Eigen::Matrix<Scalar, 7, 1>& _target_pose = target_pose.Data();
  const Eigen::Matrix<Scalar, 5, 1>& _target_calibration = target_calibration.Data();

  // Intermediate terms (351)
  const Scalar _tmp0 = std::pow(_target_pose[2], Scalar(2));
  const Scalar _tmp1 = -2 * _tmp0;
  const Scalar _tmp2 = std::pow(_target_pose[1], Scalar(2));
  const Scalar _tmp3 = 1 - 2 * _tmp2;
  const Scalar _tmp4 = _tmp1 + _tmp3;
  const Scalar _tmp5 = 2 * _source_pose[0];
  const Scalar _tmp6 = _source_pose[2] * _tmp5;
  const Scalar _tmp7 = 2 * _source_pose[1] * _source_pose[3];
  const Scalar _tmp8 = _tmp6 + _tmp7;
  const Scalar _tmp9 = -_source_calibration[3] + source_pixel(1, 0);
  const Scalar _tmp10 = std::pow(_tmp9, Scalar(2)) / std::pow(_source_calibration[1], Scalar(2));
  const Scalar _tmp11 = Scalar(0.5) * _source_calibration[4];
  const Scalar _tmp12 = std::tan(_tmp11);
  const Scalar _tmp13 = -_source_calibration[2] + source_pixel(0, 0);
  const Scalar _tmp14 = std::pow(_tmp13, Scalar(2)) / std::pow(_source_calibration[0], Scalar(2));
  const Scalar _tmp15 = _tmp10 + _tmp14 + epsilon;
  const Scalar _tmp16 = std::sqrt(_tmp15);
  const Scalar _tmp17 = _source_calibration[4] * _tmp16;
  const Scalar _tmp18 = std::tan(_tmp17);
  const Scalar _tmp19 = (Scalar(1) / Scalar(4)) * std::pow(_tmp18, Scalar(2)) / _tmp15;
  const Scalar _tmp20 = _tmp19 / std::pow(_tmp12, Scalar(2));
  const Scalar _tmp21 = epsilon + 1;
  const Scalar _tmp22 =
      std::pow(Scalar(_tmp10 * _tmp20 + _tmp14 * _tmp20 + _tmp21), Scalar(Scalar(-1) / Scalar(2)));
  const Scalar _tmp23 = _source_pose[1] * _tmp5;
  const Scalar _tmp24 = 2 * _source_pose[2];
  const Scalar _tmp25 = _source_pose[3] * _tmp24;
  const Scalar _tmp26 = -_tmp25;
  const Scalar _tmp27 = _tmp23 + _tmp26;
  const Scalar _tmp28 = _tmp9 / _source_calibration[1];
  const Scalar _tmp29 = (Scalar(1) / Scalar(2)) * _tmp18 / _tmp16;
  const Scalar _tmp30 = _tmp22 * _tmp29 / _tmp12;
  const Scalar _tmp31 = _tmp28 * _tmp30;
  const Scalar _tmp32 = std::pow(_source_pose[2], Scalar(2));
  const Scalar _tmp33 = -2 * _tmp32;
  const Scalar _tmp34 = std::pow(_source_pose[1], Scalar(2));
  const Scalar _tmp35 = -2 * _tmp34;
  const Scalar _tmp36 = _tmp13 / _source_calibration[0];
  const Scalar _tmp37 = _tmp36 * (_tmp33 + _tmp35 + 1);
  const Scalar _tmp38 = _source_pose[4] - _target_pose[4];
  const Scalar _tmp39 = _tmp38 * source_inverse_range;
  const Scalar _tmp40 = _tmp22 * _tmp8 + _tmp27 * _tmp31 + _tmp30 * _tmp37 + _tmp39;
  const Scalar _tmp41 = 2 * _target_pose[0];
  const Scalar _tmp42 = _target_pose[2] * _tmp41;
  const Scalar _tmp43 = 2 * _target_pose[3];
  const Scalar _tmp44 = _target_pose[1] * _tmp43;
  const Scalar _tmp45 = -_tmp44;
  const Scalar _tmp46 = _tmp42 + _tmp45;
  const Scalar _tmp47 = std::pow(_source_pose[0], Scalar(2));
  const Scalar _tmp48 = 1 - 2 * _tmp47;
  const Scalar _tmp49 = _tmp35 + _tmp48;
  const Scalar _tmp50 = _source_pose[3] * _tmp5;
  const Scalar _tmp51 = _source_pose[1] * _tmp24;
  const Scalar _tmp52 = _tmp50 + _tmp51;
  const Scalar _tmp53 = _source_pose[6] - _target_pose[6];
  const Scalar _tmp54 = _tmp53 * source_inverse_range;
  const Scalar _tmp55 = -_tmp7;
  const Scalar _tmp56 = _tmp55 + _tmp6;
  const Scalar _tmp57 = _tmp30 * _tmp36;
  const Scalar _tmp58 = _tmp22 * _tmp49 + _tmp31 * _tmp52 + _tmp54 + _tmp56 * _tmp57;
  const Scalar _tmp59 = _target_pose[2] * _tmp43;
  const Scalar _tmp60 = _target_pose[1] * _tmp41;
  const Scalar _tmp61 = _tmp59 + _tmp60;
  const Scalar _tmp62 = -_tmp50;
  const Scalar _tmp63 = _tmp51 + _tmp62;
  const Scalar _tmp64 = _tmp23 + _tmp25;
  const Scalar _tmp65 = _tmp33 + _tmp48;
  const Scalar _tmp66 = _source_pose[5] - _target_pose[5];
  const Scalar _tmp67 = _tmp66 * source_inverse_range;
  const Scalar _tmp68 = _tmp22 * _tmp63 + _tmp31 * _tmp65 + _tmp57 * _tmp64 + _tmp67;
  const Scalar _tmp69 = _tmp4 * _tmp40 + _tmp46 * _tmp58 + _tmp61 * _tmp68;
  const Scalar _tmp70 = Scalar(1.0) / (_target_calibration[4]);
  const Scalar _tmp71 = _target_calibration[0] * _tmp70;
  const Scalar _tmp72 = _tmp42 + _tmp44;
  const Scalar _tmp73 = std::pow(_target_pose[0], Scalar(2));
  const Scalar _tmp74 = -2 * _tmp73;
  const Scalar _tmp75 = _tmp3 + _tmp74;
  const Scalar _tmp76 = 2 * _target_pose[1] * _target_pose[2];
  const Scalar _tmp77 = _target_pose[0] * _tmp43;
  const Scalar _tmp78 = -_tmp77;
  const Scalar _tmp79 = _tmp76 + _tmp78;
  const Scalar _tmp80 = _tmp40 * _tmp72 + _tmp58 * _tmp75 + _tmp68 * _tmp79;
  const Scalar _tmp81 = std::max<Scalar>(_tmp80, epsilon);
  const Scalar _tmp82 = std::pow(_tmp81, Scalar(-2));
  const Scalar _tmp83 = -_tmp59;
  const Scalar _tmp84 = _tmp60 + _tmp83;
  const Scalar _tmp85 = _tmp76 + _tmp77;
  const Scalar _tmp86 = _tmp1 + _tmp74 + 1;
  const Scalar _tmp87 = _tmp40 * _tmp84 + _tmp58 * _tmp85 + _tmp68 * _tmp86;
  const Scalar _tmp88 = std::sqrt(Scalar(std::pow(_tmp69, Scalar(2)) * _tmp82 +
                                         _tmp82 * std::pow(_tmp87, Scalar(2)) + epsilon));
  const Scalar _tmp89 = Scalar(0.5) * _target_calibration[4];
  const Scalar _tmp90 = std::atan(2 * _tmp88 * std::tan(_tmp89)) / (_tmp81 * _tmp88);
  const Scalar _tmp91 = _target_calibration[2] - target_pixel(0, 0);
  const Scalar _tmp92 = _tmp69 * _tmp71 * _tmp90 + _tmp91;
  const Scalar _tmp93 = _target_calibration[1] * _tmp70;
  const Scalar _tmp94 = _target_calibration[3] - target_pixel(1, 0);
  const Scalar _tmp95 = _tmp87 * _tmp90 * _tmp93 + _tmp94;
  const Scalar _tmp96 = std::pow(_tmp92, Scalar(2)) + std::pow(_tmp95, Scalar(2)) + epsilon;
  const Scalar _tmp97 = Scalar(1.0) / (_tmp21 - gnc_mu);
  const Scalar _tmp98 = epsilon + std::fabs(_tmp97);
  const Scalar _tmp99 = std::pow(gnc_scale, Scalar(-2));
  const Scalar _tmp100 = _tmp99 / _tmp98;
  const Scalar _tmp101 = 2 - _tmp97;
  const Scalar _tmp102 =
      _tmp101 + epsilon * (2 * std::min<Scalar>(0, (((_tmp101) > 0) - ((_tmp101) < 0))) + 1);
  const Scalar _tmp103 = (Scalar(1) / Scalar(2)) * _tmp102;
  const Scalar _tmp104 = 2 * _tmp98 / _tmp102;
  const Scalar _tmp105 =
      std::sqrt(weight) * std::max<Scalar>(0, (((-std::fabs(_tmp17) + Scalar(M_PI_2)) > 0) -
                                               ((-std::fabs(_tmp17) + Scalar(M_PI_2)) < 0)));
  const Scalar _tmp106 =
      _tmp105 * std::sqrt(Scalar(_tmp104 * (std::pow(Scalar(_tmp100 * _tmp96 + 1), _tmp103) - 1))) *
      std::max<Scalar>(0, (((_tmp80) > 0) - ((_tmp80) < 0))) / std::sqrt(_tmp96);
  const Scalar _tmp107 = _tmp106 * _tmp92;
  const Scalar _tmp108 = _tmp106 * _tmp95;
  const Scalar _tmp109 = -_tmp47;
  const Scalar _tmp110 = _tmp109 + _tmp32;
  const Scalar _tmp111 = std::pow(_source_pose[3], Scalar(2));
  const Scalar _tmp112 = -_tmp34;
  const Scalar _tmp113 = _tmp111 + _tmp112;
  const Scalar _tmp114 = std::tan(_tmp11);
  const Scalar _tmp115 = _tmp19 / std::pow(_tmp114, Scalar(2));
  const Scalar _tmp116 = std::pow(Scalar(_tmp10 * _tmp115 + _tmp115 * _tmp14 + _tmp21),
                                  Scalar(Scalar(-1) / Scalar(2)));
  const Scalar _tmp117 = _tmp29 / _tmp114;
  const Scalar _tmp118 = _tmp116 * _tmp117;
  const Scalar _tmp119 = _tmp118 * _tmp28;
  const Scalar _tmp120 = -_tmp51;
  const Scalar _tmp121 = _tmp116 * (_tmp120 + _tmp62) + _tmp119 * (_tmp110 + _tmp113);
  const Scalar _tmp122 = -_tmp23;
  const Scalar _tmp123 = _tmp116 * _tmp8;
  const Scalar _tmp124 = _tmp117 * _tmp28;
  const Scalar _tmp125 = _tmp116 * (_tmp122 + _tmp25) + _tmp123 * _tmp124;
  const Scalar _tmp126 = -_tmp111;
  const Scalar _tmp127 = _tmp116 * _tmp63;
  const Scalar _tmp128 = _tmp116 * (_tmp112 + _tmp126 + _tmp32 + _tmp47) + _tmp124 * _tmp127;
  const Scalar _tmp129 = _tmp121 * _tmp75 + _tmp125 * _tmp72 + _tmp128 * _tmp79;
  const Scalar _tmp130 = _tmp116 * _tmp64;
  const Scalar _tmp131 = _tmp117 * _tmp36;
  const Scalar _tmp132 = _tmp119 * _tmp65 + _tmp127 + _tmp130 * _tmp131 + _tmp67;
  const Scalar _tmp133 = _tmp118 * _tmp37 + _tmp119 * _tmp27 + _tmp123 + _tmp39;
  const Scalar _tmp134 = _tmp116 * _tmp56;
  const Scalar _tmp135 = _tmp116 * _tmp49 + _tmp119 * _tmp52 + _tmp131 * _tmp134 + _tmp54;
  const Scalar _tmp136 = _tmp133 * _tmp84 + _tmp135 * _tmp85;
  const Scalar _tmp137 = _tmp132 * _tmp86 + _tmp136;
  const Scalar _tmp138 = std::pow(_tmp137, Scalar(2));
  const Scalar _tmp139 = _tmp132 * _tmp79 + _tmp133 * _tmp72;
  const Scalar _tmp140 = _tmp135 * _tmp75 + _tmp139;
  const Scalar _tmp141 = std::max<Scalar>(_tmp140, epsilon);
  const Scalar _tmp142 = (((_tmp140 - epsilon) > 0) - ((_tmp140 - epsilon) < 0)) + 1;
  const Scalar _tmp143 = _tmp142 / [&]() {
    const Scalar base = _tmp141;
    return base * base * base;
  }();
  const Scalar _tmp144 = _tmp138 * _tmp143;
  const Scalar _tmp145 = _tmp132 * _tmp61 + _tmp135 * _tmp46;
  const Scalar _tmp146 = _tmp133 * _tmp4 + _tmp145;
  const Scalar _tmp147 = std::pow(_tmp146, Scalar(2));
  const Scalar _tmp148 = _tmp143 * _tmp147;
  const Scalar _tmp149 = _tmp121 * _tmp85 + _tmp125 * _tmp84 + _tmp128 * _tmp86;
  const Scalar _tmp150 = std::pow(_tmp141, Scalar(-2));
  const Scalar _tmp151 = 2 * _tmp150;
  const Scalar _tmp152 = _tmp137 * _tmp151;
  const Scalar _tmp153 = _tmp121 * _tmp46 + _tmp125 * _tmp4 + _tmp128 * _tmp61;
  const Scalar _tmp154 = _tmp146 * _tmp151;
  const Scalar _tmp155 =
      -_tmp129 * _tmp144 - _tmp129 * _tmp148 + _tmp149 * _tmp152 + _tmp153 * _tmp154;
  const Scalar _tmp156 = _tmp137 * _tmp93;
  const Scalar _tmp157 = std::tan(_tmp89);
  const Scalar _tmp158 = Scalar(1.0) / (_tmp141);
  const Scalar _tmp159 = _tmp138 * _tmp150 + _tmp147 * _tmp150 + epsilon;
  const Scalar _tmp160 =
      _tmp157 * _tmp158 / (_tmp159 * (4 * std::pow(_tmp157, Scalar(2)) * _tmp159 + 1));
  const Scalar _tmp161 = _tmp156 * _tmp160;
  const Scalar _tmp162 = std::sqrt(_tmp159);
  const Scalar _tmp163 = Scalar(1.0) / (_tmp162);
  const Scalar _tmp164 = std::atan(2 * _tmp157 * _tmp162);
  const Scalar _tmp165 = _tmp158 * _tmp164;
  const Scalar _tmp166 = _tmp163 * _tmp165;
  const Scalar _tmp167 = _tmp166 * _tmp93;
  const Scalar _tmp168 = (Scalar(1) / Scalar(2)) * _tmp156;
  const Scalar _tmp169 = _tmp142 * _tmp150 * _tmp163 * _tmp164;
  const Scalar _tmp170 = _tmp168 * _tmp169;
  const Scalar _tmp171 = _tmp165 / (_tmp159 * std::sqrt(_tmp159));
  const Scalar _tmp172 = _tmp168 * _tmp171;
  const Scalar _tmp173 =
      -_tmp129 * _tmp170 + _tmp149 * _tmp167 + _tmp155 * _tmp161 - _tmp155 * _tmp172;
  const Scalar _tmp174 = _tmp156 * _tmp166 + _tmp94;
  const Scalar _tmp175 = 2 * _tmp174;
  const Scalar _tmp176 = _tmp146 * _tmp71;
  const Scalar _tmp177 = (Scalar(1) / Scalar(2)) * _tmp171;
  const Scalar _tmp178 = _tmp176 * _tmp177;
  const Scalar _tmp179 = _tmp166 * _tmp71;
  const Scalar _tmp180 = (Scalar(1) / Scalar(2)) * _tmp176;
  const Scalar _tmp181 = _tmp169 * _tmp180;
  const Scalar _tmp182 = _tmp160 * _tmp176;
  const Scalar _tmp183 =
      -_tmp129 * _tmp181 + _tmp153 * _tmp179 - _tmp155 * _tmp178 + _tmp155 * _tmp182;
  const Scalar _tmp184 = _tmp146 * _tmp179 + _tmp91;
  const Scalar _tmp185 = 2 * _tmp184;
  const Scalar _tmp186 = _tmp173 * _tmp175 + _tmp183 * _tmp185;
  const Scalar _tmp187 = (Scalar(1) / Scalar(2)) * _tmp184;
  const Scalar _tmp188 = std::max<Scalar>(0, (((_tmp140) > 0) - ((_tmp140) < 0)));
  const Scalar _tmp189 = std::pow(_tmp174, Scalar(2)) + std::pow(_tmp184, Scalar(2)) + epsilon;
  const Scalar _tmp190 = std::pow(_tmp189, Scalar(Scalar(-1) / Scalar(2)));
  const Scalar _tmp191 = _tmp100 * _tmp189 + 1;
  const Scalar _tmp192 = std::sqrt(Scalar(_tmp104 * (std::pow(_tmp191, _tmp103) - 1)));
  const Scalar _tmp193 =
      _tmp105 * _tmp188 * _tmp190 * std::pow(_tmp191, Scalar(_tmp103 - 1)) * _tmp99 / _tmp192;
  const Scalar _tmp194 = _tmp187 * _tmp193;
  const Scalar _tmp195 = _tmp105 * _tmp188 * _tmp192;
  const Scalar _tmp196 = _tmp190 * _tmp195;
  const Scalar _tmp197 = _tmp195 / (_tmp189 * std::sqrt(_tmp189));
  const Scalar _tmp198 = _tmp187 * _tmp197;
  const Scalar _tmp199 = _tmp183 * _tmp196 + _tmp186 * _tmp194 - _tmp186 * _tmp198;
  const Scalar _tmp200 = (Scalar(1) / Scalar(2)) * _tmp174;
  const Scalar _tmp201 = _tmp197 * _tmp200;
  const Scalar _tmp202 = _tmp193 * _tmp200;
  const Scalar _tmp203 = _tmp173 * _tmp196 - _tmp186 * _tmp201 + _tmp186 * _tmp202;
  const Scalar _tmp204 = _tmp118 * _tmp36;
  const Scalar _tmp205 = _tmp130 + _tmp204 * (_tmp120 + _tmp50);
  const Scalar _tmp206 = _tmp126 + _tmp34;
  const Scalar _tmp207 = -_tmp32;
  const Scalar _tmp208 = _tmp207 + _tmp47;
  const Scalar _tmp209 = _tmp134 + _tmp204 * (_tmp206 + _tmp208);
  const Scalar _tmp210 = -_tmp6;
  const Scalar _tmp211 = _tmp116 * (_tmp113 + _tmp208) + _tmp204 * (_tmp210 + _tmp55);
  const Scalar _tmp212 = _tmp205 * _tmp61 + _tmp209 * _tmp46 + _tmp211 * _tmp4;
  const Scalar _tmp213 = _tmp205 * _tmp79 + _tmp209 * _tmp75 + _tmp211 * _tmp72;
  const Scalar _tmp214 = _tmp205 * _tmp86 + _tmp209 * _tmp85 + _tmp211 * _tmp84;
  const Scalar _tmp215 =
      -_tmp144 * _tmp213 - _tmp148 * _tmp213 + _tmp152 * _tmp214 + _tmp154 * _tmp212;
  const Scalar _tmp216 =
      -_tmp178 * _tmp215 + _tmp179 * _tmp212 - _tmp181 * _tmp213 + _tmp182 * _tmp215;
  const Scalar _tmp217 =
      _tmp161 * _tmp215 + _tmp167 * _tmp214 - _tmp170 * _tmp213 - _tmp172 * _tmp215;
  const Scalar _tmp218 = _tmp175 * _tmp217 + _tmp185 * _tmp216;
  const Scalar _tmp219 = _tmp187 * _tmp218;
  const Scalar _tmp220 = _tmp193 * _tmp219 + _tmp196 * _tmp216 - _tmp197 * _tmp219;
  const Scalar _tmp221 = _tmp196 * _tmp217 - _tmp201 * _tmp218 + _tmp202 * _tmp218;
  const Scalar _tmp222 = _tmp119 * (_tmp110 + _tmp206) + _tmp204 * _tmp27;
  const Scalar _tmp223 = _tmp119 * (_tmp210 + _tmp7) + _tmp204 * _tmp52;
  const Scalar _tmp224 =
      _tmp119 * (_tmp122 + _tmp26) + _tmp204 * (_tmp109 + _tmp111 + _tmp207 + _tmp34);
  const Scalar _tmp225 = _tmp222 * _tmp4 + _tmp223 * _tmp46 + _tmp224 * _tmp61;
  const Scalar _tmp226 = _tmp222 * _tmp72 + _tmp223 * _tmp75 + _tmp224 * _tmp79;
  const Scalar _tmp227 = _tmp222 * _tmp84 + _tmp223 * _tmp85 + _tmp224 * _tmp86;
  const Scalar _tmp228 =
      -_tmp144 * _tmp226 - _tmp148 * _tmp226 + _tmp152 * _tmp227 + _tmp154 * _tmp225;
  const Scalar _tmp229 = _tmp169 * _tmp226;
  const Scalar _tmp230 =
      _tmp161 * _tmp228 + _tmp167 * _tmp227 - _tmp168 * _tmp229 - _tmp172 * _tmp228;
  const Scalar _tmp231 =
      -_tmp178 * _tmp228 + _tmp179 * _tmp225 - _tmp180 * _tmp229 + _tmp182 * _tmp228;
  const Scalar _tmp232 = _tmp175 * _tmp230 + _tmp185 * _tmp231;
  const Scalar _tmp233 = _tmp193 * _tmp232;
  const Scalar _tmp234 = _tmp187 * _tmp233 + _tmp196 * _tmp231 - _tmp198 * _tmp232;
  const Scalar _tmp235 = _tmp196 * _tmp230 + _tmp200 * _tmp233 - _tmp201 * _tmp232;
  const Scalar _tmp236 = _tmp144 * source_inverse_range;
  const Scalar _tmp237 = _tmp236 * _tmp72;
  const Scalar _tmp238 = _tmp152 * source_inverse_range;
  const Scalar _tmp239 = _tmp238 * _tmp84;
  const Scalar _tmp240 = _tmp148 * source_inverse_range;
  const Scalar _tmp241 = _tmp240 * _tmp72;
  const Scalar _tmp242 = _tmp154 * source_inverse_range;
  const Scalar _tmp243 = _tmp242 * _tmp4;
  const Scalar _tmp244 = -_tmp237 + _tmp239 - _tmp241 + _tmp243;
  const Scalar _tmp245 = _tmp181 * source_inverse_range;
  const Scalar _tmp246 = _tmp245 * _tmp72;
  const Scalar _tmp247 = _tmp179 * source_inverse_range;
  const Scalar _tmp248 = _tmp247 * _tmp4;
  const Scalar _tmp249 = -_tmp178 * _tmp244 + _tmp182 * _tmp244 - _tmp246 + _tmp248;
  const Scalar _tmp250 = _tmp167 * source_inverse_range;
  const Scalar _tmp251 = _tmp250 * _tmp84;
  const Scalar _tmp252 = _tmp170 * source_inverse_range;
  const Scalar _tmp253 = _tmp252 * _tmp72;
  const Scalar _tmp254 = _tmp161 * _tmp244 - _tmp172 * _tmp244 + _tmp251 - _tmp253;
  const Scalar _tmp255 = _tmp175 * _tmp254 + _tmp185 * _tmp249;
  const Scalar _tmp256 = _tmp194 * _tmp255 + _tmp196 * _tmp249 - _tmp198 * _tmp255;
  const Scalar _tmp257 = _tmp196 * _tmp254 - _tmp201 * _tmp255 + _tmp202 * _tmp255;
  const Scalar _tmp258 = _tmp247 * _tmp61;
  const Scalar _tmp259 = _tmp238 * _tmp86;
  const Scalar _tmp260 = _tmp242 * _tmp61;
  const Scalar _tmp261 = _tmp240 * _tmp79;
  const Scalar _tmp262 = _tmp236 * _tmp79;
  const Scalar _tmp263 = _tmp259 + _tmp260 - _tmp261 - _tmp262;
  const Scalar _tmp264 = _tmp245 * _tmp79;
  const Scalar _tmp265 = -_tmp178 * _tmp263 + _tmp182 * _tmp263 + _tmp258 - _tmp264;
  const Scalar _tmp266 = _tmp250 * _tmp86;
  const Scalar _tmp267 = _tmp252 * _tmp79;
  const Scalar _tmp268 = _tmp161 * _tmp263 - _tmp172 * _tmp263 + _tmp266 - _tmp267;
  const Scalar _tmp269 = _tmp175 * _tmp268 + _tmp185 * _tmp265;
  const Scalar _tmp270 = _tmp194 * _tmp269 + _tmp196 * _tmp265 - _tmp198 * _tmp269;
  const Scalar _tmp271 = _tmp196 * _tmp268 - _tmp201 * _tmp269 + _tmp202 * _tmp269;
  const Scalar _tmp272 = _tmp238 * _tmp85;
  const Scalar _tmp273 = _tmp242 * _tmp46;
  const Scalar _tmp274 = _tmp236 * _tmp75;
  const Scalar _tmp275 = _tmp240 * _tmp75;
  const Scalar _tmp276 = _tmp272 + _tmp273 - _tmp274 - _tmp275;
  const Scalar _tmp277 = _tmp245 * _tmp75;
  const Scalar _tmp278 = _tmp247 * _tmp46;
  const Scalar _tmp279 = -_tmp178 * _tmp276 + _tmp182 * _tmp276 - _tmp277 + _tmp278;
  const Scalar _tmp280 = _tmp252 * _tmp75;
  const Scalar _tmp281 = _tmp250 * _tmp85;
  const Scalar _tmp282 = _tmp161 * _tmp276 - _tmp172 * _tmp276 - _tmp280 + _tmp281;
  const Scalar _tmp283 = _tmp175 * _tmp282 + _tmp185 * _tmp279;
  const Scalar _tmp284 = _tmp194 * _tmp283 + _tmp196 * _tmp279 - _tmp198 * _tmp283;
  const Scalar _tmp285 = _tmp196 * _tmp282 - _tmp201 * _tmp283 + _tmp202 * _tmp283;
  const Scalar _tmp286 = -_tmp60;
  const Scalar _tmp287 = std::pow(_target_pose[3], Scalar(2));
  const Scalar _tmp288 = -_tmp287;
  const Scalar _tmp289 = _tmp288 + _tmp73;
  const Scalar _tmp290 = -_tmp2;
  const Scalar _tmp291 = _tmp0 + _tmp290;
  const Scalar _tmp292 = -_tmp76;
  const Scalar _tmp293 =
      _tmp132 * (_tmp289 + _tmp291) + _tmp133 * (_tmp286 + _tmp59) + _tmp135 * (_tmp292 + _tmp78);
  const Scalar _tmp294 = -_tmp73;
  const Scalar _tmp295 = _tmp287 + _tmp294;
  const Scalar _tmp296 = _tmp135 * (_tmp291 + _tmp295) + _tmp139;
  const Scalar _tmp297 = -_tmp144 * _tmp293 - _tmp148 * _tmp293 + _tmp152 * _tmp296;
  const Scalar _tmp298 = -_tmp178 * _tmp297 - _tmp181 * _tmp293 + _tmp182 * _tmp297;
  const Scalar _tmp299 =
      _tmp161 * _tmp297 + _tmp167 * _tmp296 - _tmp170 * _tmp293 - _tmp172 * _tmp297;
  const Scalar _tmp300 = _tmp175 * _tmp299 + _tmp185 * _tmp298;
  const Scalar _tmp301 = _tmp194 * _tmp300 + _tmp196 * _tmp298 - _tmp198 * _tmp300;
  const Scalar _tmp302 = _tmp196 * _tmp299 - _tmp201 * _tmp300 + _tmp202 * _tmp300;
  const Scalar _tmp303 = -_tmp0;
  const Scalar _tmp304 = _tmp133 * (_tmp287 + _tmp290 + _tmp303 + _tmp73) + _tmp145;
  const Scalar _tmp305 = -_tmp42;
  const Scalar _tmp306 = _tmp2 + _tmp303;
  const Scalar _tmp307 =
      _tmp132 * (_tmp292 + _tmp77) + _tmp133 * (_tmp305 + _tmp45) + _tmp135 * (_tmp289 + _tmp306);
  const Scalar _tmp308 = -_tmp144 * _tmp304 - _tmp148 * _tmp304 + _tmp154 * _tmp307;
  const Scalar _tmp309 =
      -_tmp178 * _tmp308 + _tmp179 * _tmp307 - _tmp181 * _tmp304 + _tmp182 * _tmp308;
  const Scalar _tmp310 = _tmp156 * _tmp308;
  const Scalar _tmp311 = _tmp160 * _tmp310 - _tmp170 * _tmp304 - _tmp177 * _tmp310;
  const Scalar _tmp312 = _tmp175 * _tmp311 + _tmp185 * _tmp309;
  const Scalar _tmp313 = _tmp194 * _tmp312 + _tmp196 * _tmp309 - _tmp198 * _tmp312;
  const Scalar _tmp314 = _tmp196 * _tmp311 - _tmp201 * _tmp312 + _tmp202 * _tmp312;
  const Scalar _tmp315 = _tmp132 * (_tmp295 + _tmp306) + _tmp136;
  const Scalar _tmp316 = _tmp132 * (_tmp286 + _tmp83) +
                         _tmp133 * (_tmp0 + _tmp2 + _tmp288 + _tmp294) +
                         _tmp135 * (_tmp305 + _tmp44);
  const Scalar _tmp317 = _tmp152 * _tmp316 + _tmp154 * _tmp315;
  const Scalar _tmp318 = -_tmp178 * _tmp317 + _tmp179 * _tmp315 + _tmp182 * _tmp317;
  const Scalar _tmp319 = _tmp161 * _tmp317 + _tmp167 * _tmp316 - _tmp172 * _tmp317;
  const Scalar _tmp320 = _tmp175 * _tmp319 + _tmp185 * _tmp318;
  const Scalar _tmp321 = _tmp194 * _tmp320 + _tmp196 * _tmp318 - _tmp198 * _tmp320;
  const Scalar _tmp322 = _tmp196 * _tmp319 - _tmp201 * _tmp320 + _tmp202 * _tmp320;
  const Scalar _tmp323 = _tmp237 - _tmp239 + _tmp241 - _tmp243;
  const Scalar _tmp324 = -_tmp178 * _tmp323 + _tmp182 * _tmp323 + _tmp246 - _tmp248;
  const Scalar _tmp325 = _tmp161 * _tmp323 - _tmp172 * _tmp323 - _tmp251 + _tmp253;
  const Scalar _tmp326 = _tmp175 * _tmp325 + _tmp185 * _tmp324;
  const Scalar _tmp327 = _tmp194 * _tmp326 + _tmp196 * _tmp324 - _tmp198 * _tmp326;
  const Scalar _tmp328 = _tmp196 * _tmp325 - _tmp201 * _tmp326 + _tmp202 * _tmp326;
  const Scalar _tmp329 = -_tmp259 - _tmp260 + _tmp261 + _tmp262;
  const Scalar _tmp330 = _tmp161 * _tmp329 - _tmp172 * _tmp329 - _tmp266 + _tmp267;
  const Scalar _tmp331 = -_tmp178 * _tmp329 + _tmp182 * _tmp329 - _tmp258 + _tmp264;
  const Scalar _tmp332 = _tmp175 * _tmp330 + _tmp185 * _tmp331;
  const Scalar _tmp333 = _tmp194 * _tmp332 + _tmp196 * _tmp331 - _tmp198 * _tmp332;
  const Scalar _tmp334 = _tmp196 * _tmp330 - _tmp201 * _tmp332 + _tmp202 * _tmp332;
  const Scalar _tmp335 = -_tmp272 - _tmp273 + _tmp274 + _tmp275;
  const Scalar _tmp336 = -_tmp178 * _tmp335 + _tmp182 * _tmp335 + _tmp277 - _tmp278;
  const Scalar _tmp337 = _tmp161 * _tmp335 - _tmp172 * _tmp335 + _tmp280 - _tmp281;
  const Scalar _tmp338 = _tmp175 * _tmp337 + _tmp185 * _tmp336;
  const Scalar _tmp339 = _tmp197 * _tmp338;
  const Scalar _tmp340 = -_tmp187 * _tmp339 + _tmp194 * _tmp338 + _tmp196 * _tmp336;
  const Scalar _tmp341 = _tmp196 * _tmp337 - _tmp200 * _tmp339 + _tmp202 * _tmp338;
  const Scalar _tmp342 = _tmp38 * _tmp84 + _tmp53 * _tmp85 + _tmp66 * _tmp86;
  const Scalar _tmp343 = _tmp38 * _tmp4 + _tmp46 * _tmp53 + _tmp61 * _tmp66;
  const Scalar _tmp344 = _tmp38 * _tmp72 + _tmp53 * _tmp75 + _tmp66 * _tmp79;
  const Scalar _tmp345 =
      -_tmp144 * _tmp344 - _tmp148 * _tmp344 + _tmp152 * _tmp342 + _tmp154 * _tmp343;
  const Scalar _tmp346 =
      _tmp161 * _tmp345 + _tmp167 * _tmp342 - _tmp170 * _tmp344 - _tmp172 * _tmp345;
  const Scalar _tmp347 =
      -_tmp178 * _tmp345 + _tmp179 * _tmp343 - _tmp181 * _tmp344 + _tmp182 * _tmp345;
  const Scalar _tmp348 = _tmp175 * _tmp346 + _tmp185 * _tmp347;
  const Scalar _tmp349 = _tmp194 * _tmp348 + _tmp196 * _tmp347 - _tmp198 * _tmp348;
  const Scalar _tmp350 = _tmp196 * _tmp346 - _tmp201 * _tmp348 + _tmp202 * _tmp348;

  // Output terms (4)
  if (res != nullptr) {
    Eigen::Matrix<Scalar, 2, 1>& _res = (*res);

    _res(0, 0) = _tmp107;
    _res(1, 0) = _tmp108;
  }

  if (jacobian != nullptr) {
    Eigen::Matrix<Scalar, 2, 13>& _jacobian = (*jacobian);

    _jacobian(0, 0) = _tmp199;
    _jacobian(1, 0) = _tmp203;
    _jacobian(0, 1) = _tmp220;
    _jacobian(1, 1) = _tmp221;
    _jacobian(0, 2) = _tmp234;
    _jacobian(1, 2) = _tmp235;
    _jacobian(0, 3) = _tmp256;
    _jacobian(1, 3) = _tmp257;
    _jacobian(0, 4) = _tmp270;
    _jacobian(1, 4) = _tmp271;
    _jacobian(0, 5) = _tmp284;
    _jacobian(1, 5) = _tmp285;
    _jacobian(0, 6) = _tmp301;
    _jacobian(1, 6) = _tmp302;
    _jacobian(0, 7) = _tmp313;
    _jacobian(1, 7) = _tmp314;
    _jacobian(0, 8) = _tmp321;
    _jacobian(1, 8) = _tmp322;
    _jacobian(0, 9) = _tmp327;
    _jacobian(1, 9) = _tmp328;
    _jacobian(0, 10) = _tmp333;
    _jacobian(1, 10) = _tmp334;
    _jacobian(0, 11) = _tmp340;
    _jacobian(1, 11) = _tmp341;
    _jacobian(0, 12) = _tmp349;
    _jacobian(1, 12) = _tmp350;
  }

  if (hessian != nullptr) {
    Eigen::Matrix<Scalar, 13, 13>& _hessian = (*hessian);

    _hessian(0, 0) = std::pow(_tmp199, Scalar(2)) + std::pow(_tmp203, Scalar(2));
    _hessian(1, 0) = _tmp199 * _tmp220 + _tmp203 * _tmp221;
    _hessian(2, 0) = _tmp199 * _tmp234 + _tmp203 * _tmp235;
    _hessian(3, 0) = _tmp199 * _tmp256 + _tmp203 * _tmp257;
    _hessian(4, 0) = _tmp199 * _tmp270 + _tmp203 * _tmp271;
    _hessian(5, 0) = _tmp199 * _tmp284 + _tmp203 * _tmp285;
    _hessian(6, 0) = _tmp199 * _tmp301 + _tmp203 * _tmp302;
    _hessian(7, 0) = _tmp199 * _tmp313 + _tmp203 * _tmp314;
    _hessian(8, 0) = _tmp199 * _tmp321 + _tmp203 * _tmp322;
    _hessian(9, 0) = _tmp199 * _tmp327 + _tmp203 * _tmp328;
    _hessian(10, 0) = _tmp199 * _tmp333 + _tmp203 * _tmp334;
    _hessian(11, 0) = _tmp199 * _tmp340 + _tmp203 * _tmp341;
    _hessian(12, 0) = _tmp199 * _tmp349 + _tmp203 * _tmp350;
    _hessian(0, 1) = 0;
    _hessian(1, 1) = std::pow(_tmp220, Scalar(2)) + std::pow(_tmp221, Scalar(2));
    _hessian(2, 1) = _tmp220 * _tmp234 + _tmp221 * _tmp235;
    _hessian(3, 1) = _tmp220 * _tmp256 + _tmp221 * _tmp257;
    _hessian(4, 1) = _tmp220 * _tmp270 + _tmp221 * _tmp271;
    _hessian(5, 1) = _tmp220 * _tmp284 + _tmp221 * _tmp285;
    _hessian(6, 1) = _tmp220 * _tmp301 + _tmp221 * _tmp302;
    _hessian(7, 1) = _tmp220 * _tmp313 + _tmp221 * _tmp314;
    _hessian(8, 1) = _tmp220 * _tmp321 + _tmp221 * _tmp322;
    _hessian(9, 1) = _tmp220 * _tmp327 + _tmp221 * _tmp328;
    _hessian(10, 1) = _tmp220 * _tmp333 + _tmp221 * _tmp334;
    _hessian(11, 1) = _tmp220 * _tmp340 + _tmp221 * _tmp341;
    _hessian(12, 1) = _tmp220 * _tmp349 + _tmp221 * _tmp350;
    _hessian(0, 2) = 0;
    _hessian(1, 2) = 0;
    _hessian(2, 2) = std::pow(_tmp234, Scalar(2)) + std::pow(_tmp235, Scalar(2));
    _hessian(3, 2) = _tmp234 * _tmp256 + _tmp235 * _tmp257;
    _hessian(4, 2) = _tmp234 * _tmp270 + _tmp235 * _tmp271;
    _hessian(5, 2) = _tmp234 * _tmp284 + _tmp235 * _tmp285;
    _hessian(6, 2) = _tmp234 * _tmp301 + _tmp235 * _tmp302;
    _hessian(7, 2) = _tmp234 * _tmp313 + _tmp235 * _tmp314;
    _hessian(8, 2) = _tmp234 * _tmp321 + _tmp235 * _tmp322;
    _hessian(9, 2) = _tmp234 * _tmp327 + _tmp235 * _tmp328;
    _hessian(10, 2) = _tmp234 * _tmp333 + _tmp235 * _tmp334;
    _hessian(11, 2) = _tmp234 * _tmp340 + _tmp235 * _tmp341;
    _hessian(12, 2) = _tmp234 * _tmp349 + _tmp235 * _tmp350;
    _hessian(0, 3) = 0;
    _hessian(1, 3) = 0;
    _hessian(2, 3) = 0;
    _hessian(3, 3) = std::pow(_tmp256, Scalar(2)) + std::pow(_tmp257, Scalar(2));
    _hessian(4, 3) = _tmp256 * _tmp270 + _tmp257 * _tmp271;
    _hessian(5, 3) = _tmp256 * _tmp284 + _tmp257 * _tmp285;
    _hessian(6, 3) = _tmp256 * _tmp301 + _tmp257 * _tmp302;
    _hessian(7, 3) = _tmp256 * _tmp313 + _tmp257 * _tmp314;
    _hessian(8, 3) = _tmp256 * _tmp321 + _tmp257 * _tmp322;
    _hessian(9, 3) = _tmp256 * _tmp327 + _tmp257 * _tmp328;
    _hessian(10, 3) = _tmp256 * _tmp333 + _tmp257 * _tmp334;
    _hessian(11, 3) = _tmp256 * _tmp340 + _tmp257 * _tmp341;
    _hessian(12, 3) = _tmp256 * _tmp349 + _tmp257 * _tmp350;
    _hessian(0, 4) = 0;
    _hessian(1, 4) = 0;
    _hessian(2, 4) = 0;
    _hessian(3, 4) = 0;
    _hessian(4, 4) = std::pow(_tmp270, Scalar(2)) + std::pow(_tmp271, Scalar(2));
    _hessian(5, 4) = _tmp270 * _tmp284 + _tmp271 * _tmp285;
    _hessian(6, 4) = _tmp270 * _tmp301 + _tmp271 * _tmp302;
    _hessian(7, 4) = _tmp270 * _tmp313 + _tmp271 * _tmp314;
    _hessian(8, 4) = _tmp270 * _tmp321 + _tmp271 * _tmp322;
    _hessian(9, 4) = _tmp270 * _tmp327 + _tmp271 * _tmp328;
    _hessian(10, 4) = _tmp270 * _tmp333 + _tmp271 * _tmp334;
    _hessian(11, 4) = _tmp270 * _tmp340 + _tmp271 * _tmp341;
    _hessian(12, 4) = _tmp270 * _tmp349 + _tmp271 * _tmp350;
    _hessian(0, 5) = 0;
    _hessian(1, 5) = 0;
    _hessian(2, 5) = 0;
    _hessian(3, 5) = 0;
    _hessian(4, 5) = 0;
    _hessian(5, 5) = std::pow(_tmp284, Scalar(2)) + std::pow(_tmp285, Scalar(2));
    _hessian(6, 5) = _tmp284 * _tmp301 + _tmp285 * _tmp302;
    _hessian(7, 5) = _tmp284 * _tmp313 + _tmp285 * _tmp314;
    _hessian(8, 5) = _tmp284 * _tmp321 + _tmp285 * _tmp322;
    _hessian(9, 5) = _tmp284 * _tmp327 + _tmp285 * _tmp328;
    _hessian(10, 5) = _tmp284 * _tmp333 + _tmp285 * _tmp334;
    _hessian(11, 5) = _tmp284 * _tmp340 + _tmp285 * _tmp341;
    _hessian(12, 5) = _tmp284 * _tmp349 + _tmp285 * _tmp350;
    _hessian(0, 6) = 0;
    _hessian(1, 6) = 0;
    _hessian(2, 6) = 0;
    _hessian(3, 6) = 0;
    _hessian(4, 6) = 0;
    _hessian(5, 6) = 0;
    _hessian(6, 6) = std::pow(_tmp301, Scalar(2)) + std::pow(_tmp302, Scalar(2));
    _hessian(7, 6) = _tmp301 * _tmp313 + _tmp302 * _tmp314;
    _hessian(8, 6) = _tmp301 * _tmp321 + _tmp302 * _tmp322;
    _hessian(9, 6) = _tmp301 * _tmp327 + _tmp302 * _tmp328;
    _hessian(10, 6) = _tmp301 * _tmp333 + _tmp302 * _tmp334;
    _hessian(11, 6) = _tmp301 * _tmp340 + _tmp302 * _tmp341;
    _hessian(12, 6) = _tmp301 * _tmp349 + _tmp302 * _tmp350;
    _hessian(0, 7) = 0;
    _hessian(1, 7) = 0;
    _hessian(2, 7) = 0;
    _hessian(3, 7) = 0;
    _hessian(4, 7) = 0;
    _hessian(5, 7) = 0;
    _hessian(6, 7) = 0;
    _hessian(7, 7) = std::pow(_tmp313, Scalar(2)) + std::pow(_tmp314, Scalar(2));
    _hessian(8, 7) = _tmp313 * _tmp321 + _tmp314 * _tmp322;
    _hessian(9, 7) = _tmp313 * _tmp327 + _tmp314 * _tmp328;
    _hessian(10, 7) = _tmp313 * _tmp333 + _tmp314 * _tmp334;
    _hessian(11, 7) = _tmp313 * _tmp340 + _tmp314 * _tmp341;
    _hessian(12, 7) = _tmp313 * _tmp349 + _tmp314 * _tmp350;
    _hessian(0, 8) = 0;
    _hessian(1, 8) = 0;
    _hessian(2, 8) = 0;
    _hessian(3, 8) = 0;
    _hessian(4, 8) = 0;
    _hessian(5, 8) = 0;
    _hessian(6, 8) = 0;
    _hessian(7, 8) = 0;
    _hessian(8, 8) = std::pow(_tmp321, Scalar(2)) + std::pow(_tmp322, Scalar(2));
    _hessian(9, 8) = _tmp321 * _tmp327 + _tmp322 * _tmp328;
    _hessian(10, 8) = _tmp321 * _tmp333 + _tmp322 * _tmp334;
    _hessian(11, 8) = _tmp321 * _tmp340 + _tmp322 * _tmp341;
    _hessian(12, 8) = _tmp321 * _tmp349 + _tmp322 * _tmp350;
    _hessian(0, 9) = 0;
    _hessian(1, 9) = 0;
    _hessian(2, 9) = 0;
    _hessian(3, 9) = 0;
    _hessian(4, 9) = 0;
    _hessian(5, 9) = 0;
    _hessian(6, 9) = 0;
    _hessian(7, 9) = 0;
    _hessian(8, 9) = 0;
    _hessian(9, 9) = std::pow(_tmp327, Scalar(2)) + std::pow(_tmp328, Scalar(2));
    _hessian(10, 9) = _tmp327 * _tmp333 + _tmp328 * _tmp334;
    _hessian(11, 9) = _tmp327 * _tmp340 + _tmp328 * _tmp341;
    _hessian(12, 9) = _tmp327 * _tmp349 + _tmp328 * _tmp350;
    _hessian(0, 10) = 0;
    _hessian(1, 10) = 0;
    _hessian(2, 10) = 0;
    _hessian(3, 10) = 0;
    _hessian(4, 10) = 0;
    _hessian(5, 10) = 0;
    _hessian(6, 10) = 0;
    _hessian(7, 10) = 0;
    _hessian(8, 10) = 0;
    _hessian(9, 10) = 0;
    _hessian(10, 10) = std::pow(_tmp333, Scalar(2)) + std::pow(_tmp334, Scalar(2));
    _hessian(11, 10) = _tmp333 * _tmp340 + _tmp334 * _tmp341;
    _hessian(12, 10) = _tmp333 * _tmp349 + _tmp334 * _tmp350;
    _hessian(0, 11) = 0;
    _hessian(1, 11) = 0;
    _hessian(2, 11) = 0;
    _hessian(3, 11) = 0;
    _hessian(4, 11) = 0;
    _hessian(5, 11) = 0;
    _hessian(6, 11) = 0;
    _hessian(7, 11) = 0;
    _hessian(8, 11) = 0;
    _hessian(9, 11) = 0;
    _hessian(10, 11) = 0;
    _hessian(11, 11) = std::pow(_tmp340, Scalar(2)) + std::pow(_tmp341, Scalar(2));
    _hessian(12, 11) = _tmp340 * _tmp349 + _tmp341 * _tmp350;
    _hessian(0, 12) = 0;
    _hessian(1, 12) = 0;
    _hessian(2, 12) = 0;
    _hessian(3, 12) = 0;
    _hessian(4, 12) = 0;
    _hessian(5, 12) = 0;
    _hessian(6, 12) = 0;
    _hessian(7, 12) = 0;
    _hessian(8, 12) = 0;
    _hessian(9, 12) = 0;
    _hessian(10, 12) = 0;
    _hessian(11, 12) = 0;
    _hessian(12, 12) = std::pow(_tmp349, Scalar(2)) + std::pow(_tmp350, Scalar(2));
  }

  if (rhs != nullptr) {
    Eigen::Matrix<Scalar, 13, 1>& _rhs = (*rhs);

    _rhs(0, 0) = _tmp107 * _tmp199 + _tmp108 * _tmp203;
    _rhs(1, 0) = _tmp107 * _tmp220 + _tmp108 * _tmp221;
    _rhs(2, 0) = _tmp107 * _tmp234 + _tmp108 * _tmp235;
    _rhs(3, 0) = _tmp107 * _tmp256 + _tmp108 * _tmp257;
    _rhs(4, 0) = _tmp107 * _tmp270 + _tmp108 * _tmp271;
    _rhs(5, 0) = _tmp107 * _tmp284 + _tmp108 * _tmp285;
    _rhs(6, 0) = _tmp107 * _tmp301 + _tmp108 * _tmp302;
    _rhs(7, 0) = _tmp107 * _tmp313 + _tmp108 * _tmp314;
    _rhs(8, 0) = _tmp107 * _tmp321 + _tmp108 * _tmp322;
    _rhs(9, 0) = _tmp107 * _tmp327 + _tmp108 * _tmp328;
    _rhs(10, 0) = _tmp107 * _tmp333 + _tmp108 * _tmp334;
    _rhs(11, 0) = _tmp107 * _tmp340 + _tmp108 * _tmp341;
    _rhs(12, 0) = _tmp107 * _tmp349 + _tmp108 * _tmp350;
  }
}  // NOLINT(readability/fn_size)

// NOLINTNEXTLINE(readability/fn_size)
}  // namespace sym
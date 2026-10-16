#ifndef TrackingTools_AnalyticalJacobians_interface_curvilinearJacobian_h
#define TrackingTools_AnalyticalJacobians_interface_curvilinearJacobian_h

#include <cmath>

/*
 * The Jacobian of AnalyticalCurvilinearJacobian (curvilinear frame, helix in a constant field) as plain
 * constexpr functions, usable on the host and on GPUs: the scalar TRPRFN variant of computeFullJacobian and
 * computeStraightLineJacobian. Positions in cm, momenta in GeV, as GlobalPoint / GlobalVector (float).
 */
namespace curvilinearJacobian {

  struct Vector3 {
    float x, y, z;
  };

  constexpr float mag2(Vector3 vec) { return vec.x * vec.x + vec.y * vec.y + vec.z * vec.z; }
  constexpr float perp(Vector3 vec) { return std::sqrt(vec.x * vec.x + vec.y * vec.y); }
  constexpr Vector3 unit(Vector3 vec) {
    const float vecMag2 = mag2(vec);
    if (vecMag2 == 0.f)
      return vec;
    const float scale = 1.f / std::sqrt(vecMag2);
    return Vector3{vec.x * scale, vec.y * scale, vec.z * scale};
  }

  // from the start state (x1, p1) with q/|p| = qbp to (x2, p2) after pathLength (cm), in the field
  // fieldInInverseGeV (tesla * 2.99792458e-3) at the start; sin/cos from std (the host uses vdt)
  constexpr void fullJacobian(Vector3 x1,
                              Vector3 p1,
                              double qbp,
                              Vector3 x2,
                              Vector3 p2,
                              Vector3 fieldInInverseGeV,
                              double pathLength,
                              double jacobian[5][5]) {
    const Vector3 p1Unit = unit(p1);
    const Vector3 p2Unit = unit(p2);
    const Vector3 dx{x1.x - x2.x, x1.y - x2.y, x1.z - x2.z};

    double t11 = p1Unit.x;
    double t12 = p1Unit.y;
    double t13 = p1Unit.z;
    double t21 = p2Unit.x;
    double t22 = p2Unit.y;
    double t23 = p2Unit.z;
    double cosl0 = perp(p1Unit);
    double cosl1 = 1. / perp(p2Unit);

    const Vector3 hn = unit(fieldInInverseGeV);
    const double fieldMag = std::sqrt(mag2(fieldInInverseGeV));
    double qp = -fieldMag;
    // signed curvature and turning angle
    double curvature = qp * qbp;
    double theta = curvature * pathLength;
    double sint = std::sin(theta);
    double cost = std::cos(theta);

    double hn1 = hn.x;
    double hn2 = hn.y;
    double hn3 = hn.z;
    double dx1 = dx.x;
    double dx2 = dx.y;
    double dx3 = dx.z;
    double gamma = hn1 * t21 + hn2 * t22 + hn3 * t23;
    double an1 = hn2 * t23 - hn3 * t22;
    double an2 = hn3 * t21 - hn1 * t23;
    double an3 = hn1 * t22 - hn2 * t21;

    double au = 1. / std::sqrt(t11 * t11 + t12 * t12);
    double u11 = -au * t12;
    double u12 = au * t11;
    double v11 = -t13 * u12;
    double v12 = t13 * u11;
    double v13 = t11 * u12 - t12 * u11;
    au = 1. / std::sqrt(t21 * t21 + t22 * t22);
    double u21 = -au * t22;
    double u22 = au * t21;
    double v21 = -t23 * u22;
    double v22 = t23 * u21;
    double v23 = t21 * u22 - t22 * u21;

    double anv = -(hn1 * u21 + hn2 * u22);
    double anu = (hn1 * v21 + hn2 * v22 + hn3 * v23);
    double omcost = 1. - cost;
    double tmsint = theta - sint;

    double hu1 = -hn3 * u12;
    double hu2 = hn3 * u11;
    double hu3 = hn1 * u12 - hn2 * u11;

    double hv1 = hn2 * v13 - hn3 * v12;
    double hv2 = hn3 * v11 - hn1 * v13;
    double hv3 = hn1 * v12 - hn2 * v11;

    // 1/p does not change
    jacobian[0][0] = 1.;
    for (int i = 1; i < 5; ++i)
      jacobian[0][i] = 0.;

    // lambda
    jacobian[1][0] = -qp * anv * (t21 * dx1 + t22 * dx2 + t23 * dx3);
    jacobian[1][1] = cost * (v11 * v21 + v12 * v22 + v13 * v23) + sint * (hv1 * v21 + hv2 * v22 + hv3 * v23) +
                     omcost * (hn1 * v11 + hn2 * v12 + hn3 * v13) * (hn1 * v21 + hn2 * v22 + hn3 * v23) +
                     anv * (-sint * (v11 * t21 + v12 * t22 + v13 * t23) + omcost * (v11 * an1 + v12 * an2 + v13 * an3) -
                            tmsint * gamma * (hn1 * v11 + hn2 * v12 + hn3 * v13));
    jacobian[1][2] = cost * (u11 * v21 + u12 * v22) + sint * (hu1 * v21 + hu2 * v22 + hu3 * v23) +
                     omcost * (hn1 * u11 + hn2 * u12) * (hn1 * v21 + hn2 * v22 + hn3 * v23) +
                     anv * (-sint * (u11 * t21 + u12 * t22) + omcost * (u11 * an1 + u12 * an2) -
                            tmsint * gamma * (hn1 * u11 + hn2 * u12));
    jacobian[1][2] *= cosl0;
    jacobian[1][3] = -curvature * anv * (u11 * t21 + u12 * t22);
    jacobian[1][4] = -curvature * anv * (v11 * t21 + v12 * t22 + v13 * t23);

    // phi
    jacobian[2][0] = -qp * anu * (t21 * dx1 + t22 * dx2 + t23 * dx3) * cosl1;
    jacobian[2][1] = cost * (v11 * u21 + v12 * u22) + sint * (hv1 * u21 + hv2 * u22) +
                     omcost * (hn1 * v11 + hn2 * v12 + hn3 * v13) * (hn1 * u21 + hn2 * u22) +
                     anu * (-sint * (v11 * t21 + v12 * t22 + v13 * t23) + omcost * (v11 * an1 + v12 * an2 + v13 * an3) -
                            tmsint * gamma * (hn1 * v11 + hn2 * v12 + hn3 * v13));
    jacobian[2][1] *= cosl1;
    jacobian[2][2] = cost * (u11 * u21 + u12 * u22) + sint * (hu1 * u21 + hu2 * u22) +
                     omcost * (hn1 * u11 + hn2 * u12) * (hn1 * u21 + hn2 * u22) +
                     anu * (-sint * (u11 * t21 + u12 * t22) + omcost * (u11 * an1 + u12 * an2) -
                            tmsint * gamma * (hn1 * u11 + hn2 * u12));
    jacobian[2][2] *= cosl1 * cosl0;
    jacobian[2][3] = -curvature * anu * (u11 * t21 + u12 * t22) * cosl1;
    jacobian[2][4] = -curvature * anu * (v11 * t21 + v12 * t22 + v13 * t23) * cosl1;

    // yt
    double cutCriterion = std::fabs(pathLength / std::sqrt(mag2(p1)));
    const double limit = 5.;  // valid for propagations with effectively float precision
    if (cutCriterion > limit) {
      double pp = 1. / qbp;
      jacobian[3][0] = pp * (u21 * dx1 + u22 * dx2);
      jacobian[4][0] = pp * (v21 * dx1 + v22 * dx2 + v23 * dx3);
    } else {
      double hp11 = hn2 * t13 - hn3 * t12;
      double hp12 = hn3 * t11 - hn1 * t13;
      double hp13 = hn1 * t12 - hn2 * t11;
      double temp1 = hp11 * u21 + hp12 * u22;
      double pathLength2 = pathLength * pathLength;
      double secondOrder41 = 0.5 * qp * temp1 * pathLength2;
      double ghnmp1 = gamma * hn1 - t11;
      double ghnmp2 = gamma * hn2 - t12;
      double ghnmp3 = gamma * hn3 - t13;
      double temp2 = ghnmp1 * u21 + ghnmp2 * u22;
      double pathLength3 = pathLength2 * pathLength;
      double pathLength4 = pathLength3 * pathLength;
      double h1 = fieldMag;
      double h2 = h1 * h1;
      double h3 = h2 * h1;
      double qbp2 = qbp * qbp;
      double thirdOrder41 = 1. / 3 * h2 * pathLength3 * qbp * temp2;
      double fourthOrder41 = 1. / 8 * h3 * pathLength4 * qbp2 * temp1;
      jacobian[3][0] = secondOrder41 + (thirdOrder41 + fourthOrder41);

      double temp3 = hp11 * v21 + hp12 * v22 + hp13 * v23;
      double secondOrder51 = 0.5 * qp * temp3 * pathLength2;
      double temp4 = ghnmp1 * v21 + ghnmp2 * v22 + ghnmp3 * v23;
      double thirdOrder51 = 1. / 3 * h2 * pathLength3 * qbp * temp4;
      double fourthOrder51 = 1. / 8 * h3 * pathLength4 * qbp2 * temp3;
      jacobian[4][0] = secondOrder51 + (thirdOrder51 + fourthOrder51);
    }

    jacobian[3][1] = (sint * (v11 * u21 + v12 * u22) + omcost * (hv1 * u21 + hv2 * u22) +
                      tmsint * (hn1 * u21 + hn2 * u22) * (hn1 * v11 + hn2 * v12 + hn3 * v13)) /
                     curvature;
    jacobian[3][2] = (sint * (u11 * u21 + u12 * u22) + omcost * (hu1 * u21 + hu2 * u22) +
                      tmsint * (hn1 * u21 + hn2 * u22) * (hn1 * u11 + hn2 * u12)) *
                     cosl0 / curvature;
    jacobian[3][3] = (u11 * u21 + u12 * u22);
    jacobian[3][4] = (v11 * u21 + v12 * u22);

    // zt
    jacobian[4][1] = (sint * (v11 * v21 + v12 * v22 + v13 * v23) + omcost * (hv1 * v21 + hv2 * v22 + hv3 * v23) +
                      tmsint * (hn1 * v21 + hn2 * v22 + hn3 * v23) * (hn1 * v11 + hn2 * v12 + hn3 * v13)) /
                     curvature;
    jacobian[4][2] = (sint * (u11 * v21 + u12 * v22) + omcost * (hu1 * v21 + hu2 * v22 + hu3 * v23) +
                      tmsint * (hn1 * v21 + hn2 * v22 + hn3 * v23) * (hn1 * u11 + hn2 * u12)) *
                     cosl0 / curvature;
    jacobian[4][3] = (u11 * v21 + u12 * v22);
    jacobian[4][4] = (v11 * v21 + v12 * v22 + v13 * v23);
  }

  constexpr void straightLineJacobian(Vector3 p1, double pathLength, double jacobian[5][5]) {
    for (int i = 0; i < 5; ++i)
      for (int j = 0; j < 5; ++j)
        jacobian[i][j] = i == j ? 1. : 0.;
    double cosl0 = perp(unit(p1));
    jacobian[3][2] = cosl0 * pathLength;
    jacobian[4][1] = pathLength;
  }

  // AnalyticalCurvilinearJacobian(globalParameters, x2, p2, pathLength) with the field fieldInTesla at the start x1
  constexpr void compute(Vector3 x1,
                         Vector3 p1,
                         int charge,
                         Vector3 x2,
                         Vector3 p2,
                         Vector3 fieldInTesla,
                         double pathLength,
                         double jacobian[5][5]) {
    const float transverseCurvature = -2.99792458e-3f * (float(charge) / perp(p1)) * fieldInTesla.z;
    if (pathLength * pathLength * std::fabs(transverseCurvature) > 1.e-5) {
      const Vector3 fieldInInverseGeV{
          2.99792458e-3f * fieldInTesla.x, 2.99792458e-3f * fieldInTesla.y, 2.99792458e-3f * fieldInTesla.z};
      const float qbp = float(charge) / std::sqrt(mag2(p1));
      fullJacobian(x1, p1, qbp, x2, p2, fieldInInverseGeV, pathLength, jacobian);
    } else {
      straightLineJacobian(p1, pathLength, jacobian);
    }
  }

}  // namespace curvilinearJacobian

#endif  // TrackingTools_AnalyticalJacobians_interface_curvilinearJacobian_h

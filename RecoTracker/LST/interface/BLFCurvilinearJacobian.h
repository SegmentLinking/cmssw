#ifndef RecoTracker_LST_interface_BLFCurvilinearJacobian_h
#define RecoTracker_LST_interface_BLFCurvilinearJacobian_h

#include <cmath>

#include <alpaka/alpaka.hpp>

namespace lst {

  // d(q/p, lambda, phi, x_T, y_T) / d(phi_e, x_T, kappa, cot, z_e), with cl = cos(lambda), sl = sin(lambda):
  //   q/p    : (0, 0, cl / B, -kappa cot cl^3 / B, 0)
  //   lambda : (0, 0, 0, cl^2, 0)
  //   phi    : (1, 0, 0, 0, kappa sl cl)
  //   x_T    : (0, 1, 0, 0, 0)
  //   y_T    : (0, 0, 0, 0, cl)
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr void blfToCurvilinearJacobian(double kappa,
                                                                              double cotTheta,
                                                                              float bField,
                                                                              double jacobian[5][5]) {
    const double cosLambda = 1. / std::sqrt(1. + cotTheta * cotTheta);
    const double sinLambda = cotTheta * cosLambda;
    const double b = bField;
    for (int row = 0; row < 5; ++row)
      for (int col = 0; col < 5; ++col)
        jacobian[row][col] = 0.;
    jacobian[0][2] = cosLambda / b;
    jacobian[0][3] = -kappa * cotTheta * cosLambda * cosLambda * cosLambda / b;
    jacobian[1][3] = cosLambda * cosLambda;
    jacobian[2][0] = 1.;
    jacobian[2][4] = kappa * sinLambda * cosLambda;
    jacobian[3][1] = 1.;
    jacobian[4][4] = cosLambda;
  }

  // The curvilinear covariance J C J^T, symmetric, from the stored covariance blocks: covCircle the upper triangle
  // (pp, px, xx, pk, xk, kk) of (phi_e, x_T, kappa), covLine the upper triangle (cc, cz, zz) of (cot, z_e).
  // C has no circle-line correlation, so J C J^T is written out over the non-zero elements of J; cov(lambda, x_T)
  // and cov(x_T, y_T) are zero.
  template <typename CircleCovariance, typename LineCovariance>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr void blfToCurvilinearCovariance(double kappa,
                                                                                double cotTheta,
                                                                                float bField,
                                                                                CircleCovariance const& covCircle,
                                                                                LineCovariance const& covLine,
                                                                                double covariance[5][5]) {
    double jacobian[5][5];
    blfToCurvilinearJacobian(kappa, cotTheta, bField, jacobian);
    const double qOverPKappa = jacobian[0][2], qOverPCot = jacobian[0][3], lambdaCot = jacobian[1][3];
    const double phiZ = jacobian[2][4], yZ = jacobian[4][4];
    const double phiPhi = covCircle[0], phiX = covCircle[1], xX = covCircle[2];
    const double phiKappa = covCircle[3], xKappa = covCircle[4], kappaKappa = covCircle[5];
    const double cotCot = covLine[0], cotZ = covLine[1], zZ = covLine[2];

    covariance[0][0] = qOverPKappa * qOverPKappa * kappaKappa + qOverPCot * qOverPCot * cotCot;
    covariance[0][1] = qOverPCot * lambdaCot * cotCot;
    covariance[0][2] = qOverPKappa * phiKappa + qOverPCot * phiZ * cotZ;
    covariance[0][3] = qOverPKappa * xKappa;
    covariance[0][4] = qOverPCot * yZ * cotZ;
    covariance[1][1] = lambdaCot * lambdaCot * cotCot;
    covariance[1][2] = lambdaCot * phiZ * cotZ;
    covariance[1][3] = 0.;
    covariance[1][4] = lambdaCot * yZ * cotZ;
    covariance[2][2] = phiPhi + phiZ * phiZ * zZ;
    covariance[2][3] = phiX;
    covariance[2][4] = phiZ * yZ * zZ;
    covariance[3][3] = xX;
    covariance[3][4] = 0.;
    covariance[4][4] = yZ * yZ * zZ;
    for (int row = 1; row < 5; ++row)
      for (int col = 0; col < row; ++col)
        covariance[row][col] = covariance[col][row];
  }

}  // namespace lst

#endif  // RecoTracker_LST_interface_BLFCurvilinearJacobian_h

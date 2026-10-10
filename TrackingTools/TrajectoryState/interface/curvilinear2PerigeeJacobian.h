#ifndef TrackingTools_TrajectoryState_interface_curvilinear2PerigeeJacobian_h
#define TrackingTools_TrajectoryState_interface_curvilinear2PerigeeJacobian_h

#include <cmath>

/*
 * PerigeeConversions::jacobianCurvilinear2Perigee as a function template, usable on the host and on GPUs:
 * PerigeeConversions calls it with GlobalVector and AlgebraicMatrix55, GPU code with the Vector3 and Matrix5 below.
 */
namespace curvilinear2PerigeeJacobian {

  // a float 3-vector with the GlobalVector operations the Jacobian uses
  class Vector3 {
  public:
    constexpr Vector3(float x, float y, float z) : components_{x, y, z} {}
    constexpr float x() const { return components_[0]; }
    constexpr float y() const { return components_[1]; }
    constexpr float z() const { return components_[2]; }
    constexpr float mag2() const { return x() * x() + y() * y() + z() * z(); }
    constexpr float mag() const { return std::sqrt(mag2()); }
    constexpr float perp() const { return std::sqrt(x() * x() + y() * y()); }
    constexpr float theta() const { return std::atan2(perp(), z()); }
    constexpr float dot(Vector3 const& other) const { return x() * other.x() + y() * other.y() + z() * other.z(); }
    constexpr Vector3 cross(Vector3 const& other) const {
      return Vector3(
          y() * other.z() - z() * other.y(), z() * other.x() - x() * other.z(), x() * other.y() - y() * other.x());
    }
    constexpr Vector3 unit() const {
      const float vectorMag2 = mag2();
      if (vectorMag2 == 0.f)
        return *this;
      const float scale = 1.f / std::sqrt(vectorMag2);
      return Vector3(x() * scale, y() * scale, z() * scale);
    }

  private:
    float components_[3];
  };

  struct Matrix5 {
    double element[5][5] = {};
    constexpr double& operator()(int row, int col) { return element[row][col]; }
  };

  // the host instantiation is inlined into PerigeeConversions as before, so that its machine code is unchanged
#if defined(__GNUC__) && !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)
#define CURVILINEAR2PERIGEE_INLINE [[gnu::always_inline]]
#else
#define CURVILINEAR2PERIGEE_INLINE
#endif

  // d(transverse curvature, theta, phi, epsilon, z) / d(q/p, lambda, phi, xT, yT) for the momentum, the field (1/GeV),
  // q/|p| and transverse curvature of a state; jacobian is zero on input, sinCos(x, sin, cos) is the caller's
  template <typename Vector, typename Matrix, typename SinCos>
  CURVILINEAR2PERIGEE_INLINE constexpr void compute(Vector const& momentum,
                                                    Vector const& field,
                                                    float signedInverseMomentum,
                                                    float transverseCurvature,
                                                    SinCos sinCos,
                                                    Matrix& jacobian) {
    const Vector zAxis(0., 0., 1.);
    const Vector tUnit = momentum.unit();
    const Vector uUnit = zAxis.cross(tUnit).unit();
    const Vector vUnit = tUnit.cross(uUnit);

    Vector iUnit(-momentum.x(), -momentum.y(), 0.);  // opposite to the track direction
    iUnit = iUnit.unit();
    const Vector jUnit(-iUnit.y(), iUnit.x(), 0.);  // counterclockwise rotation
    const Vector& kUnit(zAxis);
    const Vector hUnit = field.unit();
    const Vector hCrossT = hUnit.cross(tUnit);
    const Vector nUnit = hCrossT.unit();
    const double alpha = hCrossT.mag();
    const double qbp = signedInverseMomentum;
    const double curvature = -field.mag() * qbp;
    const double alphaQ = alpha * curvature;

    const double lambda = 0.5 * M_PI - momentum.theta();
    double sinLambda = 0, cosLambda = 0;
    sinCos(lambda, sinLambda, cosLambda);
    const double secLambda = 1. / cosLambda;

    const double iti = 1. / tUnit.dot(iUnit);
    const double nu = nUnit.dot(uUnit);
    const double nv = nUnit.dot(vUnit);
    const double ui = uUnit.dot(iUnit);
    const double vi = vUnit.dot(iUnit);
    const double uj = uUnit.dot(jUnit);
    const double vj = vUnit.dot(jUnit);
    const double uk = uUnit.dot(kUnit);
    const double vk = vUnit.dot(kUnit);

    if (std::fabs(transverseCurvature) < 1.e-10) {
      jacobian(0, 0) = secLambda;
      jacobian(0, 1) = sinLambda * secLambda * secLambda * std::abs(qbp);
    } else {
      const double bz = field.z();
      jacobian(0, 0) = -bz * secLambda;
      jacobian(0, 1) = -bz * sinLambda * secLambda * secLambda * qbp;
      jacobian(1, 3) = alphaQ * nv * ui * iti;
      jacobian(1, 4) = alphaQ * nv * vi * iti;
      jacobian(0, 3) = -jacobian(0, 1) * jacobian(1, 3);
      jacobian(0, 4) = -jacobian(0, 1) * jacobian(1, 4);
      jacobian(2, 3) = -alphaQ * secLambda * nu * ui * iti;
      jacobian(2, 4) = -alphaQ * secLambda * nu * vi * iti;
    }
    jacobian(1, 1) = -1.;
    jacobian(2, 2) = 1.;
    jacobian(3, 3) = vk * iti;
    jacobian(3, 4) = -uk * iti;
    jacobian(4, 3) = -vj * iti;
    jacobian(4, 4) = uj * iti;
  }

}  // namespace curvilinear2PerigeeJacobian

#endif  // TrackingTools_TrajectoryState_interface_curvilinear2PerigeeJacobian_h

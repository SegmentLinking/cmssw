#include "TrackingTools/PatternTools/interface/TwoTrackMinimumDistanceHelixLine.h"
#include "TrackingTools/TrajectoryParametrization/interface/GlobalTrajectoryParameters.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <iomanip>

using namespace std;

bool TwoTrackMinimumDistanceHelixLine::updateCoeffs() {
  bool isFirstALine = firstGTP->charge() == 0. || firstGTP->magneticField().inTesla(firstGTP->position()).z() == 0.;
  bool isSecondALine = secondGTP->charge() == 0. || secondGTP->magneticField().inTesla(secondGTP->position()).z() == 0.;
  if (isFirstALine && !isSecondALine) {
    theL = firstGTP;
    theH = secondGTP;
  } else if (!isFirstALine && isSecondALine) {
    theH = firstGTP;
    theL = secondGTP;
  } else {
    edm::LogWarning("TwoTrackMinimumDistanceHelixLine")
        << "Error in track charge: "
        << "One of the tracks has to be charged, and the other not." << endl
        << "Track Charges: " << firstGTP->charge() << " and " << secondGTP->charge();
    return true;
  }

  Hn = theH->momentum().mag();
  Ln = theL->momentum().mag();

  if (Hn == 0. || Ln == 0.) {
    edm::LogWarning("TwoTrackMinimumDistanceHelixLine") << "Momentum of input trajectory is zero.";
    return true;
  };

  GlobalPoint lOrig = theL->position();
  GlobalPoint hOrig = theH->position();
  posDiff = GlobalVector((lOrig - hOrig).basicVector());
  theLp = theL->momentum();

  const double Bc2kH = theH->magneticField().inTesla(hOrig).z() * 2.99792458e-3;
  //   MagneticField::inInverseGeV ( hOrig ).z();

  if (Bc2kH == 0.) {
    edm::LogWarning("TwoTrackMinimumDistanceHelixLine") << "Magnetic field at point " << hOrig << " is zero.";
    return true;
  };

  theCoeffs = helixLineClosestApproach::coefficients(posDiff.x(),
                                                     posDiff.y(),
                                                     posDiff.z(),
                                                     theLp.x(),
                                                     theLp.y(),
                                                     theLp.z(),
                                                     Hn,
                                                     theH->momentum().z(),
                                                     theH->charge(),
                                                     Bc2kH,
                                                     theH->momentum().phi());
  return false;
}

bool TwoTrackMinimumDistanceHelixLine::oneIteration(double& thePhiH, double& fct, double& derivative) const {
  helixLineClosestApproach::functionAndDerivative(theCoeffs, thePhiH, fct, derivative);
  return false;
}

bool TwoTrackMinimumDistanceHelixLine::calculate(const GlobalTrajectoryParameters& theFirstGTP,
                                                 const GlobalTrajectoryParameters& theSecondGTP,
                                                 const float qual) {
  pointsUpdated = false;
  firstGTP = &theFirstGTP;
  secondGTP = &theSecondGTP;

  if (updateCoeffs()) {
    finalPoints();
    return true;
  };

  double fctVal, derVal, dPhiH;
  thePhiH = theCoeffs.phiH0;

  double x1 = theCoeffs.phiH0 - M_PI, x2 = theCoeffs.phiH0 + M_PI;
  for (int j = 1; j <= themaxiter; ++j) {
    oneIteration(thePhiH, fctVal, derVal);
    dPhiH = fctVal / derVal;
    thePhiH -= dPhiH;
    if ((x1 - thePhiH) * (thePhiH - x2) < 0.0) {
      LogDebug("TwoTrackMinimumDistanceHelixLine") << "Jumped out of brackets in root finding. Will be moved closer.";
      thePhiH += (dPhiH * 0.8);
    }
    if (fabs(dPhiH) < qual) {
      finalPoints();
      return false;
    }
  }
  LogDebug("TwoTrackMinimumDistanceHelixLine") << "Number of steps exceeded. Has not converged.";
  finalPoints();
  return true;
}

double TwoTrackMinimumDistanceHelixLine::firstAngle() const {
  if (firstGTP == theL)
    return theL->momentum().phi();
  else
    return thePhiH;
}

double TwoTrackMinimumDistanceHelixLine::secondAngle() const {
  if (secondGTP == theL)
    return theL->momentum().phi();
  else
    return thePhiH;
}

pair<GlobalPoint, GlobalPoint> TwoTrackMinimumDistanceHelixLine::points() const {
  if (firstGTP == theL)
    return pair<GlobalPoint, GlobalPoint>(linePoint, helixPoint);
  else
    return pair<GlobalPoint, GlobalPoint>(helixPoint, linePoint);
}

pair<double, double> TwoTrackMinimumDistanceHelixLine::pathLength() const {
  if (firstGTP == theL)
    return pair<double, double>(linePath, helixPath);
  else
    return pair<double, double>(helixPath, linePath);
}

void TwoTrackMinimumDistanceHelixLine::finalPoints() {
  if (pointsUpdated)
    return;
  double dx, dy, dz;
  helixLineClosestApproach::helixOffset(theCoeffs, thePhiH, dx, dy, dz);
  helixPoint = GlobalPoint(theH->position().x() + dx, theH->position().y() + dy, theH->position().z() + dz);
  helixPath = helixLineClosestApproach::helixPathLength(theCoeffs, thePhiH);

  GlobalVector diff((theL->position() - helixPoint).basicVector());
  tL = (-diff.dot(theLp)) / (Ln * Ln);
  linePoint = GlobalPoint(theL->position().x() + tL * theCoeffs.px,
                          theL->position().y() + tL * theCoeffs.py,
                          theL->position().z() + tL * theCoeffs.pz);
  linePath = tL * theLp.mag();
  pointsUpdated = true;
}

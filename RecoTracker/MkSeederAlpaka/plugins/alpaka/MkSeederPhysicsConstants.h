#ifndef RecoTracker_MkSeederAlpaka_plugins_alpaka_MkSeederPhysicsConstants_h
#define RecoTracker_MkSeederAlpaka_plugins_alpaka_MkSeederPhysicsConstants_h

// The seeder's physical constants of the cuts, defined once. The field of the seed fits and of the helix to pT
// conversions of the fits is mkfit::Config::Bfield, passed with the other propagation constants in FitParameters::bField
// (MkSeederProducer.cc); the cuts' pT <-> radius conversion below uses the nominal 3.8 T (as mkFit's seeder cuts
// are defined), and the classifier features use the field at the origin from the EventSetup (ClassifierFeatureInputs::bz).

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkseeder {

  // the speed of light in the units of the helix conversions: pT [GeV] = kSolOver100 * B [T] * R [cm]
  constexpr float kSolOver100 = 0.299792458e-2f;
  // pT [GeV] = kNominalPtPerRadius * R [cm]: 0.003 * 3.8 T, the radius of a cut's pT in the chain finder and the
  // post step
  constexpr float kNominalPtPerRadius = 0.0114f;
  constexpr double kNominalPtPerRadiusD = 0.003 * 3.8;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkseeder

#endif

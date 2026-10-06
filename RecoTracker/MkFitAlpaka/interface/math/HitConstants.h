#ifndef RecoTracker_MkFitAlpaka_interface_math_HitConstants_h
#define RecoTracker_MkFitAlpaka_interface_math_HitConstants_h

// MkFitCore special hit indices (RecoTracker/MkFitCore/interface/Hit.h, HitOnTrack). One copy for the whole package.

namespace mkfitdev {
  constexpr int kHitMissIdx = -1;        // hit is missed
  constexpr int kHitStopIdx = -2;        // track is stopped
  constexpr int kHitEdgeIdx = -3;        // track not in sensitive region of detector
  constexpr int kHitMaxClusterIdx = -5;  // hit cluster size > maxClusterSize
  constexpr int kHitInGapIdx = -7;       // track passing through inactive module
  constexpr int kHitCCCFilterIdx = -9;   // hit filtered via CCC (counted as found, as Track::addHitIdx)
}  // namespace mkfitdev

#endif

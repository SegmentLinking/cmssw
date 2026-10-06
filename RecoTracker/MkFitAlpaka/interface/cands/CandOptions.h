#ifndef RecoTracker_MkFitAlpaka_interface_cands_CandOptions_h
#define RecoTracker_MkFitAlpaka_interface_cands_CandOptions_h

// Option (IdxChi2List) construction of MkFinder::findCandidatesCloneEngine (MkFinder.cc:1772-1850), the
// bookkeeping half of the chi2 kernel. The caller supplies the candidate's
// bookkeeping (as input to MkFinder: m_NFoundHits, m_NOverlapHits, m_NInsideMinusOneHits, m_NTailMinusOneHits,
// m_Chi2) and pt = |1/par_iP[3]| of the layer-propagated state.

#include <cstdint>

#include "RecoTracker/MkFitAlpaka/interface/cands/CandTypes.h"

namespace mkfitdev {

  // WithinSensitiveRegion_e (TrackerInfo.h:17)
  enum WsrResult : int8_t { kWsrUndef = -1, kWsrInside = 0, kWsrEdge = 1, kWsrOutside = 2, kWsrFailed = 3 };

  // MkFitCore: std::abs(1.0f / m_Par[iP].At(itrack, 3, 0))
  ALPAKA_FN_HOST_ACC inline float optionPt(float invpt) {
    const float v = 1.0f / invpt;
    return v < 0.f ? -v : v;
  }

  // Option for an accepted hit (MkFinder.cc:1772-1782); hitChi2 = |outChi2| of the hit.
  ALPAKA_FN_HOST_ACC inline CandOption makeHitOption(
      const CandBook& c, int trkIdx, float pt, int hitIdx, uint32_t module, float hitChi2) {
    CandOption o;
    o.trkIdx = trkIdx;
    o.hitIdx = hitIdx;
    o.module = module;
    o.nhits = c.nFound + 1;
    o.ntailholes = 0;
    o.noverlaps = c.nOverlap;
    o.nholes = c.nInsideMinusOne + c.nTailMinusOne;  // num_all_minus_one_hits
    o.pt = pt;
    o.chi2 = c.chi2 + hitChi2;
    o.chi2_hit = hitChi2;
    o.score = getScoreStruct(o);
    return o;
  }

  // Hit code of the "no hit" option (MkFinder.cc:1806-1822). Not called for WSR_Outside candidates (held back).
  ALPAKA_FN_HOST_ACC inline int invalidHitCode(const CandBook& c,
                                               int maxHolesPerCand,
                                               int maxConsecHoles,
                                               int wsr,
                                               bool inGap,
                                               int nHitsAdded,
                                               bool tooLargeCluster) {
    int code = ((c.nInsideMinusOne + c.nTailMinusOne) < maxHolesPerCand && c.nTailMinusOne < maxConsecHoles)
                   ? kHitMissIdx
                   : kHitStopIdx;
    if (wsr == kWsrEdge) {
      code = kHitEdgeIdx;
    } else if (inGap && nHitsAdded == 0) {
      code = kHitInGapIdx;
    } else if (tooLargeCluster && nHitsAdded == 0) {
      code = kHitMaxClusterIdx;
    }
    return code;
  }

  // The "no hit" option itself (MkFinder.cc:1830-1842).
  ALPAKA_FN_HOST_ACC inline CandOption makeInvalidOption(const CandBook& c, int trkIdx, float pt, int code) {
    CandOption o;
    o.trkIdx = trkIdx;
    o.hitIdx = code;
    o.module = static_cast<uint32_t>(-1);
    o.nhits = c.nFound;
    o.ntailholes = (code == kHitMissIdx ? c.nTailMinusOne + 1 : c.nTailMinusOne);
    o.noverlaps = c.nOverlap;
    o.nholes = c.nInsideMinusOne;  // num_inside_minus_one_hits
    o.pt = pt;
    o.chi2 = c.chi2;
    o.chi2_hit = 0;
    o.score = getScoreStruct(o);
    return o;
  }

}  // namespace mkfitdev

#endif

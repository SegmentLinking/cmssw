#ifndef RecoTracker_MkFitAlpaka_interface_cands_CandOptions_h
#define RecoTracker_MkFitAlpaka_interface_cands_CandOptions_h

// Option (IdxChi2List) construction of MkFinder::findCandidatesCloneEngine (MkFinder.cc:1772-1850), the
// bookkeeping half of the chi2 kernel. The caller supplies the candidate's bookkeeping (as input to MkFinder:
// m_NFoundHits, m_NOverlapHits, m_NInsideMinusOneHits, m_NTailMinusOneHits, m_Chi2) and pt = |1/par_iP[3]| of the
// layer-propagated state.

#include <cstdint>

#include "RecoTracker/MkFitAlpaka/interface/cands/CandTypes.h"
#include "RecoTracker/MkFitAlpaka/interface/es/ESView.h"

namespace mkfitdev {

  // MkFitCore: std::abs(1.0f / m_Par[iP].At(itrack, 3, 0))
  ALPAKA_FN_HOST_ACC inline float optionPt(float invpt) {
    const float pt = 1.0f / invpt;
    return pt < 0.f ? -pt : pt;
  }

  // Option for an accepted hit (MkFinder.cc:1772-1782); hitChi2 = |outChi2| of the hit.
  ALPAKA_FN_HOST_ACC inline CandOption makeHitOption(
      const CandBook& book, int trkIdx, float pt, int hitIdx, uint32_t module, float hitChi2) {
    CandOption option;
    option.trkIdx = trkIdx;
    option.hitIdx = hitIdx;
    option.module = module;
    option.nhits = book.nFound + 1;
    option.ntailholes = 0;
    option.noverlaps = book.nOverlap;
    option.nholes = book.nInsideMinusOne + book.nTailMinusOne;  // num_all_minus_one_hits
    option.pt = pt;
    option.chi2 = book.chi2 + hitChi2;
    option.chi2_hit = hitChi2;
    option.score = getScoreStruct(option);
    return option;
  }

  // Hit code of the "no hit" option (MkFinder.cc:1806-1822). Not called for WSR_Outside candidates (held back).
  ALPAKA_FN_HOST_ACC inline int invalidHitCode(const CandBook& book,
                                               int maxHolesPerCand,
                                               int maxConsecHoles,
                                               int wsr,
                                               bool inGap,
                                               int nHitsAdded,
                                               bool tooLargeCluster) {
    int code = ((book.nInsideMinusOne + book.nTailMinusOne) < maxHolesPerCand && book.nTailMinusOne < maxConsecHoles)
                   ? kHitMissIdx
                   : kHitStopIdx;
    if (wsr == WSR_Edge) {
      code = kHitEdgeIdx;
    } else if (inGap && nHitsAdded == 0) {
      code = kHitInGapIdx;
    } else if (tooLargeCluster && nHitsAdded == 0) {
      code = kHitMaxClusterIdx;
    }
    return code;
  }

  // The "no hit" option itself (MkFinder.cc:1830-1842).
  ALPAKA_FN_HOST_ACC inline CandOption makeInvalidOption(const CandBook& book, int trkIdx, float pt, int code) {
    CandOption option;
    option.trkIdx = trkIdx;
    option.hitIdx = code;
    option.module = static_cast<uint32_t>(-1);
    option.nhits = book.nFound;
    option.ntailholes = (code == kHitMissIdx ? book.nTailMinusOne + 1 : book.nTailMinusOne);
    option.noverlaps = book.nOverlap;
    option.nholes = book.nInsideMinusOne;  // num_inside_minus_one_hits
    option.pt = pt;
    option.chi2 = book.chi2;
    option.chi2_hit = 0;
    option.score = getScoreStruct(option);
    return option;
  }

}  // namespace mkfitdev

#endif

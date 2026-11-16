#ifndef RecoTracker_MkSeederAlpaka_interface_MkSeederChainSoA_h
#define RecoTracker_MkSeederAlpaka_interface_MkSeederChainSoA_h

#include <cstdint>

#include "DataFormats/SoATemplate/interface/SoABlocks.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"

namespace mkseeder {

  // The chain of one z side: the mkFit layers in the order a straight line from the beam region crosses them (the
  // chain positions), the pairs of positions a doublet starts on, and the search windows per layer combination.

  // One row per chain position: its layer and the envelope of the layer's crossing by a line z = z0 + cot * r (a
  // barrel layer: the line's z over the layer's radial range [positionLow, positionHigh]; a disc: its r over the z
  // range). The line crosses the layer surely if the interval is within [innerLow, innerHigh], maybe if it overlaps
  // [outerLow, outerHigh].
  GENERATE_SOA_LAYOUT(MkSeederChainPositionsLayout,
                      SOA_COLUMN(int32_t, layerId),
                      // layerId if the seeder bins the layer's hits, else -1 (the position is skipped)
                      SOA_COLUMN(int32_t, binnedLayerId),
                      SOA_COLUMN(bool, envelopeValid),
                      SOA_COLUMN(bool, isDisc),
                      SOA_COLUMN(bool, isPixel),
                      // an outer-tracker layer after OT1-P: only for lines that cross few pixel layers
                      SOA_COLUMN(bool, isExtraOuterTracker),
                      SOA_COLUMN(float, position),
                      SOA_COLUMN(float, positionLow),
                      SOA_COLUMN(float, positionHigh),
                      SOA_COLUMN(float, innerLow),
                      SOA_COLUMN(float, innerHigh),
                      SOA_COLUMN(float, outerLow),
                      SOA_COLUMN(float, outerHigh))

  // One row per start pair: the doublet's positions, whether a crossed pixel layer may be skipped between them (and
  // the one barrel layer with a gap map it skips, or -1), the usable cot range (cotMin > cotMax: none), and the
  // positions at which a crossed layer counts as a hole (rows firstHole .. firstHole + nHoles of the holes)
  GENERATE_SOA_LAYOUT(MkSeederChainStartsLayout,
                      SOA_COLUMN(int32_t, positionA),
                      SOA_COLUMN(int32_t, positionB),
                      SOA_COLUMN(bool, gapAllowed),
                      SOA_COLUMN(int32_t, gapLayer),
                      SOA_COLUMN(float, cotMin),
                      SOA_COLUMN(float, cotMax),
                      SOA_COLUMN(int32_t, firstHole),
                      SOA_COLUMN(int32_t, nHoles))

  // One row per hole position of a start pair: the chain position, and whether it is between the start's two hits
  GENERATE_SOA_LAYOUT(MkSeederChainHolesLayout, SOA_COLUMN(int32_t, position), SOA_COLUMN(bool, isBetweenStartHits))

  // One row per window table (a layer combination of a pattern): the stage c windows (C) and the stage d windows
  // a + b / pT (A, B), the reference path length of the b terms (0: none), and its |eta| slices in the eta windows
  GENERATE_SOA_LAYOUT(MkSeederWindowTablesLayout,
                      SOA_COLUMN(float, phiWindowC),
                      SOA_COLUMN(float, qWindowC),
                      SOA_COLUMN(float, phiWindowA),
                      SOA_COLUMN(float, phiWindowB),
                      SOA_COLUMN(float, qWindowA),
                      SOA_COLUMN(float, qWindowB),
                      SOA_COLUMN(float, referencePathLength),
                      SOA_COLUMN(int32_t, firstEtaWindow),
                      SOA_COLUMN(int32_t, nEtaWindows))

  // One row per |eta| slice of a window table: its stage d windows replace the table's for absEtaLow <= |eta| <
  // absEtaHigh
  GENERATE_SOA_LAYOUT(MkSeederEtaWindowsLayout,
                      SOA_COLUMN(float, absEtaLow),
                      SOA_COLUMN(float, absEtaHigh),
                      SOA_COLUMN(float, phiWindowA),
                      SOA_COLUMN(float, phiWindowB),
                      SOA_COLUMN(float, qWindowA),
                      SOA_COLUMN(float, qWindowB))

  // The window table of a combination of chain positions (-1: none, kPrunedCombination: not searched): row
  // (a * n + b) * n + c for the triplets a, b, c (n chain positions), ((a * n + b) * n + c) * n + d for the quads
  GENERATE_SOA_LAYOUT(MkSeederComboIndexLayout, SOA_COLUMN(int32_t, windowTable))

  GENERATE_SOA_BLOCKS(MkSeederChainLayout,
                      SOA_BLOCK(positions, MkSeederChainPositionsLayout),
                      SOA_BLOCK(starts, MkSeederChainStartsLayout),
                      SOA_BLOCK(holes, MkSeederChainHolesLayout),
                      SOA_BLOCK(windowTables, MkSeederWindowTablesLayout),
                      SOA_BLOCK(etaWindows, MkSeederEtaWindowsLayout),
                      SOA_BLOCK(windowTableByTriplet, MkSeederComboIndexLayout),
                      SOA_BLOCK(windowTableByQuad, MkSeederComboIndexLayout))

  using MkSeederChainPositionsSoA = MkSeederChainPositionsLayout<>;
  using MkSeederChainStartsSoA = MkSeederChainStartsLayout<>;
  using MkSeederChainHolesSoA = MkSeederChainHolesLayout<>;
  using MkSeederWindowTablesSoA = MkSeederWindowTablesLayout<>;
  using MkSeederEtaWindowsSoA = MkSeederEtaWindowsLayout<>;
  using MkSeederComboIndexSoA = MkSeederComboIndexLayout<>;
  using MkSeederChainSoA = MkSeederChainLayout<>;

  // The constants of one side's chain finder (from the configuration; host memory, passed to the kernels by value)
  struct MkSeederChainParameters {
    int nPositions;
    int side;  // +1: the +z side, -1: the -z side
    int nStartPairs;
    // holes: the crossed pixel layers without a hit that a candidate may carry (maxHoles), into an outer-tracker
    // layer (maxHolesForOuterTracker), and in its start doublet (maxStartHoles)
    int maxHoles, maxHolesForOuterTracker, maxStartHoles;
    int leadOnly;               // a start's holes may only lead (the start's a and b crossed consecutively)
    int innerOuterTrackerOnly;  // a candidate with a hole after its start may continue only on OT1-P
    int knownCombinationsOnly;  // search a layer only for a combination with a window table
    int alwaysForwardMisses;    // forward a candidate as a miss also when it found hits
    int useStageCNet;           // trip_net_c set: the stage-c network (MkSeederStageCNet.h)
    float stageCNetCut[5];      // its logit cuts per |eta| bin
    float missGapDz;            // > 0: an L1-L2 doublet's L3 miss goes on only near a L3 z gap (skipMissOutsideGap)
    int useGapMap;              // a start with a gap: the skipped layer's ladder gaps (MkSeederGapMapSoA)
    int useFakeScore;           // the fake score (or the shape cut) is applied
    int useShapeCut;            // the cluster-length band of the barrel pixel layers (MkSeederShapeSoA)
    int emitTriplets;           // a triplet that finds no fourth hit is emitted as a seed
    int allowAnyOuterTracker;   // doublets and misses may also route to the outer-tracker positions
    int maxPixelLayersForExtraOuterTracker;  // > 0: the extra outer-tracker positions only for such lines
    // the emit's gates: a seed needs minPixelHits pixel hits, a triplet a triplet-MLP score >= tripletMlpCut (> 0);
    // a triplet below with a score >= tripletMlpRescueMinScore is kept for the outer-tracker probe when
    // tripletMlpRescueOuterTrackerHits > 0
    int minPixelHits;
    // > 0: the pixel hits of a quad whose third hit is on an outer-tracker position (min_pix_ot); a doublet then goes
    // on to an outer-tracker position as a miss only if it found no third hit at its current position
    int minPixelHitsOuterTrackerThird;
    float tripletMlpCut;
    int tripletMlpRescueOuterTrackerHits;
    float tripletMlpRescueMinScore;
    // the fake score: a candidate's sum of squared normalised residuals must stay below fakeScoreCut
    // (fakeScoreCutForward for |cot| >= forwardMinAbsCot, if > 0)
    float fakeScoreCut, fakeScoreCutForward, forwardMinAbsCot;
    float tripletMinAbsCot, tripletMaxAbsCot;  // the |cot| range of the triplet seeds
    float d0Max, ptMin;
    float d0MaxOuterTracker;  // the d0 of stage c's windows on an outer-tracker position (d0_max_ot, else d0Max)
    // stage b's doublet cut in float: |dphi| <= dr * doubletInverseTwoRadius + doubletD0Max * (1 / ra - 1 / rb) +
    // doubletPhiMargin
    float doubletInverseTwoRadius, doubletD0Max, doubletPhiMargin;
    // stage b's window on the second layer (double arithmetic): pT, d0 and margin, and the z0 range half width
    float startPtMin, startD0Max, zVertexHalfWidth, doubletMargin;
    // zv_outer (> 0): the z0 range half width for the start doublets with the inner hit at r >= zVertexOuterMinRadius
    float zVertexHalfWidthOuter, zVertexOuterMinRadius;
    float zVertexOuterMinAbsCot;  // > 0: zv_outer only for the doublets with |cot| >= this (zv_outer_cotmin)
    int zVertexOuterDiscOnly;     // zv_outer only for the start pairs whose inner layer is a disk (zv_outer_disk)
    // the outer-tracker confirmations: a quad ending on OT1-P needs a hit on OT2-P (ot2LayerId, -1: off), a quad of
    // four pixel hits one on OT1-P (ot1LayerId, -1: off), within the windows below times their scale
    int ot2LayerId, ot1LayerId;
    float ot2WindowScale, ot2PhiWindowA, ot2PhiWindowB, ot2QWindowA, ot2QWindowB, ot2PhiWindowMin;
    float ot2ConfirmationSigma, ot2ConfirmationGate;
    // > 0: the OT2-P confirmation of the pixel triplets' quads with the local windows a + b / pT (x, y) times it
    float ot2LocalWindowScale, ot2LocalWindowX[2], ot2LocalWindowY[2];
    float ot1WindowScale, ot1MaxAbsCot, ot1ConfirmationGate;
    // > 0: fk_ot1 / fk_ot2 by the local chi2 (fk_ot1_lsig, fk_ot2_lsig, fk_ot_lsx, fk_ot1_lsy, fk_ot2_lsy, fk_ot2_lgate)
    float ot1LocalSigma, ot2LocalSigma, otLocalSigmaX[3], ot1LocalSigmaY, ot2LocalSigmaY, ot2LocalGateSigma;
    // > 1: stage d's windows times this for a triplet with a strong merged-cluster candidate hit
    // (isStrongMergedCandidate of MkSeederHitsSoA::mergedTag)
    float mergedWindowScale;
    // 1: a hit with the charge bit of MkSeederHitsSoA::mergedTag passes the cluster-length band's upper limit
    int mergedShapeExempt;
  };

  // the window-table index (MkSeederComboIndexLayout) of a combination that the finder does not search (prune)
  constexpr int32_t kPrunedCombination = -2;

  // the finder searches a combination with this window-table index: it has a table, or it has none and every
  // combination is searched (any_combination); a pruned one never
  constexpr bool searchesCombination(const MkSeederChainParameters &chain, int32_t windowTable) {
    return windowTable >= 0 || (windowTable == -1 && !chain.knownCombinationsOnly);
  }

}  // namespace mkseeder

#endif

#ifndef RecoTracker_LSTCore_interface_SeedStateSoA_h
#define RecoTracker_LSTCore_interface_SeedStateSoA_h

#include <cstdint>

#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/SoATemplate/interface/SoABlocks.h"

#include "RecoTracker/LSTCore/interface/Common.h"

namespace lst {

  // Outcome of the Broken Line Fit and of its checks for one seed state.
  enum class SeedStateStatus : uint8_t {
    kOk = 0,
    // No fit was attempted or possible. Pixel track: no reco::Track, hit count out of range, or IT-only state not
    // computed. Track candidate: a pLS, or fewer hits than the smallest fit size after the hit selection.
    kNotFitted,
    // A parameter or a covariance element is not finite.
    kNonFinite,
    // sign(kappa) differs from the charge (kappa = 0 counts as a mismatch).
    kChargeMismatch,
    // A diagonal variance of the native covariance blocks is not positive.
    kNonPositiveVariance,
  };

  // The stateHit encoding: bit 31 clear = an OT hit, the rest is its LSTOTHits index;
  // bit 31 set = position k in the row's pixel track, the rest is k.
  // A row with no hit to sit on has lst::kInvalidU32Idx, which has bit 31 set: check it before decoding.
  constexpr uint32_t kStateHitPixelBit = 0x80000000u;

  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr uint32_t encodeStateHitOT(uint32_t otHitIndex) { return otHitIndex; }

  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr uint32_t encodeStateHitPixel(uint32_t positionInTrack) {
    return kStateHitPixelBit | positionInTrack;
  }

  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr bool stateHitIsPixel(uint32_t stateHit) {
    return (stateHit & kStateHitPixelBit) != 0;
  }

  // The LSTOTHits index (OT hit) or the position in the pixel track (pixel hit) of an encoded stateHit.
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr uint32_t stateHitIndex(uint32_t stateHit) {
    return stateHit & ~kStateHitPixelBit;
  }

  // Broken Line Fit state at the outermost fitted node, for seeding. One row per object.
  // An unfit row has nFit = 0 and a NaN kappa, so that a consumer testing only std::isfinite also rejects it.
  // The transverse coordinate x_T is zero by construction at the state point and is not stored;
  // the covariance is block-diagonal (a circle block and a line block).
  GENERATE_SOA_LAYOUT(SeedStateSoALayout,
                      SOA_COLUMN(LSTObjType, type),
                      SOA_COLUMN(uint8_t, status),
                      // Corrected outermost fitted node, global CMS frame, cm:
                      // (x, y) is the point of the circle fit, z the fitted z of the line fit at the same node.
                      SOA_COLUMN(float, x),
                      SOA_COLUMN(float, y),
                      SOA_COLUMN(float, z),
                      SOA_COLUMN(float, phi),
                      SOA_COLUMN(float, cotTheta),
                      // Signed curvature, cm^-1; its sign is the charge (CMSSW transverseCurvature = -kappa).
                      SOA_COLUMN(float, kappa),
                      // Circle covariance, upper triangle of (phi, x_T, kappa), x_T being transverse to the
                      // momentum: (phi-phi, phi-xT, xT-xT, phi-kappa, xT-kappa, kappa-kappa).
                      SOA_COLUMN(lst::ArrayFx6, covCircle),
                      // Line covariance, upper triangle of (cotTheta, z): (cot-cot, cot-z, z-z).
                      SOA_COLUMN(lst::ArrayFx3, covLine),
                      SOA_COLUMN(int8_t, charge),
                      // Number of hits the fit actually used, 0 if unfit.
                      SOA_COLUMN(uint8_t, nFit),
                      // 1 when no earlier node was more than 2 cm away transversely from the last hit and the
                      // first fitted node was used as the reference node, 0 otherwise.
                      SOA_COLUMN(uint8_t, refFallback),
                      // Seed index of the pixel track: for a pixel-track row its own row index, for a
                      // track-candidate row the seed index of its pixel seed, lst::kInvalidU32Idx for T5 and T4.
                      SOA_COLUMN(uint32_t, pixelSeedIndex),
                      SOA_COLUMN(uint32_t, stateHit),
                      SOA_SCALAR(float, B))

  // Per pixel track quantities and per-event counters of the pixel-track product.
  GENERATE_SOA_LAYOUT(PixelSeedStateExtraSoALayout,
                      SOA_COLUMN(uint8_t, nITHits),
                      SOA_SCALAR(uint32_t, nTracks),
                      SOA_SCALAR(uint32_t, nNotFitted),
                      SOA_SCALAR(uint32_t, nNonFinite),
                      SOA_SCALAR(uint32_t, nChargeMismatch),
                      SOA_SCALAR(uint32_t, nNonPositiveVariance),
                      // Tracks meeting the condition for a rebuild without OT hits.
                      SOA_SCALAR(uint32_t, nITCandidates),
                      // Of those, tracks with fewer than 3 IT hits.
                      SOA_SCALAR(uint32_t, nITTooFewHits),
                      // IT-only fit rejections.
                      SOA_SCALAR(uint32_t, nITNonFinite),
                      SOA_SCALAR(uint32_t, nITChargeMismatch),
                      SOA_SCALAR(uint32_t, nITNonPositiveVariance),
                      // Closest-approach failures, which keep the pLS: iteration not converged, or non-finite result.
                      SOA_SCALAR(uint32_t, nPCANotConverged),
                      SOA_SCALAR(uint32_t, nPCANonFinite))

  // Product of the pixel tracks: one row per fitted pixel track in each block, the row index being the seed index.
  // The IT-only state has the columns of the full state; its status is kNotFitted when it was not computed.
  GENERATE_SOA_BLOCKS(PixelSeedStatesSoALayout,
                      SOA_BLOCK(state, SeedStateSoALayout),
                      SOA_BLOCK(itState, SeedStateSoALayout),
                      SOA_BLOCK(extra, PixelSeedStateExtraSoALayout))

  // Per-event counters of the track-candidate product. They partition the rows: every row is counted in exactly
  // one of nNotFittedPLS, nTooFewHits, nNonFinite, nChargeMismatch, nNonPositiveVariance, or has status kOk,
  // so the number of rows with status kOk is nTCs minus the sum of the other five counters.
  GENERATE_SOA_LAYOUT(TCSeedStateExtraSoALayout,
                      // Rows of the product, one per track candidate.
                      SOA_SCALAR(uint32_t, nTCs),
                      // pLS track candidates: no fit, the seed state is the pixel-track one.
                      // Rows counted here and in nTooFewHits both have status kNotFitted; the counters tell them apart.
                      SOA_SCALAR(uint32_t, nNotFittedPLS),
                      // Fewer hits than the smallest fit size after hit selection.
                      SOA_SCALAR(uint32_t, nTooFewHits),
                      // Fit rejections, with status kNonFinite, kChargeMismatch and kNonPositiveVariance respectively.
                      SOA_SCALAR(uint32_t, nNonFinite),
                      SOA_SCALAR(uint32_t, nChargeMismatch),
                      SOA_SCALAR(uint32_t, nNonPositiveVariance))

  // Product of the track candidates (T5, T4, pT3, pT5, pLS): one row per track candidate, in track-candidate order.
  GENERATE_SOA_BLOCKS(TCSeedStatesSoALayout,
                      SOA_BLOCK(state, SeedStateSoALayout),
                      SOA_BLOCK(extra, TCSeedStateExtraSoALayout))

  using SeedStateSoA = SeedStateSoALayout<>;
  using PixelSeedStateExtraSoA = PixelSeedStateExtraSoALayout<>;
  using PixelSeedStatesSoA = PixelSeedStatesSoALayout<>;
  using TCSeedStateExtraSoA = TCSeedStateExtraSoALayout<>;
  using TCSeedStatesSoA = TCSeedStatesSoALayout<>;

  using SeedState = SeedStateSoA::View;
  using SeedStateConst = SeedStateSoA::ConstView;
  using PixelSeedStateExtra = PixelSeedStateExtraSoA::View;
  using PixelSeedStateExtraConst = PixelSeedStateExtraSoA::ConstView;
  using PixelSeedStatesView = PixelSeedStatesSoA::View;
  using PixelSeedStatesConstView = PixelSeedStatesSoA::ConstView;
  using TCSeedStateExtra = TCSeedStateExtraSoA::View;
  using TCSeedStateExtraConst = TCSeedStateExtraSoA::ConstView;
  using TCSeedStatesView = TCSeedStatesSoA::View;
  using TCSeedStatesConstView = TCSeedStatesSoA::ConstView;

}  // namespace lst

#endif

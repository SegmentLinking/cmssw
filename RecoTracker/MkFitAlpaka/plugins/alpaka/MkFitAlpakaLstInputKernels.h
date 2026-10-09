#ifndef RecoTracker_MkFitAlpaka_plugins_alpaka_MkFitAlpakaLstInputKernels_h
#define RecoTracker_MkFitAlpaka_plugins_alpaka_MkFitAlpakaLstInputKernels_h

// LST's input device collection built on the device.
//   OT hits  : the device OT rechit SoA (global x/y/z, DetId, cluster size; row = cluster key).
//   pLS      : Patatrack's device pixel tracks (state at the beam-spot PCA + covariance), option (i): the LST pLS
//              fields that hltInputLST computes from the host KF-refit seed (hltInitialStepSeeds) are computed from
//              the Patatrack fit instead (uniform-field helix from the PCA to the outermost hit).
#include <cstdint>

#include <Eigen/Core>  // before any SoA header (Eigen columns of TracksSoA)

#include "DataFormats/SoATemplate/interface/SoAConstMultiView.h"
#include "DataFormats/TrackSoA/interface/TracksSoA.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTracker/LSTCore/interface/LSTInputSoA.h"
#include "RecoTracker/MkFitAlpaka/interface/hits/DeviceHitInput.h"
#include "RecoTracker/MkFitAlpaka/interface/othits/OTRecHitSoA.h"
#include "RecoTracker/MkFitAlpaka/interface/hits/HitSoA.h"
#include "RecoTracker/MkFitAlpaka/interface/seeds/alpaka/LstSeedFit.h"
#include "RecoTracker/MkFitAlpaka/interface/tracks/TrackSoA.h"

namespace mkfitdev::lstin {

  constexpr uint32_t kNoKey = 0xffffffffu;
  constexpr int kMaxTrackHits = 255;  // nHits is uint8 in the LST pLS; longer tracks are counted and dropped

  // the hits the pixel-track hit indices refer to, as the CA reads them: the pixel rechits, then the OT rechits
  using PixelTrackHits = SoAConstMultiView<::reco::TrackingRecHitConstView, 2>;

  struct Params {
    float ptCut;
    float bsx, bsy, bsz;
    float k;             // Patatrack's field, GeV^-1 cm^-1 (PixelRecoUtilities::fieldInInvGev): R[cm] = pt / k
    uint32_t nPixelSoA;  // pixel rechits of the pixel tracks: hit indices >= nPixelSoA are OT rechits
    uint32_t nOTSoA;     // OT rechits of the pixel tracks
    uint32_t nOT;        // legacy OT hits = rows [0, nOT) of the LST hits block
    uint32_t nPixKeys;   // size of the pixel SoA-row -> legacy cluster key map
    int minQuality;      // pixelTrack::Quality (tight = 5, as hltPhase2PixelTracks)
    uint32_t nSeedMap;   // > 0: seedOfTrack[t] = the edm pixel track index of SoA track t (-1: none -> no pLS)
    // pLS momentum scaled by <Bz>_hits / Bz(0) with the menu's closed-form tracker field
    // (interface/math/TkBfield.h): Patatrack's helix uses the field at the origin, the endcap tracks see less
    bool ptFieldCorrection;
    float bz0;  // tkBz(0, 0, 0) (the field Patatrack's k comes from)
    // the pLS "last hit" r3LH (pseudo-hit 2, x of pseudo-hit 3) and the momentum p3LH.
    // 0 = the earlier version-9: the outermost hit itself, p3LH along the helix tangent at the hit's azimuth;
    // 1 = the Patatrack helix point at the outermost hit's transverse radius (the host takes the KF seed state ON
    //     the last hit, not the hit), p3LH the helix tangent there
    int pseudoLH;
    // the pLS PCA quantities (PCA point, PCA momentum, dxy, dz, superbin).
    // 0 = Patatrack's PCA; 1 = the PCA of the Patatrack helix RE-ANCHORED on the outermost hit (same
    //     curvature, the helix tangent at the hit), as the host: TSCBL of the KF seed state on the last hit
    int pcaAnchor;
    // 1 = every pLS field and pseudo-hit from the device KF on the pixel-track hits
    // (the host creator of hltInitialStepSeeds emulated, then TSCBL: MkFitAlpaka fitPixelSeeds); a pixel track whose
    // device creator fails gets no pLS. pseudoLH / pcaAnchor / ptFieldCorrection are then not used.
    int pixKF;
  };

  // per pixel track: the LST pLS of option (i) (scratch, one row per SoA track)
  struct PLS {
    int32_t pass;  // 1: becomes a pLS (quality, finite, pt cut); 0: dropped
    int32_t charge;
    int32_t superbin;
    int32_t seedIdx;  // SoA track index, or the host seed index with seedOfTrack
    uint8_t nHits;    // all hits of the track (= see_hitIdx size of the host seed)
    uint8_t nToSoA;
    uint8_t hitDetBits;
    int8_t isQuad;
    int8_t pixelType;
    float ptIn, ptErr, px, py, pz, etaErr, eta, phi, deltaPhi;
    float x[4], y[4], z[4];
    uint32_t detid[4];
    uint16_t clust[4];
    uint32_t idx[4];
  };

  // status counters (counts[]): 0 nPLS (before the LST cap), 1 nHitsIT, 2 nTracks, 3 tight tracks with > kMaxTrackHits or < 3 hits,
  // 4 hits without a legacy key, 5 tracks failing the finite check;
  // pixKF: 6 device creator failures (no pLS), 7 TSCBL not valid (host: zero PCA fields, kept),
  // 8 KF inputs not built: > kMaxPixSeedHits hits (no pLS),
  // 9 hits whose mkFit row position differs from the Patatrack hit by > 10 um (row-map check, expect 0),
  // 10 TSCBL outside the closed-form field volume (part of 7), 11 KF inputs not built: a hit without a key or beyond
  // the mkFit hit rows, 12 KF inputs not built: a hit without an mkFit layer
  constexpr int kNCounts = 14;

}  // namespace mkfitdev::lstin

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev::lstin {
  using ::mkfitdev::lstin::Params;
  using ::mkfitdev::lstin::PixelTrackHits;
  using ::mkfitdev::lstin::PLS;

  // per tight pixel track the inputs of fitPixelSeeds: its hits as (mkFit layer, hit index) in
  // radius order (SeedGeneratorFromProtoTracksEDProducer: HitLessByRadius, includeFourthHit), the region origin and pT
  void launchPixSeedIn(Queue& queue,
                       ::reco::TrackBlocksConstView tracks,
                       uint32_t maxTracks,
                       PixelTrackHits hits,
                       uint32_t const* pixKey,
                       uint32_t const* otKey,
                       int32_t const* seedOfTrack,
                       Params p,
                       ::mkfitdev::HitSoAConstView mkHits,
                       ::mkfitdev::lstseeds::PixSeedIn* out,
                       uint32_t* counts);

  // the device creator's state on the last hit per EDM pixel track (row = seedOfTrack[t], the
  // hltPhase2PixelTracks index) for the build module's hits-only pLS seeds; rows of failed / absent tracks keep charge 0.
  // in != nullptr: every row with a pixel seed list (nh >= 2) also gets its hits (nTotalHits, hits)
  void launchPixSeedStates(Queue& queue,
                           ::mkfitdev::TrackSoAView states,
                           int32_t nStates,
                           uint32_t maxTracks,
                           int32_t const* seedOfTrack,
                           uint32_t nSeedMap,
                           ::mkfitdev::lstseeds::PixSeedOut const* kf,
                           ::mkfitdev::lstseeds::PixSeedIn const* in = nullptr);

  // pass 1: per-track pLS + a deterministic serial scan (pLS index and pseudo-hit offset per track); counts on device
  void launchPLS(Queue& queue,
                 ::reco::TrackBlocksConstView tracks,
                 uint32_t maxTracks,
                 PixelTrackHits hits,
                 uint32_t const* pixKey,
                 uint32_t const* otKey,
                 uint32_t const* otDetId,
                 uint16_t const* otClust,
                 int32_t const* seedOfTrack,
                 Params p,
                 PLS* scratch,
                 uint32_t* pIdx,
                 uint32_t* hOff,
                 uint32_t* counts,
                 ::mkfitdev::lstseeds::PixSeedOut const* kf = nullptr);

  // pass 2: the OT hits from the device OT rechit SoA (rows [0, nOT) = SoA rows = cluster keys, with its global
  // position, DetId and cluster size), then the pLS rows and their pseudo-hits
  void launchFillSoA(Queue& queue,
                     ::lst::LSTInputView out,
                     uint32_t maxTracks,
                     uint32_t nPLSCap,
                     ::mkfitdev::OTRecHitSoA::ConstView ot,
                     Params p,
                     PLS const* scratch,
                     uint32_t const* pIdx,
                     uint32_t const* hOff);

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev::lstin

#endif

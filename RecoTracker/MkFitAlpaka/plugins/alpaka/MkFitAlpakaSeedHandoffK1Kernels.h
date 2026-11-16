#ifndef RecoTracker_MkFitAlpaka_plugins_alpaka_MkFitAlpakaSeedHandoffK1Kernels_h
#define RecoTracker_MkFitAlpaka_plugins_alpaka_MkFitAlpakaSeedHandoffK1Kernels_h

// K1: the device seed hand-off. LST track candidates (device) -> mkFit seed rows
// (TrackSoA as the carrier) with the per-TC body of LSTOutputConverter and its hit order: the pixel seed's
// radius-ordered list, then the OT hits by LST logical layer, within a layer by rank on (subdetector, span key) with
// equal keys in input order. Rows whose state needs CMSSW host objects (placeholder states; with the legacy pixel tracks
// also the dropOTHitsPurePLS creator refit) are requested from the host; without the legacy pixel tracks the creator
// refit runs on the device (deviceRefit: the device creator emulation of the LST seed fit, MkFitAlpaka fitHostLike).

#include <cstdint>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTracker/LSTCore/interface/TrackCandidatesSoA.h"
#include "RecoTracker/MkFitAlpaka/interface/hits/HitSoA.h"
#include "RecoTracker/MkFitAlpaka/interface/seeds/SeedSoA.h"
#include "RecoTracker/MkFitAlpaka/interface/tracks/TrackSoA.h"

namespace mkfitdev::k1 {

  // what the hand-off reads of a module (LSTOutputConverter's hitLess key, its T5 / T4 selections), by (mkFit layer,
  // short id in layer) = layerBase[layer] + detIDinLayer
  struct DetInfo {
    float sortKey;      // barrel rSpan().first, endcap |zSpan().first|
    int32_t topoLayer;  // TrackerTopology::layer
    int16_t sub;        // GeomDetEnumerators::SubDetector
    int8_t isOT;        // GeomDetEnumerators::isOuterTracker
    int8_t type;        // 1 Ph2PSP, 2 Ph2PSS, 0 other
    int8_t barrel;      // GeomDetEnumerators::isBarrel
    int8_t slot;        // OT: the LST logical layer of the module (barrel layer 1-6, endcap 6 + disk); 0 otherwise
    int16_t pad;
  };
  constexpr int kOTSlots = 12;      // logical layers 1..11
  constexpr int kSlotCapacity = 4;  // hits per logical layer (pixel-track OT hits + LST hits; more: error row)

  enum RowKind : int8_t {
    kSkip = 0,          // no seed (a T5 / T4 without a selected hit, a pixel seed index beyond the pixel tracks)
    kDevice = 1,        // complete on the device
    kHostRefit = 2,     // pLS with dropOTHitsPurePLS: the creator refit on the IT hits (host)
    kHostPixState = 3,  // placeholder state on a pixel-track hit (no device pixel state: charge 0; or a light seed)
    kHostOTState = 4,   // light seed with an OT last hit: placeholder state through the OT CPE (host)
    kError = 5,         // no pixel seed list / a hit without an mkFit row or layer / list overflow (counted, no seed)
    kDeviceRefit = 6  // pLS with dropOTHitsPurePLS, deviceRefit: the creator refit on the IT hits on the device; after
                      // the refit kDevice (fitted) or kSkip (the creator failed: no seed, as on the host)
  };

  // per TC row, copied to the host
  struct RowInfo {
    int32_t pixTrack;  // pixelTrackOfSeed of the seed (-1: none)
    int16_t nTot;      // hits before the SeedSoA cap (kMaxSeedHits)
    int8_t kind;       // RowKind
    int8_t deviceFit;  // bit 0: zero state, refitted by the build module's device seed fit
  };

  // host requests (fixed capacity, overflow counted); slots taken with an atomic counter, each entry carries its TC row
  constexpr int kMaxHostReq = 1024;
  constexpr int kReqHits = 8;
  struct HostReq {
    int32_t tc;    // TC row
    int32_t e;     // pixel track (or -1)
    int32_t kind;  // RowKind
    int32_t nHot;  // hits below (refit: the IT hits in list order; state rows: the last hit)
    ::mkfitdev::HitOnTrack hot[kReqHits];
    // state rows: the last hit's det table index (mkFit layer base + detIDinLayer) and global position (EventOfHits),
    // for the pixel placeholder state without the legacy pixel tracks
    int32_t det;
    float pos[3];
  };

  struct K1Config {
    int32_t nPixTracks;        // pixel tracks of the event (pixelSeedIndex range); < 0: states.nTracks()
    int32_t nStates;           // rows of the pixelSeedStates product
    int32_t nLayers;           // mkFit layers
    int32_t includeFourthHit;  // as SeedGeneratorFromProtoTracksEDProducer
    int32_t dropOTHitsPurePLS;
    int32_t maxITHitsToDrop;
    int32_t deviceFitStates;  // T5/T4/pT3/pT5 seeds with >= 2 hits: zero state (the build module's seed fit refits them)
    int32_t deviceRefit;      // the dropOTHitsPurePLS creator refits on the device (kDeviceRefit rows, refit list)
  };

  // a state made on the host for output row `row`
  struct HostState {
    int32_t row;
    int16_t charge;
    float par[6];
    float err[21];
  };

  // device counters: [0] host-request overflow, [1] error rows, [2] rows with > kMaxSeedHits hits, [3] device refits,
  // [4] device refits whose creator failed (no seed), [5] rows in the device refit list
  constexpr int kNCounters = 6;
  constexpr int kCntRefit = 3, kCntRefitFailed = 4, kCntRefitList = 5;

}  // namespace mkfitdev::k1

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev::k1 {

  using namespace ::mkfitdev::k1;

  // K1a: one thread per TC row r < tcs.nTrackCandidates(): rows[r] (TrackSoA, uncompacted: params, errors, charge,
  // nTotalHits, hits), info[r], host requests. states = hltInputLSTDevice:pixelSeedStates with its hit lists (and, with
  // cfg.nPixTracks < 0, the pixel-track count in nTracks). cfg.deviceRefit: the refit rows' TC rows go to refitList
  // (count counters[kCntRefitList]).
  void launchK1Rows(Queue& queue,
                    ::lst::TrackCandidatesBaseConst tcs,
                    uint32_t tcCapacity,
                    ::mkfitdev::TrackSoAConstView states,
                    ::mkfitdev::HitSoAConstView mk,
                    DetInfo const* dets,
                    int32_t const* layerBase,
                    int8_t const* layerIsPixel,
                    K1Config cfg,
                    ::mkfitdev::TrackSoAView rows,
                    RowInfo* info,
                    HostReq* req,
                    int32_t* nReq,
                    int32_t* counters,
                    int32_t* nTCOut,
                    int32_t* refitList);

  // after the device refit (refitRowsHostLike on the refitList rows, status per TC row): the rows' kind (kDevice or
  // kSkip) and the refit counters
  void launchK1RefitDone(
      Queue& queue, int32_t const* refitList, int32_t const* status, RowInfo* info, int32_t* counters);

  // K1b: rows[r] -> out[outIndex[r]] for outIndex >= 0 (label = output row), then the host-made states.
  void launchK1Compact(Queue& queue,
                       ::mkfitdev::TrackSoAConstView rows,
                       int32_t const* outIndex,
                       uint32_t nTC,
                       HostState const* hostStates,
                       int32_t nHostStates,
                       int32_t nOverflowHits,
                       int32_t nOut,
                       ::mkfitdev::TrackSoAView out);

  // build module (deviceSeeds): K1 seed rows (TrackSoA) -> the device seed table, as packSeeds of the host seeds
  void launchSeedsFromK1(
      Queue& queue, ::mkfitdev::TrackSoAConstView in, int32_t n, uint32_t status, ::mkfitdev::SeedSoAView seeds);

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev::k1

#endif

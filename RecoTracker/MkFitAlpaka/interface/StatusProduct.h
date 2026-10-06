#ifndef RecoTracker_MkFitAlpaka_interface_StatusProduct_h
#define RecoTracker_MkFitAlpaka_interface_StatusProduct_h

// Per-event status of the device chain. One small product per event with every overflow,
// truncation and skip counter of the chain. All zero in a healthy event; any non-zero entry means the event's output
// differs from MkFitCore for a reason the port knows about (a fixed capacity was exceeded, the event was skipped).
//   - Device: the build module owns one MkFitStatusDeviceObject per event, zeroes it (zeroStatus), and adds its
//     counters with collectStatus (src/alpaka/Status.dev.cc, interface/alpaka/StatusCollect.h) or by atomicAdd on
//     counter[i] from its own kernels. The framework copies it to the host (PortableObject CopyToHost).
//   - Host: the host converter calls mkfitdev::warnIfNotClean(status, label) (LogWarning on non-zero), the test /
//     menu checks call mkfitdev::assertClean (throws). Both are in interface/StatusReport.h.
// Adding a counter: append an enum entry before kNumStatusCounters and its name in statusCounterName, nothing else
// (the layout is a fixed array, kMaxStatusCounters slots; no dictionary change).

#include <cstdint>

#include "DataFormats/Portable/interface/PortableHostObject.h"

namespace mkfitdev {

  enum StatusCounter : int {
    kEventSkipped = 0,     // EventOfHits beyond the device build limits: mkFit skipped for the event
    kEohOverflowFirst,     // EventOfHits: nOverflowFirst (bin first-hit index >= 2^18)
    kEohOverflowCount,     // EventOfHits: nOverflowCount (bin hit count >= 2^14)
    kSeedHitsTruncated,    // seeds with more than kMaxSeedHits hits (SeedSoA nOverflowHits): extra hits lost
    kHotOverflowSeeds,     // seeds failed by a HoT pool overflow (SeedCandsSoA nOverflowHots / EngineCounters)
    kOptsOverflow,         // per-seed option list overflow (SeedCandsSoA nOverflowOpts)
    kExtrasOverflow,       // per-seed extras overflow (SeedCandsSoA nOverflowExtras)
    kCandHitsOverflow,     // K2 found more hits than kMaxHitsPerCand (EngineCounters nOverflowHits)
    kTrackOverflow,        // tracks dropped because the TrackSoA was full (TrackSoA nOverflowTracks)
    kTrackHitsOverflow,    // tracks whose hit list exceeded kMaxTrkHits (TrackSoA nOverflowHits)
    kFitOverflow,          // final fit: more than kMaxTrkHits hits to order (FitCounters nOverflow)
    kFitHitCountMismatch,  // final fit: nFoundHits != hits with index >= 0 (FitCounters nHitCountMismatch)
    kRepackErrors,         // backward-search repack applied a number of times other than once
    kCpeClusterOverflow,   // device fit CPE clusters: digis whose SoA cluster index is out of range
    kCpeClusterRefErrors,  // device fit CPE clusters: mkFit pixel rows whose (module, originalId) names no SoA cluster
    kPixSeedTooManyHits,   // hltInputLSTDevice pixKF: pixel tracks with > kMaxPixSeedHits hits (no pLS for them)
    kPixSeedNoRow,         // hltInputLSTDevice pixKF: pixel-track hits without a device EventOfHits row (no pLS)
    kPixSeedNoLayer,       // hltInputLSTDevice pixKF: pixel-track hits on a row without an mkFit layer (no pLS)
    kPixSeedNoEOH,         // hltInputLSTDevice pixKF: empty device EventOfHits, pLS seeds keep a placeholder state
    kNumStatusCounters
  };

  constexpr int kMaxStatusCounters = 32;
  static_assert(kNumStatusCounters <= kMaxStatusCounters, "MkFitStatus: raise kMaxStatusCounters");

  inline constexpr const char* statusCounterName(int i) {
    switch (i) {
      case kEventSkipped:
        return "eventSkipped";
      case kEohOverflowFirst:
        return "eohOverflowFirst";
      case kEohOverflowCount:
        return "eohOverflowCount";
      case kSeedHitsTruncated:
        return "seedHitsTruncated";
      case kHotOverflowSeeds:
        return "hotOverflowSeeds";
      case kOptsOverflow:
        return "optsOverflow";
      case kExtrasOverflow:
        return "extrasOverflow";
      case kCandHitsOverflow:
        return "candHitsOverflow";
      case kTrackOverflow:
        return "trackOverflow";
      case kTrackHitsOverflow:
        return "trackHitsOverflow";
      case kFitOverflow:
        return "fitOverflow";
      case kFitHitCountMismatch:
        return "fitHitCountMismatch";
      case kRepackErrors:
        return "repackErrors";
      case kCpeClusterOverflow:
        return "cpeClusterOverflow";
      case kCpeClusterRefErrors:
        return "cpeClusterRefErrors";
      case kPixSeedTooManyHits:
        return "pixSeedTooManyHits";
      case kPixSeedNoRow:
        return "pixSeedNoRow";
      case kPixSeedNoLayer:
        return "pixSeedNoLayer";
      case kPixSeedNoEOH:
        return "pixSeedNoEOH";
      default:
        return "unused";
    }
  }

  struct MkFitStatus {
    uint32_t counter[kMaxStatusCounters];
  };

  using MkFitStatusHostObject = PortableHostObject<MkFitStatus>;

}  // namespace mkfitdev

#endif

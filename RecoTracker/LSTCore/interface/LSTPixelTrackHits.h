#ifndef RecoTracker_LSTCore_interface_LSTPixelTrackHits_h
#define RecoTracker_LSTCore_interface_LSTPixelTrackHits_h

#include <cstdint>
#include <vector>

#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"

namespace lst {

  // Host-only pointers to the rechits of the fitted pixel tracks: one range per fitted pixel track in seed-index
  // order, each track's hits in the order of its reco::Track. The hits of seed index i are
  // hits[offsets[i] .. offsets[i + 1]), and offsets has one entry per fitted track plus one. The pointers point into
  // the TrackingRecHitCollection of the pixel tracks (the producer's pixelTracks input) and are valid for the event.
  // Wrapped in a struct because a std::vector<T*> event product fails the framework's contained-type dictionary check.
  struct LSTPixelTrackHits {
    std::vector<uint32_t> offsets;
    std::vector<TrackingRecHit const*> hits;
  };

}  // namespace lst

#endif

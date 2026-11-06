#ifndef RecoTracker_MkFitAlpaka_src_alpaka_clean_CleanAlgo_h
#define RecoTracker_MkFitAlpaka_src_alpaka_clean_CleanAlgo_h

// Host-side driver of the device duplicate cleaner and LST-step track filter. All work is enqueued on the given
// queue; nothing synchronises. Declaration only: the kernels are instantiated once, in src/alpaka/Clean.dev.cc
// (library symbols); callers include this header and link the package library.
// Buffers are sized once for 'capacity' tracks (the TrackSoA row count).

#include <optional>
#include <stdexcept>
#include <string>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "RecoTracker/MkFitAlpaka/interface/tracks/TrackSoA.h"
#include "RecoTracker/MkFitAlpaka/src/alpaka/clean/CleanFunctions.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev::clean {

  class CleanAlgo {
  public:
    CleanAlgo(Queue& queue, int capacity);

    // StdSeq::clean_duplicates_sharedhits_pixelseed decisions: sets the 'duplicate' column of every row of 'tracks'
    // (rows [0, tracks.nTracks())). Does not remove anything.
    void flagDuplicates(Queue& queue, ::mkfitdev::TrackSoAView tracks, ::mkfitdev::clean::DupCleanParams const& params);

    // StdSeq::remove_duplicates: stable copy of the rows with duplicate == 0 of the last flagDuplicates() input into
    // 'out' (capacity >= in.nTracks()); sets out.nTracks().
    void removeDuplicates(Queue& queue, ::mkfitdev::TrackSoAConstView in, ::mkfitdev::TrackSoAView out);

    // LST-step quality filter (qfilter_n_hits_pixseed && !qfilter_nan_n_silly) on finished tracks, then stable copy of
    // the passing rows into 'out'.
    void filterTracks(Queue& queue, ::mkfitdev::TrackSoAConstView in, ::mkfitdev::TrackSoAView out, int minHitsQF);

  private:
    void checkRows(int rows) const;
    void compact(Queue& queue, ::mkfitdev::TrackSoAConstView in, ::mkfitdev::TrackSoAView out);

    int capacity_;
    cms::alpakatools::device_buffer<Device, float[]> ct_;
    cms::alpakatools::device_buffer<Device, int8_t[]> hasPix_;  // per track hasPixelHits
    cms::alpakatools::device_buffer<Device, int[]> cell_;
    cms::alpakatools::device_buffer<Device, int[]> flags_;
    cms::alpakatools::device_buffer<Device, int[]> keep_;
    cms::alpakatools::device_buffer<Device, int[]> dest_;
    cms::alpakatools::device_buffer<Device, int[]> pairCount_;
    cms::alpakatools::device_buffer<Device, int[]> pairOffset_;
    cms::alpakatools::device_buffer<Device, int[]> wild_;
    cms::alpakatools::device_buffer<Device, int[]> cellContent_;
    cms::alpakatools::device_buffer<Device, int[]> cellCount_;
    cms::alpakatools::device_buffer<Device, int[]> cellStart_;
    cms::alpakatools::device_buffer<Device, int[]> cellCursor_;
    cms::alpakatools::device_buffer<Device, int[]> counters_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev::clean

#endif

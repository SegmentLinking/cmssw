#ifndef RecoTracker_MkFitAlpaka_src_alpaka_engine_FitHandoff_h
#define RecoTracker_MkFitAlpaka_src_alpaka_engine_FitHandoff_h

// Device handoff building -> final fit.
//   selectFitInput   MkFitFitProducer's candCutSel preselection (MkFitFitProducer.cc:139-151) on the device:
//                    the rows of the building's TrackSoA that pass are copied, IN INPUT ORDER, into the fit's TrackSoA
//                    (an order-preserving compaction: one-block count + exclusive scan, then a parallel row copy).
//                    The kept count is written to out.nTracks() on the device; no host synchronization.
//   buildClusterCpe  the per-pixel-hit cluster quantities of the device CPE (ClusterCpe: edges, edge charges, charge,
//                    module) for EVERY mkFit pixel row, computed on the device from the pixel digi + cluster SoA (the
//                    inputs of the legacy SiPixelDigisClustersFromSoAAlpaka clusters), reproducing the legacy cluster
//                    exactly: same digi selection, AccretionCluster's 256-pixel cap (first 256 digis in digi order),
//                    SiPixelCluster's 255 span cap. Integer atomics only (deterministic). The host supplies, per mkFit
//                    pixel row (= legacy cluster key), the SoA module index and the SoA cluster id in the module
//                    (SiPixelCluster::originalId): the legacy cluster order inside a module is a heap sort by
//                    minPixelRow, which is not reproduced on the device.

#include <cstdint>

#include "DataFormats/SiPixelClusterSoA/interface/SiPixelClustersSoA.h"
#include "DataFormats/SiPixelDigiSoA/interface/SiPixelDigisSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTracker/MkFitAlpaka/interface/fit/CpeGeneric.h"
#include "RecoTracker/MkFitAlpaka/interface/tracks/TrackSoA.h"

namespace mkfitdev::handoff {

  // MkFitFitProducer candCutSel parameters (the menu module's values)
  struct CandCutSel {
    bool enabled = false;
    float minPt = 0.f;
    int minNHits = 0;
    float minPtRelaxed = 0.f;
    float minAbsEtaRelaxed = 0.f;
  };

  // per mkFit pixel row: SoA module index (-1: unknown module, no CPE) and SoA cluster id inside the module
  struct ClusterRef {
    int32_t module;
    int32_t ic;
  };

  // overflow / consistency counters of buildClusterCpe (device, zeroed by the caller)
  struct ClusterCpeCounters {
    int32_t nClusterOverflow;  // digi whose SoA cluster index is beyond the cluster capacity
    int32_t nRefOutOfRange;    // mkFit row whose (module, ic) does not name a SoA cluster
    int32_t nBigClusters;      // clusters with more than 256 pixels (legacy truncation reproduced)
  };

}  // namespace mkfitdev::handoff

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev::handoff {

  using ::mkfitdev::handoff::CandCutSel;

  // `in`: building output (rows [0, in.nTracks()) valid, device count); `capacity` = in's row capacity (host value);
  // `out` must have capacity >= `capacity`. Writes out rows [0, kept), out.nTracks() = kept and the overflow scalars
  // (copied from `in`). Asynchronous in `queue`.
  void selectFitInput(Queue& queue,
                      ::mkfitdev::TrackSoAConstView in,
                      int capacity,
                      CandCutSel const& sel,
                      ::mkfitdev::TrackSoAView out);

  // ClusterCpe of the mkFit pixel rows [0, nRows) from the device digis ([0, nDigis)) and clusters (nClusters, the
  // SoA cluster count = capacity of the per-cluster scratch). `refs` [nRows] and `out` [nRows] are device memory;
  // `counters` zeroed device memory. Asynchronous in `queue`.
  void buildClusterCpe(Queue& queue,
                       SiPixelDigisSoAConstView digis,
                       uint32_t nDigis,
                       SiPixelClustersSoAConstView clusters,
                       uint32_t nClusters,
                       const ::mkfitdev::handoff::ClusterRef* refs,
                       uint32_t nRows,
                       ::mkfitdev::cpe::ClusterCpe* out,
                       ::mkfitdev::handoff::ClusterCpeCounters* counters);

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev::handoff

#endif

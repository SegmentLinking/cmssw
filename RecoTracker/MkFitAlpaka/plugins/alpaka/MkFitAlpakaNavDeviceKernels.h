#ifndef RecoTracker_MkFitAlpaka_plugins_alpaka_MkFitAlpakaNavDeviceKernels_h
#define RecoTracker_MkFitAlpaka_plugins_alpaka_MkFitAlpakaNavDeviceKernels_h
// The converter's missing-hit navigation on the device, after the device fit
#include <cstdint>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTracker/MkFitAlpaka/interface/math/NavSearch.h"
#include "RecoTracker/MkFitAlpaka/interface/navdev/NavResultSoA.h"
#include "RecoTracker/MkFitAlpaka/interface/tracks/TrackSoA.h"

#include "../MkFitAlpakaNavDevice.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev::navdevice {
  constexpr int kNavKinds =
      3;  // device call lists: 0 OT endcap, 1 barrel with stacked (OT) rods, 2 barrel with pixel rods
  // device pointers of the ES tables (MkFitAlpakaNavDeviceProducer's per-device copy)
  struct NavDeviceTables {
    ::mkfitdev::navdev::NavTables flat;    // NavSearch.h flat tables
    ::mkfitdev::nav::Table const* layers;  // compatibleLayers candidate lists (GeometricSearchTracker::allLayers order)
    int const* layerFlat;                  // 3 per layer: kind, first, count (first < 0: host search)
    int const* mkFitToLayer;               // mkFit layer number -> layer index
    int nMkFit, nLayers;
    // first layer slot of each device call list: list k holds at most 2 * capacity * (listStart[k + 1] - listStart[k])
    // calls (one per (track, direction, layer) of its layers)
    int listStart[kNavKinds + 1];
  };
  // the converter's Chi2MeasurementEstimator(30., -3.0, 0.5, 2.0, 0.5, 1.e12)
  struct NavEstimator {
    double maxSagitta, minTolerance2, nSigma, maxDisplacement;
  };
  // the rods part of an OT barrel call, kept for the rings kernel
  struct NavPart {
    int front, n, overflow, detTests;
  };
  constexpr int kNavSearchBlock = 32;  // threads per block of the navigation kernels
  // threads of each search kernel's grid: the group-list scratch has navSearchThreads(capacity) * kNavArena entries
  int navSearchThreads(int capacity);
  // rows [0, trk.nTracks()) of trk: per (track, direction, layer) the front det of compatibleDets, or a code
  // (NavResultSoA.h); out has capacity * 2 * nLayers rows. Per-event scratch: starts (capacity), calls (2 * capacity *
  // listStart[kNavKinds]), nCalls (kNavKinds, zeroed by the caller on the queue), arena (navSearchThreads(capacity) *
  // kNavArena), memo / memoDet (navSearchThreads(capacity) * kNavMemo), parts (2 * capacity * (listStart[2] -
  // listStart[1]): the calls of list 1).
  void launchNavDevice(Queue& queue,
                       ::mkfitdev::TrackSoAConstView trk,
                       int capacity,
                       NavDeviceTables const& tables,
                       NavEstimator const& est,
                       ::mkfitdev::navdev::DetTestStart* starts,
                       int* calls,
                       int* nCalls,
                       ::mkfitdev::navdev::NavGroups* arena,
                       ::mkfitdev::navdev::DetTestResult* memo,
                       int* memoDet,
                       NavPart* parts,
                       ::mkfitdev::NavResultSoAView out);
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev::navdevice

#endif

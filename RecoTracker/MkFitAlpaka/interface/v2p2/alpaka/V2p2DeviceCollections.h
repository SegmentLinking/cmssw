#ifndef RecoTracker_MkFitAlpaka_interface_v2p2_alpaka_V2p2DeviceCollections_h
#define RecoTracker_MkFitAlpaka_interface_v2p2_alpaka_V2p2DeviceCollections_h

// Device collections of the v2p2 search (interface/v2p2/V2p2SoA.h) and the per-event set of them. One set serves
// the forward and the backward search, which run one after the other on the event's queue.

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTracker/MkFitAlpaka/interface/v2p2/V2p2SoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev {
  using ::mkfitdev::V2p2CandSoA;
  using ::mkfitdev::V2p2LayerHitSoA;
  using ::mkfitdev::V2p2NodeSoA;
  using ::mkfitdev::V2p2SeedSoA;

  using V2p2SeedDeviceCollection = PortableCollection<V2p2SeedSoA>;
  using V2p2CandDeviceCollection = PortableCollection<V2p2CandSoA>;
  using V2p2LayerHitDeviceCollection = PortableCollection<V2p2LayerHitSoA>;
  using V2p2NodeDeviceCollection = PortableCollection<V2p2NodeSoA>;

  // Scratch of the v2p2 search for up to maxSeeds seed rows.
  //   preselStride = V2p2Params::max_presel_hits (rows of V2p2LayerHitSoA per CandIncoming)
  //   nodesPerSeed = node pool rows per seed (MkFitAlpakaBuildProducer parameter v2p2NodesPerSeed)
  struct V2p2Buffers {
    V2p2Buffers(Queue& queue, int maxSeeds, int preselStride, int nodesPerSeed);
    int preselStride;
    int nodeCapacity;  // maxSeeds * nodesPerSeed
    V2p2SeedDeviceCollection seeds;
    V2p2CandDeviceCollection cands;
    V2p2LayerHitDeviceCollection layerHits;
    V2p2NodeDeviceCollection nodes;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev

#endif

#ifndef RecoTracker_MkSeederAlpaka_interface_MkSeederESData_h
#define RecoTracker_MkSeederAlpaka_interface_MkSeederESData_h

// The seeder's static tables (MkSeederESProducer): the layers' geometry and binning axes, the two chains, the shape
// bands, the gap map, the modules and the material of the seed fits, the patterns' order and the triplet MLP. They
// depend only on the configuration and the geometry. Built once on the host per IOV and copied once per device; on
// a host backend the device product is the host product (the collections are shared).

#include <array>
#include <cstdint>
#include <memory>
#include <type_traits>

#include "DataFormats/Portable/interface/PortableCollection.h"
#include "DataFormats/Portable/interface/PortableObject.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToDevice.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTracker/MkSeederAlpaka/interface/MkSeederHostTables.h"
#include "RecoTracker/MkSeederAlpaka/interface/MkSeederTablesHostCollection.h"

namespace mkseeder {

  template <typename TDev>
  struct MkSeederESData {
    // host-side values, copied with the product
    int32_t nLayers = 0;
    // the chains' constants: [0] the +z side, [1] the -z side
    std::array<MkSeederChainParameters, 2> chainParameters{};
    // the patterns' order: the configured positions, the table's rows
    int32_t nPatternPositions = 0, nPatternEntries = 0;

    std::shared_ptr<const PortableCollection<TDev, MkSeederLayerGeometrySoA>> layerGeometry;
    std::array<std::shared_ptr<const PortableCollection<TDev, MkSeederChainSoA>>, 2> chains;
    std::shared_ptr<const PortableCollection<TDev, MkSeederShapeSoA>> shapes;
    std::shared_ptr<const PortableCollection<TDev, MkSeederGapMapSoA>> gapMap;
    std::shared_ptr<const PortableCollection<TDev, MkSeederModulesSoA>> modules;
    std::shared_ptr<const PortableCollection<TDev, MkSeederMaterialSoA>> material;
    std::shared_ptr<const PortableCollection<TDev, MkSeederPatternsSoA>> patterns;
    std::shared_ptr<const PortableObject<TDev, MkSeederTripletMlpWeights>> tripletMlpWeights;

    // host-side, shared by the host and device products: the binned layers' axes (the hits' sort keys)
    std::shared_ptr<const MkSeederHostTables> hostTables;
  };

  using MkSeederESDataHost = MkSeederESData<alpaka_common::DevHost>;

}  // namespace mkseeder

namespace cms::alpakatools {

  template <>
  struct CopyToDevice<mkseeder::MkSeederESDataHost> {
    template <typename TQueue>
    static mkseeder::MkSeederESData<alpaka::Dev<TQueue>> copyAsync(TQueue& queue,
                                                                   mkseeder::MkSeederESDataHost const& hostData) {
      using TDev = alpaka::Dev<TQueue>;
      mkseeder::MkSeederESData<TDev> deviceData;
      deviceData.nLayers = hostData.nLayers;
      deviceData.chainParameters = hostData.chainParameters;
      deviceData.nPatternPositions = hostData.nPatternPositions;
      deviceData.nPatternEntries = hostData.nPatternEntries;
      deviceData.hostTables = hostData.hostTables;
      if constexpr (std::is_same_v<TDev, alpaka_common::DevHost>) {
        deviceData.layerGeometry = hostData.layerGeometry;
        deviceData.chains = hostData.chains;
        deviceData.shapes = hostData.shapes;
        deviceData.gapMap = hostData.gapMap;
        deviceData.modules = hostData.modules;
        deviceData.material = hostData.material;
        deviceData.patterns = hostData.patterns;
        deviceData.tripletMlpWeights = hostData.tripletMlpWeights;
      } else {
        auto copyCollection = [&queue](auto const& host) {
          using THost = std::remove_cvref_t<decltype(*host)>;
          using TDevice = std::remove_cvref_t<decltype(CopyToDevice<THost>::copyAsync(queue, *host))>;
          return std::make_shared<const TDevice>(CopyToDevice<THost>::copyAsync(queue, *host));
        };
        deviceData.layerGeometry = copyCollection(hostData.layerGeometry);
        for (int side = 0; side < 2; ++side)
          deviceData.chains[side] = copyCollection(hostData.chains[side]);
        deviceData.shapes = copyCollection(hostData.shapes);
        deviceData.gapMap = copyCollection(hostData.gapMap);
        deviceData.modules = copyCollection(hostData.modules);
        deviceData.material = copyCollection(hostData.material);
        deviceData.patterns = copyCollection(hostData.patterns);
        deviceData.tripletMlpWeights = copyCollection(hostData.tripletMlpWeights);
      }
      return deviceData;
    }
  };

}  // namespace cms::alpakatools

#endif

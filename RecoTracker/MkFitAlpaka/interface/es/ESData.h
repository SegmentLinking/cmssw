#ifndef RecoTracker_MkFitAlpaka_interface_es_ESData_h
#define RecoTracker_MkFitAlpaka_interface_es_ESData_h

// ES product of MkFitAlpaka: device-resident mkFit geometry, material and iteration configuration
// (LSTESData style). The host product ESData<DevHost> is made by MkFitAlpakaESProducer from the geometry-only
// product (ESGeometryData.h, MkFitAlpakaGeometryESProducer), the MkFitCore MkFitGeometry and the IterationConfig; the
// framework copies it to each device with CopyToDevice below.

#include <memory>
#include <type_traits>

#include "DataFormats/Portable/interface/PortableCollection.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/Portable/interface/PortableHostObject.h"
#include "DataFormats/Portable/interface/PortableObject.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToDevice.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

#include "RecoTracker/MkFitAlpaka/interface/es/ESConfig.h"
#include "RecoTracker/MkFitAlpaka/interface/es/ESGeometryData.h"
#include "RecoTracker/MkFitAlpaka/interface/es/ESLayouts.h"
#include "RecoTracker/MkFitAlpaka/interface/es/ESView.h"
#include "RecoTracker/MkFitAlpakaMath/interface/es/MaterialView.h"

namespace mkfit {
  class TrackerInfo;
  class IterationConfig;
}  // namespace mkfit

namespace mkfitdev {

  template <typename TDev>
  struct ESData {
    // Using shared_ptr so that for the serial backend all streams can use the same data (as LSTESData)
    std::shared_ptr<const PortableCollection<TDev, LayerInfoSoA>> layers;
    std::shared_ptr<const PortableCollection<TDev, ModuleInfoSoA>> modules;
    std::shared_ptr<const PortableCollection<TDev, ModuleShapeSoA>> shapes;
    std::shared_ptr<const PortableCollection<TDev, DetIdMapSoA>> detIdMap;
    std::shared_ptr<const PortableCollection<TDev, MaterialSoA>> material;
    std::shared_ptr<const PortableObject<TDev, ESConfig>> config;
    // host copy of the configuration, shared between the ESData<TDev> of all devices (host-side steering)
    std::shared_ptr<const PortableHostObject<ESConfig>> hostConfig;
    ESSizes sizes;

    ESData(std::shared_ptr<const PortableCollection<TDev, LayerInfoSoA>> layersIn,
           std::shared_ptr<const PortableCollection<TDev, ModuleInfoSoA>> modulesIn,
           std::shared_ptr<const PortableCollection<TDev, ModuleShapeSoA>> shapesIn,
           std::shared_ptr<const PortableCollection<TDev, DetIdMapSoA>> detIdMapIn,
           std::shared_ptr<const PortableCollection<TDev, MaterialSoA>> materialIn,
           std::shared_ptr<const PortableObject<TDev, ESConfig>> configIn,
           std::shared_ptr<const PortableHostObject<ESConfig>> hostConfigIn,
           ESSizes const& sizesIn)
        : layers(std::move(layersIn)),
          modules(std::move(modulesIn)),
          shapes(std::move(shapesIn)),
          detIdMap(std::move(detIdMapIn)),
          material(std::move(materialIn)),
          config(std::move(configIn)),
          hostConfig(std::move(hostConfigIn)),
          sizes(sizesIn) {}

    // Views into this memory space; pass by value to kernels.
    ESView view() const {
      return ESView{{layers->const_view(),
                     modules->const_view(),
                     shapes->const_view(),
                     detIdMap->const_view(),
                     MaterialView{material->const_view().metadata().addressOf_bbxi(),
                                  material->const_view().metadata().addressOf_radl(),
                                  sizes.matNBinsZ,
                                  sizes.matNBinsR,
                                  sizes.matFacZ,
                                  sizes.matFacR,
                                  sizes.bField,
                                  sizes.bFieldFit}},
                    config->const_data()};
    }

    ESConfig const& hostConfigValue() const { return hostConfig->const_value(); }
  };

  using ESDataHost = ESData<alpaka_common::DevHost>;

  // Fill the host product from the geometry-only product (fillESGeometryHost of the same TrackerInfo: its module,
  // shape, detId and material collections are shared, its layer rows copied), the MkFitCore host ES objects
  // (MkFitGeometry::trackerInfo(), the IterationConfig of the iteration) and the runtime mkfit::Config values. Throws
  // cms::Exception on anything that does not fit the fixed capacities (es::kMaxRegions, es::kMaxPlanLayers), non-empty
  // hit-window parameters, unknown function names, a geometry product of another TrackerInfo (layer / module counts).
  // buildingFitField: track building with the final fit's field constants (mkfit::Config::mag_fit_*, fitted to the
  // field map) instead of MkFitCore's (mag_*, CMSSW's ParabolicMf). DEVIATION DEV-11.
  std::unique_ptr<ESDataHost> fillESDataHost(ESGeometryDataHost const& geometry,
                                             mkfit::TrackerInfo const& trackerInfo,
                                             mkfit::IterationConfig const& iterConfig,
                                             bool buildingFitField = false);
  // the same with its own geometry fill
  std::unique_ptr<ESDataHost> fillESDataHost(mkfit::TrackerInfo const& trackerInfo,
                                             mkfit::IterationConfig const& iterConfig,
                                             bool buildingFitField = false);

}  // namespace mkfitdev

namespace cms::alpakatools {

  template <>
  struct CopyToDevice<mkfitdev::ESDataHost> {
    template <typename TQueue>
    static mkfitdev::ESData<alpaka::Dev<TQueue>> copyAsync(TQueue& queue, mkfitdev::ESDataHost const& src) {
      using TDev = alpaka::Dev<TQueue>;
      if constexpr (std::is_same_v<TDev, alpaka_common::DevHost>) {
        return mkfitdev::ESData<TDev>(
            src.layers, src.modules, src.shapes, src.detIdMap, src.material, src.config, src.hostConfig, src.sizes);
      } else {
        auto copyColl = [&queue](auto const& hostColl) {
          using Layout = typename std::remove_cvref_t<decltype(hostColl)>::Layout;
          return std::make_shared<const PortableCollection<TDev, Layout>>(
              CopyToDevice<PortableHostCollection<Layout>>::copyAsync(queue, hostColl));
        };
        return mkfitdev::ESData<TDev>(
            copyColl(*src.layers),
            copyColl(*src.modules),
            copyColl(*src.shapes),
            copyColl(*src.detIdMap),
            copyColl(*src.material),
            std::make_shared<const PortableObject<TDev, mkfitdev::ESConfig>>(
                CopyToDevice<PortableHostObject<mkfitdev::ESConfig>>::copyAsync(queue, *src.config)),
            src.hostConfig,
            src.sizes);
      }
    }
  };

}  // namespace cms::alpakatools

#endif

#ifndef RecoTracker_MkFitAlpaka_interface_es_ESGeometryData_h
#define RecoTracker_MkFitAlpaka_interface_es_ESGeometryData_h

// Geometry-only ES product of MkFitAlpaka: the device-resident mkFit layers, module planes, module shapes, detId map,
// (z, r) material map and field constants of mkfit::TrackerInfo, without any iteration configuration. Made by
// MkFitAlpakaGeometryESProducer from the MkFitCore MkFitGeometry; the iteration's ES product (ESData.h,
// MkFitAlpakaESProducer) is built on it (it shares the module, shape, detId and material collections and copies the
// layer rows, adding the IterationLayerConfig columns). Other consumers of the mkFit geometry on the device (e.g. a
// seeder) can use this product without an mkFit iteration. The framework copies the host product to each device.

#include <memory>
#include <type_traits>

#include "DataFormats/Portable/interface/PortableCollection.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToDevice.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

#include "RecoTracker/MkFitAlpaka/interface/es/ESLayouts.h"
#include "RecoTracker/MkFitAlpaka/interface/es/ESView.h"
#include "RecoTracker/MkFitAlpaka/interface/es/MaterialView.h"

namespace mkfit {
  class TrackerInfo;
}  // namespace mkfit

namespace mkfitdev {

  // Sizes and material-map scalars, kept on the host for every memory space (kernel launch sizing, MaterialView).
  struct ESSizes {
    int nLayers = 0;
    int nModules = 0;
    int nShapes = 0;
    int detIdMapCapacity = 0;
    int nMaterialBins = 0;
    int matNBinsZ = 0;
    int matNBinsR = 0;
    float matFacZ = 0.f;
    float matFacR = 0.f;
    // mkfit::Config::mag_* at fill time (from the ES): track building. In the geometry product always MkFitCore's
    // (CMSSW's ParabolicMf); the iteration's product may take bFieldFit instead (MkFitAlpakaESProducer buildingFitField)
    Config::BFieldParams bField;
    // mkfit::Config::mag_fit_* at fill time (from the ES): the final fit
    Config::BFieldParams bFieldFit{Config::mag_fit_c1, Config::mag_fit_b0, Config::mag_fit_b1, Config::mag_fit_a};
  };

  template <typename TDev>
  struct ESGeometryData {
    // shared_ptr: the serial backend's streams and the iteration ES products built on this one share the collections
    std::shared_ptr<const PortableCollection<TDev, LayerInfoSoA>> layers;  // select_* columns 0 (no iteration)
    std::shared_ptr<const PortableCollection<TDev, ModuleInfoSoA>> modules;
    std::shared_ptr<const PortableCollection<TDev, ModuleShapeSoA>> shapes;
    std::shared_ptr<const PortableCollection<TDev, DetIdMapSoA>> detIdMap;
    std::shared_ptr<const PortableCollection<TDev, MaterialSoA>> material;
    ESSizes sizes;

    ESGeometryData(std::shared_ptr<const PortableCollection<TDev, LayerInfoSoA>> layersIn,
                   std::shared_ptr<const PortableCollection<TDev, ModuleInfoSoA>> modulesIn,
                   std::shared_ptr<const PortableCollection<TDev, ModuleShapeSoA>> shapesIn,
                   std::shared_ptr<const PortableCollection<TDev, DetIdMapSoA>> detIdMapIn,
                   std::shared_ptr<const PortableCollection<TDev, MaterialSoA>> materialIn,
                   ESSizes const& sizesIn)
        : layers(std::move(layersIn)),
          modules(std::move(modulesIn)),
          shapes(std::move(shapesIn)),
          detIdMap(std::move(detIdMapIn)),
          material(std::move(materialIn)),
          sizes(sizesIn) {}

    // Views into this memory space; pass by value to kernels.
    ESGeometryView view() const {
      return ESGeometryView{layers->const_view(),
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
                                         sizes.bFieldFit}};
    }
  };

  using ESGeometryDataHost = ESGeometryData<alpaka_common::DevHost>;

  // Fill the host product from MkFitGeometry::trackerInfo() and the runtime mkfit::Config field constants (set by
  // MkFitGeometryESProducer). Throws cms::Exception on a module with detId 0 or a duplicate detId.
  std::unique_ptr<ESGeometryDataHost> fillESGeometryHost(mkfit::TrackerInfo const& trackerInfo);

}  // namespace mkfitdev

namespace cms::alpakatools {

  template <>
  struct CopyToDevice<mkfitdev::ESGeometryDataHost> {
    template <typename TQueue>
    static mkfitdev::ESGeometryData<alpaka::Dev<TQueue>> copyAsync(TQueue& queue,
                                                                   mkfitdev::ESGeometryDataHost const& src) {
      using TDev = alpaka::Dev<TQueue>;
      if constexpr (std::is_same_v<TDev, alpaka_common::DevHost>) {
        return mkfitdev::ESGeometryData<TDev>(
            src.layers, src.modules, src.shapes, src.detIdMap, src.material, src.sizes);
      } else {
        auto copyColl = [&queue](auto const& hostColl) {
          using Layout = typename std::remove_cvref_t<decltype(hostColl)>::Layout;
          return std::make_shared<const PortableCollection<TDev, Layout>>(
              CopyToDevice<PortableHostCollection<Layout>>::copyAsync(queue, hostColl));
        };
        return mkfitdev::ESGeometryData<TDev>(copyColl(*src.layers),
                                              copyColl(*src.modules),
                                              copyColl(*src.shapes),
                                              copyColl(*src.detIdMap),
                                              copyColl(*src.material),
                                              src.sizes);
      }
    }
  };

}  // namespace cms::alpakatools

#endif

#include <algorithm>
#include <memory>
#include <type_traits>
#include <utility>

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "Geometry/CommonTopologies/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ModuleFactory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/host.h"
#include "RecoTracker/MkFit/interface/MkFitGeometry.h"
#include "RecoTracker/MkFitCore/interface/TrackerInfo.h"
#include "RecoTracker/MkSeederAlpaka/interface/MkSeederConfig.h"
#include "RecoTracker/MkSeederAlpaka/interface/MkSeederESData.h"
#include "RecoTracker/MkSeederAlpaka/interface/MkSeederPSMark.h"
#include "RecoTracker/MkSeederAlpaka/interface/MkSeederTableBuilder.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using namespace ::mkseeder;

  // The seeder's static tables (MkSeederESData) from its configuration (MkSeederConfig), the mkFit geometry (the
  // layers, the modules, the material) and the tracker geometry (the pixel modules' frames), and the triplet MLP's
  // weights from their data file. The chains, the binning, the shape bands, the gap map and the patterns' order:
  // buildMkSeederTables() (MkSeederTableBuilder.h).
  class MkSeederESProducer : public ESProducer {
  public:
    MkSeederESProducer(edm::ParameterSet const &iConfig)
        : ESProducer(iConfig),
          tripletMlpWeights_{
              readMkSeederTripletMlpWeights(iConfig.getParameter<edm::FileInPath>("tripletMlpWeights").fullPath())} {
      auto cc = setWhatProduced(this);
      mkFitGeometryToken_ = cc.consumes();
      seederConfigToken_ = cc.consumes(iConfig.getParameter<edm::ESInputTag>("config"));
      trackerGeometryToken_ = cc.consumes();
    }

    static void fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::ESInputTag>("config", edm::ESInputTag{"", "mkSeederConfig"})
          ->setComment("The seeder's configuration (MkSeederConfigESProducer)");
      desc.add<edm::FileInPath>("tripletMlpWeights",
                                edm::FileInPath("RecoTracker/MkFit/data/mkfit-phase2-seeder-triplet-mlp.json"))
          ->setComment("The weights of the triplet MLP (JSON)");
      descriptions.addWithDefaultLabel(desc);
    }

    std::unique_ptr<MkSeederESDataHost> produce(TrackerRecoGeometryRecord const &iRecord) {
      const MkFitGeometry &mkFitGeometry = iRecord.get(mkFitGeometryToken_);
      if (mkFitGeometry.isPhase1())
        throw cms::Exception("Configuration") << "MkSeederESProducer: the mkFit seeder exists for phase 2 only";
      const mkfit::TrackerInfo &trackerInfo = mkFitGeometry.trackerInfo();
      const MkSeederConfig &seederConfig = iRecord.get(seederConfigToken_);
      const TrackerGeometry &trackerGeometry = iRecord.get(trackerGeometryToken_);
      if (seederConfig.psMark.mode > 0 && !mkFitGeometry.topology())
        throw cms::Exception("Configuration") << "MkSeederESProducer: ps_mark needs the tracker topology";

      auto data = std::make_unique<MkSeederESDataHost>();
      auto hostTables = std::make_shared<MkSeederHostTables>();
      // a CPU backend: q bins 4x wider (barrel 8 cm, disc 4 cm): fewer, longer runs per window for the chain
      // finder's vector scans; the finder's tests decide which hits pass, so the same seeds in the same order
      const double qBinScale = std::is_same_v<Device, alpaka::DevCpu> ? 4.0 : 1.0;
      buildMkSeederTables(seederConfig, trackerDescription(trackerInfo), qBinScale, *data, *hostTables);
      data->hostTables = std::move(hostTables);
      fillModules(
          *data, trackerInfo, trackerGeometry, seederConfig.psMark.mode > 0 ? mkFitGeometry.topology() : nullptr);
      fillMaterial(*data, trackerInfo);
      auto weights = std::make_shared<MkSeederTripletMlpWeightsHostObject>(cms::alpakatools::host());
      weights->value() = tripletMlpWeights_;
      data->tripletMlpWeights = std::move(weights);
      return data;
    }

  private:
    // the layers' extents and the barrel modules' outlines (the gap map) of the mkFit geometry
    static MkSeederTrackerDescription trackerDescription(const mkfit::TrackerInfo &trackerInfo) {
      MkSeederTrackerDescription tracker;
      const int nLayers = trackerInfo.n_layers();
      tracker.layers.resize(nLayers);
      tracker.modules.resize(nLayers);
      for (int l = 0; l < nLayers; ++l) {
        const mkfit::LayerInfo &layerInfo = trackerInfo[l];
        MkSeederTrackerLayer &layer = tracker.layers[l];
        layer.isPixel = layerInfo.is_pixel(), layer.isBarrel = layerInfo.is_barrel();
        layer.rIn = layerInfo.rin(), layer.rOut = layerInfo.rout();
        layer.zMin = layerInfo.zmin(), layer.zMax = layerInfo.zmax();
        layer.nModules = layerInfo.n_modules();
        if (!layerInfo.is_barrel())
          continue;
        for (int m = 0; m < layerInfo.n_modules(); ++m) {
          const mkfit::ModuleInfo &moduleInfo = layerInfo.module_info(m);
          const mkfit::ModuleShape &shape = layerInfo.module_shape(moduleInfo.shapeid);
          MkSeederGapMapModule module;
          for (int d = 0; d < 3; ++d) {
            module.position[d] = moduleInfo.pos[d];
            module.zDirection[d] = moduleInfo.zdir[d];
            module.xDirection[d] = moduleInfo.xdir[d];
          }
          // the z component of the module's y direction = zdir x xdir (not mkfit::ModuleInfo::calc_ydir(), whose third
          // component is wrong in CMSSW_20_1_0_pre5)
          module.yDirectionZ = moduleInfo.zdir[0] * moduleInfo.xdir[1] - moduleInfo.zdir[1] * moduleInfo.xdir[0];
          module.halfWidth = shape.dx1, module.halfLength = shape.dy;
          module.isRectangle = shape.is_rect();
          tracker.modules[l].push_back(module);
        }
      }
      return tracker;
    }

    // the modules; with the tracker topology (ps_mark) the OT1-3 P modules' S partners and frames
    static void fillModules(MkSeederESDataHost &data,
                            const mkfit::TrackerInfo &trackerInfo,
                            const TrackerGeometry &trackerGeometry,
                            const TrackerTopology *psTopology) {
      int nModules = 0;
      for (int l = 0; l < trackerInfo.n_layers(); ++l)
        nModules += trackerInfo[l].n_modules();
      auto collection =
          std::make_shared<MkSeederModulesHostCollection>(cms::alpakatools::host(), std::max(nModules, 1));
      auto view = collection->view();
      int m = 0;
      for (int l = 0; l < trackerInfo.n_layers(); ++l) {
        const mkfit::LayerInfo &layerInfo = trackerInfo[l];
        for (int k = 0; k < layerInfo.n_modules(); ++k, ++m) {
          const mkfit::ModuleInfo &moduleInfo = layerInfo.module_info(k);
          auto row = view[m];
          for (int d = 0; d < 3; ++d) {
            row.position()(d) = moduleInfo.pos[d];
            row.zDirection()(d) = moduleInfo.zdir[d];
            row.xDirection()(d) = moduleInfo.xdir[d];
          }
          row.isPixel() = false;
          row.rotation() = Eigen::Matrix3f::Zero();
          row.pitchX() = 0, row.pitchY() = 0, row.thickness() = 0;
          row.psPartner() = -1;
          row.psFrame() = Eigen::Matrix3f::Zero();
          const GeomDet *det = trackerGeometry.idToDet(DetId(moduleInfo.detid));
          if (const auto *pixelDet = dynamic_cast<const PixelGeomDetUnit *>(det)) {
            const auto &rotation = pixelDet->surface().rotation();
            row.isPixel() = true;
            row.rotation()(0, 0) = rotation.xx(), row.rotation()(0, 1) = rotation.xy(),
                              row.rotation()(0, 2) = rotation.xz();
            row.rotation()(1, 0) = rotation.yx(), row.rotation()(1, 1) = rotation.yy(),
                              row.rotation()(1, 2) = rotation.yz();
            row.rotation()(2, 0) = rotation.zx(), row.rotation()(2, 1) = rotation.zy(),
                              row.rotation()(2, 2) = rotation.zz();
            const auto pitch = pixelDet->specificTopology().pitch();
            row.pitchX() = pitch.first, row.pitchY() = pitch.second;
            row.thickness() = pixelDet->surface().bounds().thickness();
          }
        }
      }
      if (psTopology) {
        const PSMarkGeometry ps = buildPSMarkGeometry(trackerInfo, *psTopology);
        for (const PSMarkLayer &layer : ps.layers) {
          if (layer.pLayer < 0)
            continue;
          int first = 0;
          for (int l = 0; l < layer.pLayer; ++l)
            first += trackerInfo[l].n_modules();
          for (size_t k = 0; k < layer.partner.size(); ++k) {
            auto row = view[first + k];
            row.psPartner() = layer.partner[k];
            for (int a = 0; a < 3; ++a)
              for (int d = 0; d < 3; ++d)
                row.psFrame()(a, d) = layer.frame[k][3 * a + d];
          }
        }
      }
      data.modules = std::move(collection);
    }

    static void fillMaterial(MkSeederESDataHost &data, const mkfit::TrackerInfo &trackerInfo) {
      const int nZ = trackerInfo.mat_nbins_z(), nR = trackerInfo.mat_nbins_r();
      auto collection =
          std::make_shared<MkSeederMaterialHostCollection>(cms::alpakatools::host(), std::max(nZ * nR, 1));
      auto view = collection->view();
      for (int z = 0; z < nZ; ++z)
        for (int r = 0; r < nR; ++r)
          view[z * nR + r].bbxi() = trackerInfo.material_bbxi(z, r),
                        view[z * nR + r].radiationLength() = trackerInfo.material_radl(z, r);
      view.nBinsZ() = nZ, view.nBinsR() = nR;
      view.binsPerCmZ() = (float)trackerInfo.mat_nbins_z() / trackerInfo.mat_range_z();
      view.binsPerCmR() = (float)trackerInfo.mat_nbins_r() / trackerInfo.mat_range_r();
      data.material = std::move(collection);
    }

    edm::ESGetToken<MkFitGeometry, TrackerRecoGeometryRecord> mkFitGeometryToken_;
    edm::ESGetToken<MkSeederConfig, TrackerRecoGeometryRecord> seederConfigToken_;
    edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> trackerGeometryToken_;
    const MkSeederTripletMlpWeights tripletMlpWeights_;
  };  // class MkSeederESProducer

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_EVENTSETUP_ALPAKA_MODULE(MkSeederESProducer);

#include <algorithm>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <alpaka/alpaka.hpp>

#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Phase2TrackerCluster/interface/Phase2TrackerCluster1D.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/TrackingRecHitSoA/interface/alpaka/Phase2OTRecHitsSoACollection.h"
#include "DataFormats/TrackingRecHitSoA/interface/alpaka/TrackingRecHitsSoACollection.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/global/EDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

#include "Phase2OTCAHitsKernels.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  namespace {
    // The OT modules of the CA: the P sensors of the PS modules in the OT barrel, in GeomDetUnit order
    struct CAModules {
      std::vector<int32_t> moduleByDetIndex;  // CA module of each GeomDetUnit index, -1 if not a CA module
      uint32_t nModules = 0;
      uint16_t nPixelModules = 0;  // the CA module i has detectorIndex nPixelModules + i
    };
  }  // namespace

  // The OT hits of the CA (CAHitNtupletAlpakaPhase2OT) from the device OT rechits, with the module starts the
  // pixel-track converter reads on the host
  class Phase2OTCAHitsAlpaka : public global::EDProducer<edm::RunCache<CAModules>> {
  public:
    explicit Phase2OTCAHitsAlpaka(edm::ParameterSet const& iConfig)
        : EDProducer<edm::RunCache<CAModules>>(iConfig),
          recHitsToken_{consumes(iConfig.getParameter<edm::InputTag>("otRecHitsSoA"))},
          clustersToken_{consumes(iConfig.getParameter<edm::InputTag>("otClusters"))},
          pixelHitsToken_{consumes(iConfig.getParameter<edm::InputTag>("pixelRecHitSoASource"))},
          beamSpotToken_{consumes(iConfig.getParameter<edm::InputTag>("beamSpot"))},
          geometryToken_{esConsumes()},
          geometryRunToken_{esConsumes<edm::Transition::BeginRun>()},
          hitsToken_{produces()},
          moduleStartToken_{produces()} {}

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::InputTag>("otRecHitsSoA", edm::InputTag("hltSiPhase2RecHitsSoA"))
          ->setComment("OT rechits on the device (Phase2TrackerRecHitsAlpaka)");
      desc.add<edm::InputTag>("otClusters", edm::InputTag("hltSiPhase2Clusters"))
          ->setComment("the OT clusters of the rechits");
      desc.add<edm::InputTag>("pixelRecHitSoASource", edm::InputTag("hltPhase2SiPixelRecHitsSoA"))
          ->setComment("pixel rechits on the device: the OT hits follow them in the CA hit indices");
      desc.add<edm::InputTag>("beamSpot", edm::InputTag("hltOnlineBeamSpot"));
      descriptions.addWithDefaultLabel(desc);
    }

    std::shared_ptr<CAModules> globalBeginRun(edm::Run const&, edm::EventSetup const& iSetup) const override {
      auto const& geometry = iSetup.getData(geometryRunToken_);
      auto modules = std::make_shared<CAModules>();
      modules->moduleByDetIndex.assign(geometry.detUnits().size(), -1);
      for (auto const* detUnit : geometry.detUnits()) {
        const DetId detId = detUnit->geographicalId();
        if (detId.subdetId() == PixelSubdetector::PixelBarrel || detId.subdetId() == PixelSubdetector::PixelEndcap)
          ++modules->nPixelModules;
        if (geometry.getDetectorType(detId) == TrackerGeometry::ModuleType::Ph2PSP &&
            detId.subdetId() == StripSubdetector::TOB)
          modules->moduleByDetIndex[detUnit->index()] = modules->nModules++;
      }
      return modules;
    }

    void globalEndRun(edm::Run const&, edm::EventSetup const&) const override {}

    void produce(edm::StreamID, device::Event& iEvent, device::EventSetup const& iSetup) const override {
      using ::phase2OTCAHits::ModuleHits;

      edm::Event const& event = iEvent;
      CAModules const& caModules = *runCache(event.getRun().index());
      auto const& clusters = iEvent.get(clustersToken_);
      auto const& geometry = iSetup.getData(geometryToken_);
      auto const& beamSpot = iEvent.get(beamSpotToken_);
      const uint32_t nPixelHits = iEvent.get(pixelHitsToken_).nHits();
      auto& queue = iEvent.queue();

      // rechit and CA hit ranges of the CA modules, in CA module order
      auto modulesHost = cms::alpakatools::make_host_buffer<ModuleHits[]>(queue, caModules.nModules);
      std::fill_n(modulesHost.data(), caModules.nModules, ModuleHits{});
      auto const* firstCluster = clusters.data().data();
      for (auto const& detSet : clusters) {
        if (detSet.empty())
          continue;
        const int32_t module = caModules.moduleByDetIndex[geometry.idToDetUnit(DetId(detSet.detId()))->index()];
        if (module >= 0)
          modulesHost[module] = {
              static_cast<uint32_t>(&*detSet.begin() - firstCluster), 0, static_cast<uint32_t>(detSet.size())};
      }
      std::vector<uint32_t> moduleStart(caModules.nModules + 1);
      uint32_t nHits = 0;
      for (uint32_t module = 0; module < caModules.nModules; ++module) {
        modulesHost[module].firstHit = nHits;
        moduleStart[module] = nPixelHits + nHits;
        nHits += modulesHost[module].nHits;
      }
      moduleStart[caModules.nModules] = nPixelHits + nHits;

      auto modulesDevice = cms::alpakatools::make_device_buffer<ModuleHits[]>(queue, caModules.nModules);
      alpaka::memcpy(queue, modulesDevice, modulesHost);
      reco::TrackingRecHitsSoACollection hits(queue, nHits, caModules.nModules);
      phase2OTCAHits::makeCAHits(queue,
                                 iEvent.get(recHitsToken_).const_view(),
                                 modulesDevice.data(),
                                 caModules.nModules,
                                 {beamSpot.x0(), beamSpot.y0(), beamSpot.z0()},
                                 nPixelHits,
                                 caModules.nPixelModules,
                                 hits.view().trackingHits(),
                                 hits.view().hitModules());
      iEvent.emplace(hitsToken_, std::move(hits));
      iEvent.emplace(moduleStartToken_, std::move(moduleStart));
    }

  private:
    const device::EDGetToken<reco::Phase2OTRecHitsSoACollection> recHitsToken_;
    const edm::EDGetTokenT<Phase2TrackerCluster1DCollectionNew> clustersToken_;
    const device::EDGetToken<reco::TrackingRecHitsSoACollection> pixelHitsToken_;
    const edm::EDGetTokenT<::reco::BeamSpot> beamSpotToken_;
    const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geometryToken_;
    const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geometryRunToken_;
    const device::EDPutToken<reco::TrackingRecHitsSoACollection> hitsToken_;
    const edm::EDPutTokenT<std::vector<uint32_t>> moduleStartToken_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_ALPAKA_MODULE(Phase2OTCAHitsAlpaka);

#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include <alpaka/alpaka.hpp>

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/Phase2TrackerCluster/interface/Phase2TrackerCluster1D.h"
#include "DataFormats/TrackingRecHitSoA/interface/alpaka/Phase2OTRecHitsSoACollection.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/ESInputTag.h"
#include "FWCore/Utilities/interface/Exception.h"
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
#include "RecoLocalTracker/Phase2TrackerRecHits/interface/Phase2OTModuleParamsHost.h"
#include "RecoLocalTracker/Phase2TrackerRecHits/interface/alpaka/Phase2OTModuleParamsCollection.h"
#include "RecoLocalTracker/Records/interface/TkPhase2OTCPERecord.h"

#include "Phase2OTCAModules.h"
#include "Phase2OTRecHitsKernels.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  namespace {
    struct CAModules {
      std::vector<int32_t> caModuleByDetUnit;  // CA module of each GeomDetUnit::index(), -1 if not a CA module
      uint32_t nModules = 0;
    };
  }  // namespace

  // The Phase-2 OT rechits of Phase2TrackerRecHits made on the device from the OT clusters, with their global position,
  // and on the host the rechit row of the first cluster of each CA module
  class Phase2OTRecHitsAlpaka : public global::EDProducer<edm::RunCache<CAModules>> {
  public:
    explicit Phase2OTRecHitsAlpaka(edm::ParameterSet const& iConfig)
        : EDProducer<edm::RunCache<CAModules>>(iConfig),
          clustersToken_{consumes(iConfig.getParameter<edm::InputTag>("otClusters"))},
          moduleParamsToken_{esConsumes(iConfig.getParameter<edm::ESInputTag>("moduleParams"))},
          moduleParamsHostToken_{esConsumes(iConfig.getParameter<edm::ESInputTag>("moduleParams"))},
          geometryToken_{esConsumes()},
          geometryRunToken_{esConsumes<edm::Transition::BeginRun>()},
          recHitsToken_{produces()},
          caRecHitStartToken_{produces()} {}

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::InputTag>("otClusters", edm::InputTag("siPhase2Clusters"));
      desc.add<edm::ESInputTag>("moduleParams", edm::ESInputTag())
          ->setComment(
              "the OT module table of Phase2OTModuleParamsESProducerAlpaka, indexed by the rechits' moduleIndex");
      descriptions.setComment(
          "One rechit per OT cluster, its row being the cluster's index in the cluster collection. Also a vector "
          "with the row of the first cluster of each CA module: the CA OT hit caIndex of CA module m is the row "
          "vector[m] + caIndex - moduleStart[m], with moduleStart of Phase2OTCAHitsAlpaka.");
      descriptions.addWithDefaultLabel(desc);
    }

    std::shared_ptr<CAModules> globalBeginRun(edm::Run const&, edm::EventSetup const& iSetup) const override {
      auto const& geometry = iSetup.getData(geometryRunToken_);
      auto const modules = phase2OTCAHits::caModules(geometry);
      auto caModules = std::make_shared<CAModules>();
      caModules->caModuleByDetUnit.assign(geometry.detUnits().size(), -1);
      caModules->nModules = modules.size();
      for (uint32_t module = 0; module < modules.size(); ++module)
        caModules->caModuleByDetUnit[modules[module]->index()] = module;
      return caModules;
    }

    void globalEndRun(edm::Run const&, edm::EventSetup const&) const override {}

    void produce(edm::StreamID, device::Event& iEvent, device::EventSetup const& iSetup) const override {
      using phase2OTHits::Cluster;
      using ModuleIndex = decltype(Cluster::module);

      CAModules const& caModules = *runCache(static_cast<edm::Event const&>(iEvent).getRun().index());
      auto const& clusters = iEvent.get(clustersToken_);
      auto const& geometry = iSetup.getData(geometryToken_);
      auto const moduleParams = iSetup.getData(moduleParamsHostToken_).const_view();
      const uint32_t firstModuleIndex = moduleParams.firstModuleIndex();
      const uint32_t nModules = moduleParams.metadata().size();
      if (nModules > std::numeric_limits<ModuleIndex>::max() + 1ul)
        throw cms::Exception("BadGeometry")
            << "Phase2OTRecHitsAlpaka: " << nModules << " OT modules do not fit the module index of the rechits";
      auto& queue = iEvent.queue();

      // a cluster's row is its index in the cluster collection; caRecHitStart is set only for CA modules with clusters
      const uint32_t nClusters = clusters.dataSize();
      std::vector<uint32_t> caRecHitStart(caModules.nModules, nClusters);
      auto clustersHost = cms::alpakatools::make_host_buffer<Cluster[]>(queue, nClusters);
      for (auto const& detSet : clusters) {
        const int32_t detUnitIndex = geometry.idToDetUnit(DetId(detSet.detId()))->index();
        // wraps around for a module before the OT
        const uint32_t module = static_cast<uint32_t>(detUnitIndex) - firstModuleIndex;
        if (module >= nModules)
          throw cms::Exception("LogicError")
              << "Phase2OTRecHitsAlpaka: the clusters of module " << detSet.detId() << " are not in the OT";
        uint32_t row = detSet.offset();
        if (const int32_t caModule = caModules.caModuleByDetUnit[detUnitIndex]; caModule >= 0)
          caRecHitStart[caModule] = row;
        for (auto const& cluster : detSet)
          clustersHost[row++] = {static_cast<uint16_t>(cluster.firstStrip()),
                                 static_cast<uint16_t>(cluster.column()),
                                 cluster.size(),
                                 static_cast<ModuleIndex>(module)};
      }

      reco::Phase2OTRecHitsSoACollection recHits(queue, nClusters);
      if (nClusters > 0) {
        auto clustersDevice = cms::alpakatools::make_device_buffer<Cluster[]>(queue, nClusters);
        alpaka::memcpy(queue, clustersDevice, clustersHost);
        phase2OTRecHits::makeRecHits(
            queue, iSetup.getData(moduleParamsToken_).const_view(), clustersDevice.data(), recHits.view());
      }
      iEvent.emplace(recHitsToken_, std::move(recHits));
      iEvent.emplace(caRecHitStartToken_, std::move(caRecHitStart));
    }

  private:
    const edm::EDGetTokenT<Phase2TrackerCluster1DCollectionNew> clustersToken_;
    const device::ESGetToken<Phase2OTModuleParamsCollection, TkPhase2OTCPERecord> moduleParamsToken_;
    const edm::ESGetToken<Phase2OTModuleParamsHost, TkPhase2OTCPERecord> moduleParamsHostToken_;
    const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geometryToken_;
    const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geometryRunToken_;
    const device::EDPutToken<reco::Phase2OTRecHitsSoACollection> recHitsToken_;
    const edm::EDPutTokenT<std::vector<uint32_t>> caRecHitStartToken_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_ALPAKA_MODULE(Phase2OTRecHitsAlpaka);

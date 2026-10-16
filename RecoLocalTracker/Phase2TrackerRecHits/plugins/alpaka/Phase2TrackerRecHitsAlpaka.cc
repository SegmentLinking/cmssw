#include <cstdint>
#include <utility>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Phase2TrackerCluster/interface/Phase2TrackerCluster1D.h"
#include "DataFormats/TrackingRecHitSoA/interface/alpaka/Phase2OTRecHitsSoACollection.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/ESInputTag.h"
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
#include "RecoLocalTracker/Phase2TrackerRecHits/interface/alpaka/Phase2StripCPEParamsCollection.h"
#include "RecoLocalTracker/Records/interface/TkPhase2OTCPERecord.h"

#include "Phase2TrackerRecHitsKernels.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // The Phase-2 OT rechits of Phase2TrackerRecHits (Phase2StripCPE) made on the device from the host clusters, with
  // their global positions: one row per cluster, in the order of the cluster collection
  class Phase2TrackerRecHitsAlpaka : public global::EDProducer<> {
  public:
    explicit Phase2TrackerRecHitsAlpaka(edm::ParameterSet const& iConfig)
        : EDProducer<>(iConfig),
          clustersToken_{consumes(iConfig.getParameter<edm::InputTag>("src"))},
          geometryToken_{esConsumes()},
          cpeParamsToken_{esConsumes(iConfig.getParameter<edm::ESInputTag>("cpeParams"))},
          recHitsToken_{produces()} {}

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::InputTag>("src", edm::InputTag("siPhase2Clusters"))->setComment("Phase-2 OT clusters");
      desc.add<edm::ESInputTag>("cpeParams", edm::ESInputTag())
          ->setComment("the Phase2StripCPE parameters on the device (Phase2StripCPEParamsESProducerAlpaka)");
      descriptions.addWithDefaultLabel(desc);
    }

    void produce(edm::StreamID, device::Event& iEvent, device::EventSetup const& iSetup) const override {
      using ::phase2TrackerRecHits::ClusterData;
      using ::phase2TrackerRecHits::ModuleClusters;

      auto const& clusters = iEvent.get(clustersToken_);
      auto const& geometry = iSetup.getData(geometryToken_);
      auto& queue = iEvent.queue();

      const uint32_t nClusters = clusters.dataSize();
      reco::Phase2OTRecHitsSoACollection recHits(queue, nClusters);
      if (nClusters > 0) {
        auto modulesHost = cms::alpakatools::make_host_buffer<ModuleClusters[]>(queue, clusters.size());
        auto clustersHost = cms::alpakatools::make_host_buffer<ClusterData[]>(queue, nClusters);
        auto const* firstCluster = clusters.data().data();
        uint32_t nModules = 0;
        for (auto const& detSet : clusters) {
          if (detSet.empty())
            continue;
          const uint32_t first = &*detSet.begin() - firstCluster;
          modulesHost[nModules++] = {first,
                                     static_cast<uint32_t>(detSet.size()),
                                     static_cast<uint32_t>(geometry.idToDetUnit(DetId(detSet.detId()))->index())};
          uint32_t index = first;
          for (auto const& cluster : detSet) {
            clustersHost[index++] = {
                static_cast<uint16_t>(cluster.firstStrip()), static_cast<uint16_t>(cluster.column()), cluster.size()};
          }
        }
        auto modulesDevice = cms::alpakatools::make_device_buffer<ModuleClusters[]>(queue, nModules);
        auto clustersDevice = cms::alpakatools::make_device_buffer<ClusterData[]>(queue, nClusters);
        alpaka::memcpy(queue, modulesDevice, cms::alpakatools::make_host_view(modulesHost.data(), nModules));
        alpaka::memcpy(queue, clustersDevice, clustersHost);
        phase2TrackerRecHits::makeRecHits(queue,
                                          iSetup.getData(cpeParamsToken_).const_view(),
                                          modulesDevice.data(),
                                          nModules,
                                          clustersDevice.data(),
                                          recHits.view());
      }
      iEvent.emplace(recHitsToken_, std::move(recHits));
    }

  private:
    const edm::EDGetTokenT<Phase2TrackerCluster1DCollectionNew> clustersToken_;
    const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geometryToken_;
    const device::ESGetToken<Phase2StripCPEParamsCollection, TkPhase2OTCPERecord> cpeParamsToken_;
    const device::EDPutToken<reco::Phase2OTRecHitsSoACollection> recHitsToken_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_ALPAKA_MODULE(Phase2TrackerRecHitsAlpaka);

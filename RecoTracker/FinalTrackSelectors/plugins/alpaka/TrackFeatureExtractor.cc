#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/EDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "RecoTracker/FinalTrackSelectors/interface/alpaka/TrackFeaturesDeviceCollection.h"
#include "RecoTracker/FinalTrackSelectors/interface/TrackFeaturesSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class TrackFeatureExtractor : public stream::EDProducer<> {
  public:
    TrackFeatureExtractor(const edm::ParameterSet& iConfig)
        : EDProducer<>(iConfig),
          tracks_token_(consumes<edm::InEvent>(iConfig.getParameter<edm::InputTag>("src"))),
          beamspot_token_(consumes<edm::InEvent>(iConfig.getParameter<edm::InputTag>("beamSpot"))),
          vertices_token_(consumes<edm::InEvent>(iConfig.getParameter<edm::InputTag>("vertices"))),
          features_token_{produces()} {}

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::InputTag>("src", edm::InputTag("hltInitialStepTracks"));
      desc.add<edm::InputTag>("beamSpot", edm::InputTag("hltOnlineBeamSpot"));
      desc.add<edm::InputTag>("vertices", edm::InputTag("hltPhase2PixelVertices"));
      descriptions.addWithDefaultLabel(desc);
    }

    void produce(device::Event& iEvent, const device::EventSetup& iSetup) override {
      auto const& tracks = iEvent.get(tracks_token_);
      auto const& beamspot = iEvent.get(beamspot_token_);
      auto const& vertices = iEvent.get(vertices_token_);

      const auto nTracks = tracks.size();

      // Create HOST collection first, fill it, then copy to device
      PortableHostCollection<TrackFeaturesSoA> features_host(nTracks);
      
      auto features_view = features_host.view();

      const reco::Vertex* bestVertex = nullptr;
      if (!vertices.empty()) {
        bestVertex = &vertices[0];
      }

      for (size_t i = 0; i < nTracks; ++i) {
        const auto& track = tracks[i];
        const auto& innerMom = track.innerMomentum();
        const auto& outerMom = track.outerMomentum();

        features_view[i].pt() = track.pt();
        features_view[i].innerMomentumX() = innerMom.x();
        features_view[i].innerMomentumY() = innerMom.y();
        features_view[i].innerMomentumZ() = innerMom.z();
        features_view[i].innerMomentumRho() = innerMom.Rho();
        features_view[i].outerMomentumX() = outerMom.x();
        features_view[i].outerMomentumY() = outerMom.y();
        features_view[i].outerMomentumZ() = outerMom.z();
        features_view[i].outerMomentumRho() = outerMom.Rho();
        features_view[i].ptError() = track.ptError();

        if (bestVertex) {
          features_view[i].dxyBestVertex() = track.dxy(bestVertex->position());
          features_view[i].dzBestVertex() = track.dz(bestVertex->position());
        } else {
          features_view[i].dxyBestVertex() = 0.0f;
          features_view[i].dzBestVertex() = 0.0f;
        }

        features_view[i].dxyBeamSpot() = track.dxy(beamspot.position());
        features_view[i].dzBeamSpot() = track.dz(beamspot.position());
        features_view[i].dxyError() = track.dxyError();
        features_view[i].dzError() = track.dzError();

        features_view[i].normalizedChi2() = track.normalizedChi2();
        features_view[i].eta() = track.eta();
        features_view[i].phi() = track.phi();
        features_view[i].etaError() = track.etaError();
        features_view[i].phiError() = track.phiError();
        features_view[i].ndof() = track.ndof();

        const auto& hitPattern = track.hitPattern();
        features_view[i].lostInnerHits() = hitPattern.numberOfLostTrackerHits(reco::HitPattern::MISSING_INNER_HITS);
        features_view[i].lostOuterHits() = hitPattern.numberOfLostTrackerHits(reco::HitPattern::MISSING_OUTER_HITS);
        features_view[i].layersOffInner() = hitPattern.trackerLayersTotallyOffOrBad(reco::HitPattern::MISSING_INNER_HITS);
        features_view[i].layersOffOuter() = hitPattern.trackerLayersTotallyOffOrBad(reco::HitPattern::MISSING_OUTER_HITS);
        features_view[i].layersWithoutMeas() = hitPattern.trackerLayersWithoutMeasurement(reco::HitPattern::TRACK_HITS);
        features_view[i].validPixelHits() = hitPattern.numberOfValidPixelHits();
        features_view[i].validStripHits() = hitPattern.numberOfValidStripHits();
      }

      // Create device collection and copy from host
      TrackFeaturesDeviceCollection features_device(nTracks, iEvent.queue());
      alpaka::memcpy(iEvent.queue(), features_device.buffer(), features_host.const_buffer());
      
      iEvent.emplace(features_token_, std::move(features_device));
    }

  private:
    const edm::EDGetTokenT<reco::TrackCollection> tracks_token_;
    const edm::EDGetTokenT<reco::BeamSpot> beamspot_token_;
    const edm::EDGetTokenT<reco::VertexCollection> vertices_token_;
    const device::EDPutToken<TrackFeaturesDeviceCollection> features_token_;
  };

}

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(TrackFeatureExtractor);

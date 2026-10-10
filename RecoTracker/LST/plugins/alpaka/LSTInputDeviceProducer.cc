#include <Eigen/Core>  // before the SoA headers with Eigen columns

#include <cstdint>
#include <optional>
#include <vector>

#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Common/interface/OwnVector.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackSoA/interface/alpaka/TracksSoACollection.h"
#include "DataFormats/TrackerRecHit2D/interface/Phase2TrackerRecHit1D.h"
#include "DataFormats/TrackingRecHitSoA/interface/alpaka/Phase2OTRecHitsSoACollection.h"
#include "DataFormats/TrackingRecHitSoA/interface/alpaka/TrackingRecHitsSoACollection.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/SynchronizingEDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/ParametrizedEngine/interface/TkBfield.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "RecoTracker/LST/interface/alpaka/LSTSeedFitModulesCollection.h"
#include "RecoTracker/LSTCore/interface/LSTOTHits.h"
#include "RecoTracker/LSTCore/interface/alpaka/LSTInputDeviceCollection.h"
#include "RecoTracker/MkFitCore/interface/Config.h"
#include "RecoTracker/MkFitCore/interface/Track.h"
#include "TrackingTools/TrajectoryParametrization/interface/CurvilinearTrajectoryError.h"
#include "TrackingTools/TrajectoryParametrization/interface/GlobalTrajectoryParameters.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"

#include "LSTInputDeviceKernels.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // LST's input made on the device: each pixel track refitted as the pixel-seed creator does, its pLS as LSTInputProducer
  // computes it from that seed; the seeds for LSTOutputConverter are made on the host from the fitted states.
  class LSTInputDeviceProducer : public stream::SynchronizingEDProducer<> {
  public:
    explicit LSTInputDeviceProducer(edm::ParameterSet const& iConfig)
        : SynchronizingEDProducer<>(iConfig),
          ptCut_(iConfig.getParameter<double>("ptCut")),
          originRadius_(iConfig.getParameter<double>("originRadius")),
          originHalfLength_(iConfig.getParameter<double>("originHalfLength")),
          pixelTracksSoAToken_(consumes(iConfig.getParameter<edm::InputTag>("pixelTracksSoA"))),
          pixelTracksToken_(consumes(iConfig.getParameter<edm::InputTag>("pixelTracks"))),
          pixelTrackIndexToken_(consumes(iConfig.getParameter<edm::InputTag>("pixelTracks"))),
          pixelHitsToken_(consumes(iConfig.getParameter<edm::InputTag>("pixelRecHitsSoA"))),
          caModuleStartToken_(consumes(iConfig.getParameter<edm::InputTag>("caOTHits"))),
          caRecHitStartToken_(
              consumes(edm::InputTag(iConfig.getParameter<edm::InputTag>("caOTHits").label(), "recHitStart"))),
          otRecHitsSoAToken_(consumes(iConfig.getParameter<edm::InputTag>("otRecHitsSoA"))),
          otRecHitsToken_(consumes(iConfig.getParameter<edm::InputTag>("otRecHits"))),
          beamSpotToken_(consumes(iConfig.getParameter<edm::InputTag>("beamSpot"))),
          modulesToken_(esConsumes()),
          magneticFieldToken_(esConsumes()),
          lstInputPutToken_(produces()),
          lstOTHitsPutToken_(produces()),
          seedsPutToken_(produces()) {}

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<double>("ptCut", 0.8);
      desc.add<double>("originRadius", 0.2)->setComment("cm, the creator's region around the pixel-track vertex");
      desc.add<double>("originHalfLength", 0.2)->setComment("cm");
      desc.add<edm::InputTag>("pixelTracksSoA", edm::InputTag("hltPhase2PixelTrackTorchHighPuritySelector"));
      desc.add<edm::InputTag>("pixelTracks", edm::InputTag("hltPhase2PixelTracks"))
          ->setComment("the reco::Tracks of pixelTracksSoA and their index map; one seed per fitted track");
      desc.add<edm::InputTag>("pixelRecHitsSoA", edm::InputTag("hltPhase2SiPixelRecHitsSoA"));
      desc.add<edm::InputTag>("caOTHits", edm::InputTag("hltPhase2OtRecHitsSoA"))
          ->setComment("the CA's OT hits: their module starts and the first OT rechit of each module");
      desc.add<edm::InputTag>("otRecHitsSoA", edm::InputTag("hltSiPhase2RecHitsSoA"));
      desc.add<edm::InputTag>("otRecHits", edm::InputTag("hltSiPhase2RecHits"))
          ->setComment("the OT rechits of otRecHitsSoA, in the same order, for LSTOutputConverter");
      desc.add<edm::InputTag>("beamSpot", edm::InputTag("hltOnlineBeamSpot"));
      descriptions.addWithDefaultLabel(desc);
    }

    void acquire(device::Event const& iEvent, device::EventSetup const& iSetup) override {
      auto& queue = iEvent.queue();
      auto const& indexToEdm = iEvent.get(pixelTrackIndexToken_);
      const uint32_t nEdmTracks = iEvent.get(pixelTracksToken_).size();
      auto const& caModuleStart = iEvent.get(caModuleStartToken_);
      auto const& caRecHitStart = iEvent.get(caRecHitStartToken_);
      auto const& pixelHits = iEvent.get(pixelHitsToken_);
      auto const& beamSpot = iEvent.get(beamSpotToken_);
      nTracks_ = indexToEdm.size();

      ::lstInputDevice::Settings settings{static_cast<float>(beamSpot.x0()),
                                          static_cast<float>(beamSpot.y0()),
                                          static_cast<float>(beamSpot.z0()),
                                          static_cast<float>(beamSpot.dxdz()),
                                          static_cast<float>(beamSpot.dydz()),
                                          ptCut_,
                                          originRadius_,
                                          originHalfLength_,
                                          pixelHits.nHits(),
                                          static_cast<uint32_t>(caModuleStart.size() - 1),
                                          {}};
      // MkFitCore's field parametrisation, and the multiple scattering on 1/pT that MkFitGeometryESProducer enables
      settings.fit.env = mkfit::PropagationEnv{.mag_c1 = mkfit::Config::mag_c1,
                                               .mag_b0 = mkfit::Config::mag_b0,
                                               .mag_b1 = mkfit::Config::mag_b1,
                                               .mag_a = mkfit::Config::mag_a,
                                               .use_pt_mult_scat = true};
      const magfieldparam::TkBfield field(iSetup.getData(magneticFieldToken_).nominalValue() / 10.f);

      edmIndex_.emplace(cms::alpakatools::make_host_buffer<int32_t[]>(queue, nTracks_));
      for (uint32_t track = 0; track < nTracks_; ++track)
        edmIndex_->data()[track] = indexToEdm[track] < nEdmTracks ? static_cast<int32_t>(indexToEdm[track]) : -1;
      auto edmIndexDevice = cms::alpakatools::make_device_buffer<int32_t[]>(queue, nTracks_);
      alpaka::memcpy(queue, edmIndexDevice, *edmIndex_);
      auto caModuleStartDevice = cms::alpakatools::make_device_buffer<uint32_t[]>(queue, caModuleStart.size());
      alpaka::memcpy(
          queue, caModuleStartDevice, cms::alpakatools::make_host_view(caModuleStart.data(), caModuleStart.size()));
      auto caRecHitStartDevice = cms::alpakatools::make_device_buffer<uint32_t[]>(queue, caRecHitStart.size());
      alpaka::memcpy(
          queue, caRecHitStartDevice, cms::alpakatools::make_host_view(caRecHitStart.data(), caRecHitStart.size()));

      candidates_.emplace(cms::alpakatools::make_device_buffer<PixelSegmentCandidate[]>(queue, nTracks_));
      lstInputDevice::fitPixelTracks(queue,
                                     iEvent.get(pixelTracksSoAToken_).const_view(),
                                     nTracks_,
                                     edmIndexDevice.data(),
                                     pixelHits.const_view().trackingHits(),
                                     caModuleStartDevice.data(),
                                     caRecHitStartDevice.data(),
                                     iEvent.get(otRecHitsSoAToken_).const_view(),
                                     iSetup.getData(modulesToken_).const_view(),
                                     field.bcycl(),
                                     settings,
                                     candidates_->data());
      seedIndex_.emplace(cms::alpakatools::make_device_buffer<uint32_t[]>(queue, nTracks_));
      segmentIndex_.emplace(cms::alpakatools::make_device_buffer<uint32_t[]>(queue, nTracks_));
      hitOffset_.emplace(cms::alpakatools::make_device_buffer<uint32_t[]>(queue, nTracks_));
      auto totalsDevice = cms::alpakatools::make_device_buffer<lstInputDevice::Totals>(queue);
      lstInputDevice::indexPixelSegments(queue,
                                         candidates_->data(),
                                         nTracks_,
                                         seedIndex_->data(),
                                         segmentIndex_->data(),
                                         hitOffset_->data(),
                                         totalsDevice.data());
      totals_.emplace(cms::alpakatools::make_host_buffer<lstInputDevice::Totals>(queue));
      alpaka::memcpy(queue, *totals_, totalsDevice);
      candidatesHost_.emplace(cms::alpakatools::make_host_buffer<PixelSegmentCandidate[]>(queue, nTracks_));
      alpaka::memcpy(queue, *candidatesHost_, *candidates_);
    }

    void produce(device::Event& iEvent, device::EventSetup const& iSetup) override {
      auto& queue = iEvent.queue();
      auto const& totals = *totals_->data();
      auto const& otRecHitsSoA = iEvent.get(otRecHitsSoAToken_);
      const uint32_t nOTHits = otRecHitsSoA.const_view().metadata().size();

      lst::LSTInputDeviceCollection lstInput(
          queue, nOTHits + totals.nSegmentHits, totals.nSegments, totals.nSegmentHits);
      lstInputDevice::fillLSTInput(queue,
                                   lstInput.view(),
                                   otRecHitsSoA.const_view(),
                                   nOTHits,
                                   candidates_->data(),
                                   nTracks_,
                                   seedIndex_->data(),
                                   segmentIndex_->data(),
                                   hitOffset_->data());
      iEvent.emplace(lstInputPutToken_, std::move(lstInput));

      // OT rechit pointers in the order of the OT hits of the LST input
      auto const& otRecHits = iEvent.get(otRecHitsToken_);
      std::vector<TrackingRecHit const*> otHits;
      otHits.reserve(otRecHits.dataSize());
      for (auto const& detSet : otRecHits)
        for (auto const& hit : detSet)
          otHits.push_back(&hit);
      if (otHits.size() != nOTHits)
        throw cms::Exception("LSTInputDeviceProducer")
            << "OT rechits " << otHits.size() << " != device OT rechits " << nOTHits;
      iEvent.emplace(lstOTHitsPutToken_, lst::LSTOTHits{std::move(otHits)});

      // the seeds of the fitted pixel tracks in track order: their hits and the fitted state on the last hit
      auto const& pixelTracks = iEvent.get(pixelTracksToken_);
      auto const& magneticField = iSetup.getData(magneticFieldToken_);
      TrajectorySeedCollection seeds;
      seeds.reserve(totals.nSeeds);
      for (uint32_t track = 0; track < nTracks_; ++track) {
        auto const& candidate = candidatesHost_->data()[track];
        if (candidate.seedStatus != static_cast<int8_t>(seedFromConsecutiveHits::Status::ok))
          continue;
        auto const& pixelTrack = pixelTracks[edmIndex_->data()[track]];
        edm::OwnVector<TrackingRecHit> hits;
        for (auto const& hit : pixelTrack.recHits())
          hits.push_back(hit->clone());
        seeds.emplace_back(seedState(candidate.state, hits.back(), magneticField), hits, alongMomentum);
      }
      if (seeds.size() != totals.nSeeds)
        throw cms::Exception("LSTInputDeviceProducer")
            << "seeds " << seeds.size() << " != fitted pixel tracks " << totals.nSeeds;
      iEvent.emplace(seedsPutToken_, std::move(seeds));

      edmIndex_.reset();
      candidates_.reset();
      candidatesHost_.reset();
      seedIndex_.reset();
      segmentIndex_.reset();
      hitOffset_.reset();
      totals_.reset();
    }

  private:
    using PixelSegmentCandidate = ::lstInputDevice::PixelSegmentCandidate;

    // the fitted CCS state as the persistent state on the last hit, as MkFitOutputConverter converts mkFit states
    static PTrajectoryStateOnDet seedState(seedFromConsecutiveHits::State const& fitted,
                                           TrackingRecHit const& lastHit,
                                           MagneticField const& magneticField) {
      mkfit::TrackState state;
      state.charge = fitted.charge;
      for (int k = 0; k < 6; ++k)
        state.parameters[k] = fitted.parameters[k];
      state.errors.SetElements(fitted.errors, fitted.errors + 21, true, true);
      state.convertFromCCSToGlbCurvilinear();
      AlgebraicSymMatrix55 curvilinear;
      for (int row = 0; row < 5; ++row)
        for (int col = row; col < 5; ++col)
          curvilinear[row][col] = state.errors.At(row, col);
      auto const& parameters = state.parameters;
      const FreeTrajectoryState freeState(
          GlobalTrajectoryParameters(GlobalPoint(parameters[0], parameters[1], parameters[2]),
                                     GlobalVector(parameters[3], parameters[4], parameters[5]),
                                     state.charge,
                                     &magneticField),
          CurvilinearTrajectoryError(curvilinear));
      return trajectoryStateTransform::persistentState(TrajectoryStateOnSurface(freeState, lastHit.det()->surface()),
                                                       lastHit.geographicalId().rawId());
    }

    const float ptCut_;
    const float originRadius_;
    const float originHalfLength_;
    const device::EDGetToken<reco::TracksSoACollection> pixelTracksSoAToken_;
    const edm::EDGetTokenT<::reco::TrackCollection> pixelTracksToken_;
    const edm::EDGetTokenT<std::vector<uint32_t>> pixelTrackIndexToken_;
    const device::EDGetToken<reco::TrackingRecHitsSoACollection> pixelHitsToken_;
    const edm::EDGetTokenT<std::vector<uint32_t>> caModuleStartToken_;
    const edm::EDGetTokenT<std::vector<uint32_t>> caRecHitStartToken_;
    const device::EDGetToken<reco::Phase2OTRecHitsSoACollection> otRecHitsSoAToken_;
    const edm::EDGetTokenT<Phase2TrackerRecHit1DCollectionNew> otRecHitsToken_;
    const edm::EDGetTokenT<::reco::BeamSpot> beamSpotToken_;
    const device::ESGetToken<lst::LSTSeedFitModulesCollection, TrackerDigiGeometryRecord> modulesToken_;
    const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> magneticFieldToken_;
    const device::EDPutToken<lst::LSTInputDeviceCollection> lstInputPutToken_;
    const edm::EDPutTokenT<lst::LSTOTHits> lstOTHitsPutToken_;
    const edm::EDPutTokenT<TrajectorySeedCollection> seedsPutToken_;

    // per-event buffers from acquire to produce
    uint32_t nTracks_ = 0;
    std::optional<cms::alpakatools::host_buffer<int32_t[]>> edmIndex_;
    std::optional<cms::alpakatools::device_buffer<Device, PixelSegmentCandidate[]>> candidates_;
    std::optional<cms::alpakatools::host_buffer<PixelSegmentCandidate[]>> candidatesHost_;
    std::optional<cms::alpakatools::device_buffer<Device, uint32_t[]>> seedIndex_;
    std::optional<cms::alpakatools::device_buffer<Device, uint32_t[]>> segmentIndex_;
    std::optional<cms::alpakatools::device_buffer<Device, uint32_t[]>> hitOffset_;
    std::optional<cms::alpakatools::host_buffer<lstInputDevice::Totals>> totals_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_ALPAKA_MODULE(LSTInputDeviceProducer);

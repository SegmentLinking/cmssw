#include <Eigen/Core>  // before the SoA headers with Eigen columns

#include <algorithm>
#include <cmath>
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
#include "FWCore/Utilities/interface/EDMException.h"
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
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "RecoTracker/LST/interface/BLFCurvilinearJacobian.h"
#include "RecoTracker/LST/interface/alpaka/LSTSeedFitModulesCollection.h"
#include "RecoTracker/LSTCore/interface/LSTOTHits.h"
#include "RecoTracker/LSTCore/interface/SeedStateSoA.h"
#include "RecoTracker/LSTCore/interface/alpaka/LSTInputDeviceCollection.h"
#include "RecoTracker/LSTCore/interface/alpaka/PixelSeedStatesDeviceCollection.h"
#include "TrackingTools/TrajectoryParametrization/interface/CurvilinearTrajectoryError.h"
#include "TrackingTools/TrajectoryParametrization/interface/GlobalTrajectoryParameters.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"

#include "LSTInputDeviceKernels.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // LST's input made on the device. Each pixel track is fitted with the Broken Line Fit, its state taken at its last
  // hit; its pLS is computed from that state as LSTInputProducer computes it from a seed. The states of the fitted
  // tracks go to the seed-state product, and the seeds for LSTOutputConverter are made on the host from them.
  class LSTInputDeviceProducer : public stream::SynchronizingEDProducer<> {
  public:
    explicit LSTInputDeviceProducer(edm::ParameterSet const& iConfig)
        : SynchronizingEDProducer<>(iConfig),
          ptCut_(iConfig.getParameter<double>("ptCut")),
          maxITHitsToDropOTHitsPurePLS_(iConfig.getParameter<int>("maxITHitsToDropOTHitsPurePLS")),
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
          seedStatesPutToken_(produces()),
          lstOTHitsPutToken_(produces()),
          seedsPutToken_(produces()) {
      // the inner-tracker-only fit is compiled for kITFitHits hits only
      if (maxITHitsToDropOTHitsPurePLS_ > static_cast<int>(::lstInputDevice::kITFitHits))
        throw edm::Exception(edm::errors::Configuration)
            << "LSTInputDeviceProducer: maxITHitsToDropOTHitsPurePLS = " << maxITHitsToDropOTHitsPurePLS_
            << " is larger than the " << ::lstInputDevice::kITFitHits
            << " hits of the inner-tracker-only fit of the seeds rebuilt without their OT hits";
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<double>("ptCut", 0.8);
      desc.add<int>("maxITHitsToDropOTHitsPurePLS", 3)
          ->setComment(
              "a pixel track whose last hit is in the OT and that has at most this many IT hits also gets an IT-only "
              "fit, for the pLS seed that LSTOutputConverter rebuilds without its OT hits; must equal the converters' "
              "maxITHitsToDropOTHitsPurePLS, and be at most 3");
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
      // the field of the fit, as the pixel-track fit takes it
      bField_ = static_cast<float>(1. / iSetup.getData(magneticFieldToken_).inverseBzAtOriginInGeV());

      const ::lstInputDevice::Settings settings{static_cast<float>(beamSpot.x0()),
                                                static_cast<float>(beamSpot.y0()),
                                                static_cast<float>(beamSpot.z0()),
                                                static_cast<float>(beamSpot.dxdz()),
                                                static_cast<float>(beamSpot.dydz()),
                                                ptCut_,
                                                bField_,
                                                pixelHits.nHits(),
                                                static_cast<uint32_t>(caModuleStart.size() - 1),
                                                static_cast<uint32_t>(std::max(maxITHitsToDropOTHitsPurePLS_, 0))};

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

      // the hits of each fit, the per-N track lists and the counters
      auto inputs = cms::alpakatools::make_device_buffer<::lstInputDevice::FitInput[]>(queue, nTracks_);
      auto lists = cms::alpakatools::make_device_buffer<uint32_t[]>(queue, ::lstInputDevice::kNumFitLists * nTracks_);
      counters_.emplace(cms::alpakatools::make_device_buffer<::lstInputDevice::Counters>(queue));
      alpaka::memset(queue, *counters_, 0);
      candidates_.emplace(cms::alpakatools::make_device_buffer<PixelSegmentCandidate[]>(queue, nTracks_));
      states_.emplace(cms::alpakatools::make_device_buffer<FittedState[]>(queue, nTracks_));
      itStates_.emplace(cms::alpakatools::make_device_buffer<FittedState[]>(queue, nTracks_));

      lstInputDevice::prepareTracks(queue,
                                    iEvent.get(pixelTracksSoAToken_).const_view(),
                                    nTracks_,
                                    edmIndexDevice.data(),
                                    pixelHits.const_view().trackingHits(),
                                    caModuleStartDevice.data(),
                                    caRecHitStartDevice.data(),
                                    iEvent.get(otRecHitsSoAToken_).const_view(),
                                    iSetup.getData(modulesToken_).const_view(),
                                    settings,
                                    inputs.data(),
                                    candidates_->data(),
                                    states_->data(),
                                    itStates_->data(),
                                    lists.data(),
                                    counters_->data());
      fitAllTracks(queue, inputs.data(), lists.data());
      lstInputDevice::makePixelSegments(
          queue, nTracks_, settings, states_->data(), candidates_->data(), counters_->data());

      seedIndex_.emplace(cms::alpakatools::make_device_buffer<uint32_t[]>(queue, nTracks_));
      segmentIndex_.emplace(cms::alpakatools::make_device_buffer<uint32_t[]>(queue, nTracks_));
      hitOffset_.emplace(cms::alpakatools::make_device_buffer<uint32_t[]>(queue, nTracks_));
      auto totalsDevice = cms::alpakatools::make_device_buffer<::lstInputDevice::Totals>(queue);
      lstInputDevice::indexPixelSegments(queue,
                                         states_->data(),
                                         candidates_->data(),
                                         nTracks_,
                                         seedIndex_->data(),
                                         segmentIndex_->data(),
                                         hitOffset_->data(),
                                         totalsDevice.data());
      totals_.emplace(cms::alpakatools::make_host_buffer<::lstInputDevice::Totals>(queue));
      alpaka::memcpy(queue, *totals_, totalsDevice);
      statesHost_.emplace(cms::alpakatools::make_host_buffer<FittedState[]>(queue, nTracks_));
      alpaka::memcpy(queue, *statesHost_, *states_);
    }

    void produce(device::Event& iEvent, device::EventSetup const& iSetup) override {
      auto& queue = iEvent.queue();
      auto const& totals = *totals_->data();
      auto const& otRecHitsSoA = iEvent.get(otRecHitsSoAToken_);
      const uint32_t nOTHits = otRecHitsSoA.const_view().metadata().size();
      auto const modules = iSetup.getData(modulesToken_).const_view();

      lst::LSTInputDeviceCollection lstInput(
          queue, nOTHits + totals.nSegmentHits, totals.nSegments, totals.nSegmentHits);
      lstInputDevice::fillLSTInput(queue,
                                   lstInput.view(),
                                   otRecHitsSoA.const_view(),
                                   nOTHits,
                                   modules,
                                   bField_,
                                   candidates_->data(),
                                   nTracks_,
                                   seedIndex_->data(),
                                   segmentIndex_->data(),
                                   hitOffset_->data());
      iEvent.emplace(lstInputPutToken_, std::move(lstInput));

      lst::PixelSeedStatesDeviceCollection seedStates(queue, totals.nSeeds, totals.nSeeds, totals.nSeeds);
      lstInputDevice::fillSeedStates(queue,
                                     seedStates.view(),
                                     bField_,
                                     states_->data(),
                                     itStates_->data(),
                                     candidates_->data(),
                                     nTracks_,
                                     seedIndex_->data(),
                                     counters_->data());
      iEvent.emplace(seedStatesPutToken_, std::move(seedStates));

      // OT rechit pointers in the order of the OT hits of the LST input
      auto const& otRecHits = iEvent.get(otRecHitsToken_);
      std::vector<TrackingRecHit const*> otHits;
      otHits.reserve(otRecHits.dataSize());
      for (auto const& detSet : otRecHits)
        for (auto const& hit : detSet)
          otHits.push_back(&hit);
      if (otHits.size() != nOTHits)
        throw edm::Exception(edm::errors::EventCorruption)
            << "LSTInputDeviceProducer: OT rechits " << otHits.size() << " != device OT rechits " << nOTHits;
      iEvent.emplace(lstOTHitsPutToken_, lst::LSTOTHits{std::move(otHits)});

      // the seeds of the fitted pixel tracks in track order: their hits and the fitted state on the last hit
      auto const& pixelTracks = iEvent.get(pixelTracksToken_);
      auto const& magneticField = iSetup.getData(magneticFieldToken_);
      TrajectorySeedCollection seeds;
      seeds.reserve(totals.nSeeds);
      for (uint32_t track = 0; track < nTracks_; ++track) {
        auto const& state = statesHost_->data()[track];
        if (state.status != static_cast<uint8_t>(::lst::SeedStateStatus::kOk))
          continue;
        auto const& pixelTrack = pixelTracks[edmIndex_->data()[track]];
        edm::OwnVector<TrackingRecHit> hits;
        for (auto const& hit : pixelTrack.recHits())
          hits.push_back(hit->clone());
        seeds.emplace_back(seedState(state.row, bField_, hits.back(), magneticField), hits, alongMomentum);
      }
      if (seeds.size() != totals.nSeeds)
        throw edm::Exception(edm::errors::LogicError)
            << "LSTInputDeviceProducer: seeds " << seeds.size() << " != fitted pixel tracks " << totals.nSeeds;
      iEvent.emplace(seedsPutToken_, std::move(seeds));

      edmIndex_.reset();
      counters_.reset();
      candidates_.reset();
      states_.reset();
      itStates_.reset();
      statesHost_.reset();
      seedIndex_.reset();
      segmentIndex_.reset();
      hitOffset_.reset();
      totals_.reset();
    }

  private:
    using FittedState = ::lstInputDevice::FittedState;
    using PixelSegmentCandidate = ::lstInputDevice::PixelSegmentCandidate;

    // the fit of every list: the inner-tracker-only fits and the full fits of each number of hits
    void fitAllTracks(Queue& queue, ::lstInputDevice::FitInput const* inputs, uint32_t const* lists) {
      auto* counters = counters_->data();
      auto* states = states_->data();
      auto* itStates = itStates_->data();
      lstInputDevice::fitTracks<3>(queue, nTracks_, inputs, bField_, lists, counters, states, itStates);
      lstInputDevice::fitTracks<4>(queue, nTracks_, inputs, bField_, lists, counters, states, itStates);
      lstInputDevice::fitTracks<5>(queue, nTracks_, inputs, bField_, lists, counters, states, itStates);
      lstInputDevice::fitTracks<6>(queue, nTracks_, inputs, bField_, lists, counters, states, itStates);
      lstInputDevice::fitTracks<7>(queue, nTracks_, inputs, bField_, lists, counters, states, itStates);
      lstInputDevice::fitTracks<8>(queue, nTracks_, inputs, bField_, lists, counters, states, itStates);
      lstInputDevice::fitTracks<9>(queue, nTracks_, inputs, bField_, lists, counters, states, itStates);
      lstInputDevice::fitTracks<10>(queue, nTracks_, inputs, bField_, lists, counters, states, itStates);
      lstInputDevice::fitTracks<11>(queue, nTracks_, inputs, bField_, lists, counters, states, itStates);
      lstInputDevice::fitTracks<12>(queue, nTracks_, inputs, bField_, lists, counters, states, itStates);
      lstInputDevice::fitTracks<13>(queue, nTracks_, inputs, bField_, lists, counters, states, itStates);
      lstInputDevice::fitTracks<14>(queue, nTracks_, inputs, bField_, lists, counters, states, itStates);
      static_assert(::lstInputDevice::kITFitHits == 3 && ::lstInputDevice::kMaxFitHits == 14,
                    "one fitTracks call and one compilation unit per number of fitted hits");
    }

    // The fitted state as the persistent state on the last hit: the free state at the stored point, with momentum
    // B / |kappa| (cos phi, sin phi, cot(theta)), the charge, and the curvilinear error of the stored covariance. The
    // state is attached to the last hit's surface as it is, not propagated onto it.
    static PTrajectoryStateOnDet seedState(::lst::BLFStateRow const& row,
                                           float bField,
                                           TrackingRecHit const& lastHit,
                                           MagneticField const& magneticField) {
      double covariance[5][5];
      ::lst::blfToCurvilinearCovariance(row.kappa, row.cotTheta, bField, row.covCircle, row.covLine, covariance);
      AlgebraicSymMatrix55 curvilinear;
      for (int i = 0; i < 5; ++i)
        for (int j = i; j < 5; ++j)
          curvilinear(i, j) = covariance[i][j];
      const float pt = bField / std::abs(row.kappa);
      const FreeTrajectoryState freeState(
          GlobalTrajectoryParameters(GlobalPoint(row.x, row.y, row.z),
                                     GlobalVector(pt * std::cos(row.phi), pt * std::sin(row.phi), pt * row.cotTheta),
                                     row.charge,
                                     &magneticField),
          CurvilinearTrajectoryError(curvilinear));
      return trajectoryStateTransform::persistentState(TrajectoryStateOnSurface(freeState, lastHit.det()->surface()),
                                                       lastHit.geographicalId().rawId());
    }

    const float ptCut_;
    const int maxITHitsToDropOTHitsPurePLS_;
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
    const device::EDPutToken<lst::PixelSeedStatesDeviceCollection> seedStatesPutToken_;
    const edm::EDPutTokenT<lst::LSTOTHits> lstOTHitsPutToken_;
    const edm::EDPutTokenT<TrajectorySeedCollection> seedsPutToken_;

    // per-event buffers and values from acquire to produce
    uint32_t nTracks_ = 0;
    float bField_ = 0.f;
    std::optional<cms::alpakatools::host_buffer<int32_t[]>> edmIndex_;
    std::optional<cms::alpakatools::device_buffer<Device, ::lstInputDevice::Counters>> counters_;
    std::optional<cms::alpakatools::device_buffer<Device, PixelSegmentCandidate[]>> candidates_;
    std::optional<cms::alpakatools::device_buffer<Device, FittedState[]>> states_;
    std::optional<cms::alpakatools::device_buffer<Device, FittedState[]>> itStates_;
    std::optional<cms::alpakatools::host_buffer<FittedState[]>> statesHost_;
    std::optional<cms::alpakatools::device_buffer<Device, uint32_t[]>> seedIndex_;
    std::optional<cms::alpakatools::device_buffer<Device, uint32_t[]>> segmentIndex_;
    std::optional<cms::alpakatools::device_buffer<Device, uint32_t[]>> hitOffset_;
    std::optional<cms::alpakatools::host_buffer<::lstInputDevice::Totals>> totals_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_ALPAKA_MODULE(LSTInputDeviceProducer);

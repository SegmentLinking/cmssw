#include <Eigen/Core>  // before the SoA headers with Eigen columns

#include <algorithm>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackSoA/interface/alpaka/TracksSoACollection.h"
#include "DataFormats/TrackerRecHit2D/interface/Phase2TrackerRecHit1D.h"
#include "DataFormats/TrackingRecHitSoA/interface/alpaka/Phase2OTRecHitsSoACollection.h"
#include "DataFormats/TrackingRecHitSoA/interface/alpaka/TrackingRecHitsSoACollection.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/ESInputTag.h"
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
#include "RecoTracker/LST/interface/BLFSeedFromState.h"
#include "RecoTracker/LST/interface/alpaka/LSTSeedFitModulesCollection.h"
#include "RecoTracker/LSTCore/interface/LSTOTHits.h"
#include "RecoTracker/LSTCore/interface/LSTPixelSeedRefs.h"
#include "RecoTracker/LSTCore/interface/LSTPixelTrackHits.h"
#include "RecoTracker/LSTCore/interface/SeedStateSoA.h"
#include "RecoTracker/LSTCore/interface/alpaka/LSTInputDeviceCollection.h"
#include "RecoTracker/LSTCore/interface/alpaka/PixelSeedStatesDeviceCollection.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"

#include "LSTInputDeviceKernels.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // LST's input made on the device. Each pixel track is fitted with the Broken Line Fit, its state taken at its last
  // hit; its pLS is computed from that state as LSTInputProducer computes it from a seed. The states of the fitted
  // tracks go to the seed-state product and their rechits to a pointer product, both in seed-index order; with
  // produceSeeds, the seeds of the fitted tracks are also made on the host from the states and those rechits.
  class LSTInputDeviceProducer : public stream::SynchronizingEDProducer<> {
  public:
    explicit LSTInputDeviceProducer(edm::ParameterSet const& iConfig)
        : SynchronizingEDProducer<>(iConfig),
          ptCut_(iConfig.getParameter<double>("ptCut")),
          maxITHitsToDropOTHitsPurePLS_(iConfig.getParameter<int>("maxITHitsToDropOTHitsPurePLS")),
          produceSeeds_(iConfig.getParameter<bool>("produceSeeds")),
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
          propagatorToken_(produceSeeds_ ? edm::ESGetToken<Propagator, TrackingComponentsRecord>{esConsumes(
                                               iConfig.getParameter<edm::ESInputTag>("propagator"))}
                                         : edm::ESGetToken<Propagator, TrackingComponentsRecord>{}),
          lstInputPutToken_(produces()),
          seedStatesPutToken_(produces()),
          lstOTHitsPutToken_(produces()),
          lstPixelTrackHitsPutToken_(produces()),
          seedsPutToken_(produceSeeds_ ? edm::EDPutTokenT<TrajectorySeedCollection>{produces()}
                                       : edm::EDPutTokenT<TrajectorySeedCollection>{}),
          lstPixelSeedRefsPutToken_(produceSeeds_ ? edm::EDPutTokenT<lst::LSTPixelSeedRefs>{produces()}
                                                  : edm::EDPutTokenT<lst::LSTPixelSeedRefs>{}) {
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
          ->setComment("the reco::Tracks of pixelTracksSoA and their index map, for the fitted tracks' rechits");
      desc.add<edm::InputTag>("pixelRecHitsSoA", edm::InputTag("hltPhase2SiPixelRecHitsSoA"));
      desc.add<edm::InputTag>("caOTHits", edm::InputTag("hltPhase2OtRecHitsSoA"))
          ->setComment("the CA's OT hits: their module starts and the first OT rechit of each module");
      desc.add<edm::InputTag>("otRecHitsSoA", edm::InputTag("hltSiPhase2RecHitsSoA"));
      desc.add<edm::InputTag>("otRecHits", edm::InputTag("hltSiPhase2RecHits"))
          ->setComment("the OT rechits of otRecHitsSoA, in the same order, for LSTOutputConverter");
      desc.add<edm::InputTag>("beamSpot", edm::InputTag("hltOnlineBeamSpot"));
      desc.add<bool>("produceSeeds", false)
          ->setComment(
              "also make the seeds of the fitted pixel tracks, in seed-index order: each state stepped onto the "
              "surface of its last hit with the propagator, or attached to it without propagation if the step fails");
      desc.add<edm::ESInputTag>("propagator", edm::ESInputTag("", "AnyDirectionAnalyticalPropagator"))
          ->setComment("the any-direction analytical propagator of the seeds; used only with produceSeeds");
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

      // the hits of each fit, one track list per fit size and the counters
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
      lstInputDevice::fitAllTracks(
          queue, nTracks_, inputs.data(), bField_, lists.data(), counters_->data(), states_->data(), itStates_->data());
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

      // the rechits of the fitted pixel tracks in seed-index order, each track's in track order
      auto const& pixelTracks = iEvent.get(pixelTracksToken_);
      std::vector<uint32_t> offsets;
      offsets.reserve(totals.nSeeds + 1);
      offsets.push_back(0);
      std::vector<TrackingRecHit const*> pixelHits;
      for (uint32_t track = 0; track < nTracks_; ++track) {
        if (statesHost_->data()[track].status != static_cast<uint8_t>(::lst::SeedStateStatus::kOk))
          continue;
        const int32_t edmTrack = edmIndex_->data()[track];
        if (edmTrack < 0)
          throw edm::Exception(edm::errors::LogicError)
              << "LSTInputDeviceProducer: fitted pixel track " << track << " has no reco::Track";
        for (auto const& hit : pixelTracks[edmTrack].recHits())
          pixelHits.push_back(&*hit);
        offsets.push_back(static_cast<uint32_t>(pixelHits.size()));
      }
      if (offsets.size() - 1 != totals.nSeeds)
        throw edm::Exception(edm::errors::LogicError) << "LSTInputDeviceProducer: rechits of " << offsets.size() - 1
                                                      << " pixel tracks != fitted pixel tracks " << totals.nSeeds;
      auto const pixelTrackHits =
          iEvent.emplace(lstPixelTrackHitsPutToken_, lst::LSTPixelTrackHits{std::move(offsets), std::move(pixelHits)});

      // the seeds of the fitted pixel tracks in seed-index order; the device keeps only the states that pass
      // lst::checkState, which blfSeedFromState applies too; dropping a seed would shift the seed indices of the
      // next ones
      if (produceSeeds_) {
        auto const& magneticField = iSetup.getData(magneticFieldToken_);
        auto const& propagator = iSetup.getData(propagatorToken_);
        auto const& seedHitOffsets = pixelTrackHits->offsets;
        auto const& seedHits = pixelTrackHits->hits;
        TrajectorySeedCollection seeds;
        seeds.reserve(totals.nSeeds);
        for (uint32_t track = 0; track < nTracks_; ++track) {
          auto const& state = statesHost_->data()[track];
          if (state.status != static_cast<uint8_t>(::lst::SeedStateStatus::kOk))
            continue;
          const uint32_t seed = seeds.size();
          const std::span<TrackingRecHit const* const> hits(seedHits.data() + seedHitOffsets[seed],
                                                            seedHitOffsets[seed + 1] - seedHitOffsets[seed]);
          auto result = ::lst::blfSeedFromState(
              state.row, state.nFit, bField_, hits, magneticField, propagator, /*attachIfStepFails=*/true);
          if (result.status != ::lst::BLFSeedStatus::kOk &&
              result.status != ::lst::BLFSeedStatus::kAttachedAfterFailedStep)
            throw edm::Exception(edm::errors::LogicError) << "LSTInputDeviceProducer: no seed for fitted pixel track "
                                                          << track << ", status " << static_cast<int>(result.status);
          seeds.push_back(std::move(*result.seed));
        }
        if (seeds.size() != totals.nSeeds)
          throw edm::Exception(edm::errors::LogicError)
              << "LSTInputDeviceProducer: seeds " << seeds.size() << " != fitted pixel tracks " << totals.nSeeds;
        auto const seedsHandle = iEvent.emplace(seedsPutToken_, std::move(seeds));

        // references to the seeds, in the order of the pixel seeds of the LST input, for LSTOutputConverter
        std::vector<edm::Ref<TrajectorySeedCollection>> seedRefs;
        seedRefs.reserve(seedsHandle->size());
        for (unsigned int iSeed = 0; iSeed < seedsHandle->size(); ++iSeed)
          seedRefs.emplace_back(seedsHandle, iSeed);
        iEvent.emplace(lstPixelSeedRefsPutToken_, lst::LSTPixelSeedRefs{std::move(seedRefs)});
      }

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

    const float ptCut_;
    const int maxITHitsToDropOTHitsPurePLS_;
    const bool produceSeeds_;
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
    const edm::ESGetToken<Propagator, TrackingComponentsRecord> propagatorToken_;
    const device::EDPutToken<lst::LSTInputDeviceCollection> lstInputPutToken_;
    const device::EDPutToken<lst::PixelSeedStatesDeviceCollection> seedStatesPutToken_;
    const edm::EDPutTokenT<lst::LSTOTHits> lstOTHitsPutToken_;
    const edm::EDPutTokenT<lst::LSTPixelTrackHits> lstPixelTrackHitsPutToken_;
    const edm::EDPutTokenT<TrajectorySeedCollection> seedsPutToken_;
    const edm::EDPutTokenT<lst::LSTPixelSeedRefs> lstPixelSeedRefsPutToken_;

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

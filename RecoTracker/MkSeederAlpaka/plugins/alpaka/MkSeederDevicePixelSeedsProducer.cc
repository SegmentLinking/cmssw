// MkSeederDevicePixelSeedsProducer: the mkFit seeder's selected seeds as device pixel seeds for the LST input on the
// device (RecoTracker/MkFitAlpaka MkFitAlpakaLstInputProducer 'pixelSeeds'), without the host TrajectorySeed path.
// Rows (TrackSoA, mkfitdev::TrackSoADeviceCollection): per seed that the classifier selected and that has a perigee,
// in the seeder's order, its hits as (mkFit layer, cluster index) in the seed's order (inside-out) and its state on its
// last hit (MkSeederSeedsSoA); row k is pixel track k of MkSeederPixelTrackProducer (the same selection), so the
// seed hand-off's host requests find the seed's hits in hltPhase2PixelTracks. The rows after the last seed: charge 0,
// no hits. With stateAtLastPixelHit also 'plsStates' (the same rows): the state the pLS starts from, the state at the
// last pixel hit of a seed that ends on an outer-tracker hit (MkSeederProducer 'lastPixelStates'; LSTInputProducer's
// option of the same name), else charge 0 (the row's own state).
// The collections have exactly as many rows as seeds (at least one): acquire() flags the seeds on the device and copies
// their number to the host, produce() allocates and fills the rows (instead of one row per seeder output row: the
// device memory of the rows and of the LST input's copies of them follows the selected seeds).

#include <algorithm>
#include <optional>

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/SynchronizingEDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "RecoTracker/MkFitAlpaka/interface/alpaka/tracks/TrackSoADeviceCollection.h"
#include "RecoTracker/MkSeederAlpaka/interface/alpaka/MkSeederSeedsDeviceCollection.h"

#include "MkSeederDevicePixelSeeds.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class MkSeederDevicePixelSeedsProducer : public stream::SynchronizingEDProducer<> {
  public:
    explicit MkSeederDevicePixelSeedsProducer(edm::ParameterSet const &iConfig)
        : SynchronizingEDProducer(iConfig),
          seedsToken_{consumes(iConfig.getParameter<edm::InputTag>("seeds"))},
          seedCountersToken_{consumes(iConfig.getParameter<edm::InputTag>("seeds"))},
          scoresToken_{consumes(iConfig.getParameter<edm::InputTag>("scores"))},
          rowsPutToken_{produces()} {
      p_.stateAtLastPixelHit = iConfig.getParameter<bool>("stateAtLastPixelHit");
      p_.stateAtLastPixelHitMaxPt = iConfig.getParameter<double>("stateAtLastPixelHitMaxPt");
      if (p_.stateAtLastPixelHit) {
        const auto &s = iConfig.getParameter<edm::InputTag>("seeds");
        pixelStatesToken_ = consumes(edm::InputTag(s.label(), "lastPixelStates", s.process()));
        plsStatesPutToken_ = produces("plsStates");
      }
    }

    static void fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::InputTag>("seeds", edm::InputTag{"hltMkFitSeederSeedsMkFitDevice"})
          ->setComment(
              "The MkSeederProducer module (the final seeds, their number; with stateAtLastPixelHit its "
              "'lastPixelStates', produceLastPixelStates)");
      desc.add<edm::InputTag>("scores", edm::InputTag{"hltMkFitSeederSeedsClassifiedDevice"})
          ->setComment("The MkSeederSeedClassifier module (the selection)");
      desc.add<bool>("stateAtLastPixelHit", false)
          ->setComment(
              "As LSTInputProducer: the pLS of a seed with >= 3 pixel hits that ends on an outer-tracker hit starts "
              "from its state at its last pixel hit (put as 'plsStates', MkFitAlpakaLstInputProducer "
              "pixelSeedsPLSStates); the seed's state stays the one on its last hit");
      desc.add<double>("stateAtLastPixelHitMaxPt", 0.)
          ->setComment("stateAtLastPixelHit only for seeds with pT (at the last hit) below this (GeV); 0: all seeds");
      descriptions.addWithDefaultLabel(desc);
    }

    void acquire(device::Event const &iEvent, device::EventSetup const &) override {
      auto &queue = iEvent.queue();
      const auto &seeds = iEvent.get(seedsToken_);
      const auto &counters = iEvent.get(seedCountersToken_);
      const auto &scores = iEvent.get(scoresToken_);
      cap_ = seeds.const_view().hits().metadata().size();
      nRows_.emplace(cms::alpakatools::make_host_buffer<uint32_t[]>(queue, 1u));
      nRows_->data()[0] = 0;
      if (cap_ == 0)
        return;
      flags_.emplace(cms::alpakatools::make_device_buffer<uint32_t[]>(queue, cap_));
      pos_.emplace(cms::alpakatools::make_device_buffer<uint32_t[]>(queue, cap_));
      mkseeder::flagDevicePixelSeeds(queue,
                                     seeds.const_view(),
                                     &counters.const_data()->nOutput,
                                     scores.const_view(),
                                     cap_,
                                     flags_->data(),
                                     pos_->data());
      alpaka::memcpy(queue, *nRows_, cms::alpakatools::make_device_view(queue, pos_->data() + cap_ - 1, 1u));
    }

    void produce(device::Event &iEvent, device::EventSetup const &) override {
      auto &queue = iEvent.queue();
      const auto &seeds = iEvent.get(seedsToken_);
      const auto &scores = iEvent.get(scoresToken_);
      const uint32_t n = nRows_->data()[0];
      // at least one row (no zero-size collection)
      mkfitdev::TrackSoADeviceCollection rows(queue, int32_t(std::max<uint32_t>(n, 1)));
      rows.zeroInitialise(queue);
      std::optional<mkfitdev::TrackSoADeviceCollection> plsStates;
      ::mkseeder::MkSeederSeedPixelStatesSoA::ConstView pixelStates;
      ::mkfitdev::TrackSoAView plsView;
      if (p_.stateAtLastPixelHit) {
        pixelStates = iEvent.get(pixelStatesToken_).const_view();
        plsStates.emplace(queue, int32_t(std::max<uint32_t>(n, 1)));
        plsStates->zeroInitialise(queue);
        plsView = plsStates->view();
      }
      if (cap_ > 0)
        mkseeder::fillDevicePixelSeeds(queue,
                                       seeds.const_view(),
                                       scores.const_view(),
                                       pixelStates,
                                       cap_,
                                       flags_->data(),
                                       pos_->data(),
                                       p_,
                                       rows.view(),
                                       plsView);
      flags_.reset(), pos_.reset(), nRows_.reset();
      iEvent.emplace(rowsPutToken_, std::move(rows));
      if (plsStates)
        iEvent.emplace(plsStatesPutToken_, std::move(*plsStates));
    }

  private:
    const device::EDGetToken<mkseeder::MkSeederSeedsDeviceCollection> seedsToken_;
    const device::EDGetToken<mkseeder::MkSeederDupCountersDeviceObject> seedCountersToken_;
    const device::EDGetToken<mkseeder::MkSeederSeedScoresDeviceCollection> scoresToken_;
    device::EDGetToken<mkseeder::MkSeederSeedPixelStatesDeviceCollection> pixelStatesToken_;
    const device::EDPutToken<mkfitdev::TrackSoADeviceCollection> rowsPutToken_;
    device::EDPutToken<mkfitdev::TrackSoADeviceCollection> plsStatesPutToken_;
    mkseeder::DevicePixelSeedsParameters p_{};
    // per event, from acquire() to produce(): the seeder's output rows, the seeds' row flags and their inclusive scan,
    // the number of rows
    uint32_t cap_ = 0;
    std::optional<cms::alpakatools::device_buffer<Device, uint32_t[]>> flags_, pos_;
    std::optional<cms::alpakatools::host_buffer<uint32_t[]>> nRows_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(MkSeederDevicePixelSeedsProducer);

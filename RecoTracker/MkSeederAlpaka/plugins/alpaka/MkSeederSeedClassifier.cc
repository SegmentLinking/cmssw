// MkSeederSeedClassifier: the seed classifier of the mkFit seeder on the device. Per final seed of MkSeederProducer
// (its features, MkSeederSeedFeaturesSoA.h, and their number) the xgboost model's score (MkSeederForestESProducer's
// trees) and the selection: score >= the threshold of the seed's |eta| bin, or of the 3-hit seeds; a dropped seed (its
// state does not convert) is never selected. The output stays on the device (MkSeederSeedScoresSoA.h) until a host
// module (MkSeederTrajectorySeedConverter) needs it.

#include <vector>

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/global/EDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTracker/MkSeederAlpaka/interface/alpaka/MkSeederForestDeviceCollection.h"
#include "RecoTracker/MkSeederAlpaka/interface/alpaka/MkSeederSeedsDeviceCollection.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"

#include "MkSeederClassifier.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class MkSeederSeedClassifier : public global::EDProducer<> {
  public:
    explicit MkSeederSeedClassifier(edm::ParameterSet const &iConfig)
        : EDProducer(iConfig),
          featuresToken_{consumes(iConfig.getParameter<edm::InputTag>("seeds"))},
          seedCountersToken_{consumes(iConfig.getParameter<edm::InputTag>("seeds"))},
          forestToken_{esConsumes(iConfig.getParameter<edm::ESInputTag>("forest"))},
          scoresPutToken_{produces()} {
      const auto etaEdges = iConfig.getParameter<std::vector<double>>("thresholdAbsEtaEdges");
      const auto etaThresholds = iConfig.getParameter<std::vector<double>>("thresholdsVsAbsEta");
      if (!etaThresholds.empty() && etaThresholds.size() != etaEdges.size() + 1)
        throw cms::Exception("Configuration")
            << "MkSeederSeedClassifier: thresholdsVsAbsEta needs one more entry than thresholdAbsEtaEdges";
      if (etaEdges.size() > size_t(mkseeder::SeedClassifierThresholds::kMaxEtaEdges))
        throw cms::Exception("Configuration") << "MkSeederSeedClassifier: at most "
                                              << mkseeder::SeedClassifierThresholds::kMaxEtaEdges << " |eta| bin edges";
      thresholds_.threshold = iConfig.getParameter<double>("threshold");
      thresholds_.tripletThreshold = iConfig.getParameter<double>("thresholdTriplets");
      thresholds_.zFold = iConfig.getParameter<bool>("zFold");
      thresholds_.nEtaEdges = etaThresholds.empty() ? 0 : etaEdges.size();
      for (int k = 0; k < thresholds_.nEtaEdges; ++k)
        thresholds_.etaEdges[k] = etaEdges[k];
      for (int k = 0; k < thresholds_.nEtaEdges + 1 && !etaThresholds.empty(); ++k)
        thresholds_.etaThresholds[k] = etaThresholds[k];
    }

    static void fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::InputTag>("seeds", edm::InputTag{"mkSeederProducer"})
          ->setComment("The MkSeederProducer module (the final seeds' features and their number)");
      desc.add<edm::ESInputTag>("forest", edm::ESInputTag{"", ""})
          ->setComment("The model's trees (MkSeederForestESProducer)");
      desc.add<bool>("zFold", false)
          ->setComment(
              "Evaluate the model on the z-mirrored features (dz, first and last hit z) of the seeds with last z < "
              "first z");
      desc.add<double>("threshold", 0.)->setComment("A seed is selected when its score >= threshold; 0 keeps all");
      desc.add<std::vector<double>>("thresholdAbsEtaEdges", {})
          ->setComment("Optional |eta| bin edges (ascending) for thresholdsVsAbsEta");
      desc.add<std::vector<double>>("thresholdsVsAbsEta", {})
          ->setComment("Optional per-|eta|-bin thresholds (one more than the edges); empty: 'threshold' everywhere");
      desc.add<double>("thresholdTriplets", -1.)->setComment("Threshold for 3-hit seeds; < 0: as the others");
      descriptions.addWithDefaultLabel(desc);
    }

    void produce(edm::StreamID, device::Event &iEvent, device::EventSetup const &iSetup) const override {
      const auto &features = iEvent.get(featuresToken_);
      const auto &seedCounters = iEvent.get(seedCountersToken_);
      const auto &forest = iSetup.getData(forestToken_);
      mkseeder::MkSeederSeedScoresDeviceCollection scores(iEvent.queue(), features.const_view().metadata().size());
      mkseeder::classifySeeds(iEvent.queue(),
                              features.const_view(),
                              &seedCounters.const_data()->nOutput,
                              forest.const_view(),
                              thresholds_,
                              scores.view());
      iEvent.emplace(scoresPutToken_, std::move(scores));
    }

  private:
    const device::EDGetToken<mkseeder::MkSeederSeedFeaturesDeviceCollection> featuresToken_;
    const device::EDGetToken<mkseeder::MkSeederDupCountersDeviceObject> seedCountersToken_;
    const device::ESGetToken<mkseeder::MkSeederForestDeviceCollection, TrackerRecoGeometryRecord> forestToken_;
    const device::EDPutToken<mkseeder::MkSeederSeedScoresDeviceCollection> scoresPutToken_;
    mkseeder::SeedClassifierThresholds thresholds_{};
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(MkSeederSeedClassifier);

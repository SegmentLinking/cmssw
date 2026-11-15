// Menu / test check of the per-event device status product, read from its host copy:
//   requireClean = True : throws cms::Exception("MkFitAlpakaStatus") on the first event with a non-zero counter;
//   requireClean = False: LogWarning per event with a non-zero counter.
// At the end of the job it prints the number of events, the number not clean and the sum of every counter.

#include <array>
#include <atomic>
#include <sstream>
#include <string>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

#include "RecoTracker/MkFitAlpaka/interface/StatusProduct.h"
#include "RecoTracker/MkFitAlpaka/interface/StatusReport.h"

class MkFitAlpakaStatusCheck : public edm::global::EDAnalyzer<> {
public:
  explicit MkFitAlpakaStatusCheck(edm::ParameterSet const& iConfig)
      : token_{consumes(iConfig.getParameter<edm::InputTag>("src"))},
        requireClean_{iConfig.getParameter<bool>("requireClean")},
        label_{iConfig.getParameter<edm::InputTag>("src").encode()} {
    for (auto& sum : sums_)
      sum = 0;
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("src", edm::InputTag("hltInitialStepTrackCandidatesMkFitDevice"))
        ->setComment("MkFitStatusHostObject (host copy of the device status product)");
    desc.add<bool>("requireClean", true)->setComment("throw on any non-zero counter");
    descriptions.addWithDefaultLabel(desc);
  }

  void analyze(edm::StreamID, edm::Event const& iEvent, edm::EventSetup const&) const override {
    const auto& st = iEvent.get(token_).value();
    nEvents_.fetch_add(1, std::memory_order_relaxed);
    if (mkfitdev::isClean(st))
      return;
    nDirty_.fetch_add(1, std::memory_order_relaxed);
    for (int i = 0; i < mkfitdev::kMaxStatusCounters; ++i)
      sums_[i].fetch_add(st.counter[i], std::memory_order_relaxed);
    std::ostringstream where;
    where << label_ << " run " << iEvent.id().run() << " event " << iEvent.id().event();
    if (requireClean_)
      mkfitdev::assertClean(st, where.str());
    mkfitdev::warnIfNotClean(st, where.str());
  }

  void endJob() override {
    std::ostringstream summary;
    summary << "MkFitAlpakaStatusCheck " << label_ << ": events " << nEvents_.load() << ", not clean " << nDirty_.load()
            << ";";
    for (int i = 0; i < mkfitdev::kNumStatusCounters; ++i)
      summary << " " << mkfitdev::statusCounterName(i) << "=" << sums_[i].load();
    edm::LogSystem("MkFitAlpakaStatusCheck") << summary.str();
  }

private:
  const edm::EDGetTokenT<mkfitdev::MkFitStatusHostObject> token_;
  const bool requireClean_;
  const std::string label_;
  mutable std::atomic<unsigned long long> nEvents_{0}, nDirty_{0};
  mutable std::array<std::atomic<unsigned long long>, mkfitdev::kMaxStatusCounters> sums_;
};

DEFINE_FWK_MODULE(MkFitAlpakaStatusCheck);

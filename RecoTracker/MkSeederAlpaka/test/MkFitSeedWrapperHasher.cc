// Validation tool for the refactoring of the mkFit seeder on alpaka: a stable 64-bit hash, per event, of the
// products of the seeder chain, printed one line per event and product (and a total per product at the end of the job),
// so that two runs (two commits, two backends) can be compared with diff.
//
// Hashed (all values at full float bits):
//  - MkFitSeedWrapper: for every seed the state (parameters, errors, charge), chi2, score, label, hits (index, layer) and the
//    seeder's quality records. Used for the main product and the "beamline" copy.
//  - std::vector<float> / std::vector<int>: the raw values (the classified seeds' "scores" and "mkFitIndices").
//  - the seeder's products (MkSeederProducer, on the host): the final seeds' perigees ("perigee") and the classifier's
//    features ("features") of the first nOutput seeds.
//  - TrajectorySeedCollection: the starting state (local parameters, errors, detector, side), the direction and the
//    hits (detector, local position and error).
//  - reco::TrackCollection: perigee parameters, covariance, chi2, hit counts, algorithm and the seed reference.
// Output lines: "HASH <event run:lumi:event> <product> n=<count> <hash>" and, at endJob, "HASHTOTAL <product> events=<n>
// n=<count> <hash>". The event-number order does not matter for the total (the per-event hashes are combined with
// a sum). With dumpFile, the values themselves of the seeds and of the float vectors of the first dumpEvents events
// are written to that file (hexadecimal floats, one seed or record per line), to find what differs between two runs.
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "RecoTracker/MkFit/interface/MkFitSeedWrapper.h"
#include "RecoTracker/MkFitCore/interface/Track.h"
#include "RecoTracker/MkSeederAlpaka/interface/MkSeederSeedsHostCollection.h"

namespace {

  // FNV-1a over the bytes, 64 bit
  class Hasher {
  public:
    void bytes(void const* p, size_t n) {
      auto const* b = static_cast<unsigned char const*>(p);
      for (size_t i = 0; i < n; ++i) {
        h_ ^= b[i];
        h_ *= 1099511628211ull;
      }
    }
    template <typename T>
    void add(T v) {
      static_assert(std::is_trivially_copyable_v<T>);
      bytes(&v, sizeof(T));
    }
    uint64_t value() const { return h_; }

  private:
    uint64_t h_ = 14695981039346656037ull;
  };

  struct Result {
    size_t n = 0;
    uint64_t hash = 0;
  };

  Result hashWrapper(MkFitSeedWrapper const& w) {
    Hasher h;
    auto const& seeds = w.seeds();
    h.add<uint64_t>(seeds.size());
    for (auto const& s : seeds) {
      for (int i = 0; i < 6; ++i)
        h.add<float>(s.state().parameters[i]);
      float const* e = s.state().errors.Array();
      for (int i = 0; i < 21; ++i)
        h.add<float>(e[i]);
      h.add<int>(s.charge());
      h.add<float>(s.chi2());
      h.add<float>(s.score());
      h.add<int>(s.label());
      h.add<int>(s.nTotalHits());
      h.add<int>(s.nFoundHits());
      h.add<int>(s.algoint());
      for (int i = 0; i < s.nTotalHits(); ++i) {
        h.add<int>(s.getHitIdx(i));
        h.add<int>(s.getHitLyr(i));
      }
    }
    auto const& q = w.quality();
    h.add<uint64_t>(q.size());
    for (auto const& x : q) {
      h.add<float>(x.clean_score);
      h.add<float>(x.fake_score);
      h.add<int>(x.n_amb);
    }
    return {seeds.size(), h.value()};
  }

  template <typename T>
  Result hashVector(std::vector<T> const& v) {
    Hasher h;
    h.add<uint64_t>(v.size());
    if (!v.empty())
      h.bytes(v.data(), v.size() * sizeof(T));
    return {v.size(), h.value()};
  }

  Result hashSeeds(TrajectorySeedCollection const& seeds) {
    Hasher h;
    h.add<uint64_t>(seeds.size());
    for (auto const& s : seeds) {
      auto const& st = s.startingState();
      auto const& lp = st.parameters();
      auto v = lp.mixedFormatVector();
      for (int i = 0; i < 5; ++i)
        h.add<double>(v[i]);
      h.add<int>(lp.charge());
      h.add<float>(lp.pzSign());
      for (int i = 0; i < 15; ++i)
        h.add<float>(st.error(i));
      h.add<float>(st.pt());
      h.add<unsigned>(st.detId());
      h.add<int>(st.surfaceSide());
      h.add<int>(static_cast<int>(s.direction()));
      h.add<unsigned>(s.nHits());
      for (auto const& hit : s.recHits()) {
        h.add<unsigned>(hit.geographicalId().rawId());
        auto const p = hit.localPosition();
        h.add<float>(p.x());
        h.add<float>(p.y());
        h.add<float>(p.z());
        auto const err = hit.localPositionError();
        h.add<float>(err.xx());
        h.add<float>(err.xy());
        h.add<float>(err.yy());
      }
    }
    return {seeds.size(), h.value()};
  }

  // the perigees of the first n seeds
  Result hashPerigee(mkseeder::MkSeederSeedsHostCollection const& seeds, uint32_t n) {
    Hasher h;
    h.add<uint64_t>(n);
    auto const perigee = seeds.const_view().perigee();
    for (uint32_t i = 0; i < n; ++i) {
      h.add<uint8_t>(perigee.status(i));
      if (perigee.status(i) != mkseeder::kPerigeeValid)
        continue;
      h.add<int8_t>(perigee.charge(i));
      for (int k = 0; k < 3; ++k)
        h.add<float>(perigee.vertex(i)(k));
      for (int k = 0; k < 3; ++k)
        h.add<float>(perigee.momentum(i)(k));
      for (int k = 0; k < 15; ++k)
        h.add<float>(perigee.covariance(i)(k));
    }
    return {n, h.value()};
  }

  // the classifier's features of the first n seeds
  Result hashFeatures(mkseeder::MkSeederSeedFeaturesHostCollection const& features, uint32_t n) {
    Hasher h;
    h.add<uint64_t>(n);
    auto const view = features.const_view();
    for (uint32_t i = 0; i < n; ++i) {
      h.add<uint8_t>(view.isDropped(i));
      h.add<uint8_t>(view.hasStateAtBeamline(i));
      for (int k = 0; k < mkseeder::seedfeature::kNFeatures; ++k)
        h.add<float>(view.inputs(i)(k));
    }
    return {n, h.value()};
  }

  Result hashTracks(reco::TrackCollection const& tracks) {
    Hasher h;
    h.add<uint64_t>(tracks.size());
    for (auto const& t : tracks) {
      for (int i = 0; i < 5; ++i)
        h.add<double>(t.parameter(i));
      for (int i = 0; i < 5; ++i)
        for (int j = i; j < 5; ++j)
          h.add<double>(t.covariance(i, j));
      h.add<double>(t.vx());
      h.add<double>(t.vy());
      h.add<double>(t.vz());
      h.add<double>(t.chi2());
      h.add<double>(t.ndof());
      h.add<int>(t.charge());
      h.add<int>(t.numberOfValidHits());
      h.add<int>(t.algo());
      h.add<unsigned>(t.seedRef().isNonnull() ? static_cast<unsigned>(t.seedRef().key()) : 0xffffffffu);
      for (auto it = t.recHitsBegin(); it != t.recHitsEnd(); ++it)
        h.add<unsigned>((*it)->geographicalId().rawId());
    }
    return {tracks.size(), h.value()};
  }

}  // namespace

class MkFitSeedWrapperHasher : public edm::global::EDAnalyzer<> {
public:
  explicit MkFitSeedWrapperHasher(edm::ParameterSet const& iConfig);
  void analyze(edm::StreamID, edm::Event const& iEvent, edm::EventSetup const&) const override;
  void endJob() override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  template <typename T>
  using Entries = std::vector<std::pair<std::string, edm::EDGetTokenT<T>>>;

  static std::string name(edm::InputTag const& tag) {
    return tag.label() + (tag.instance().empty() ? "" : ":" + tag.instance());
  }
  template <typename T, typename F>
  void run(edm::Event const& iEvent, Entries<T> const& entries, std::string const& ev, F&& f) const;
  void record(std::string const& product, std::string const& ev, Result const& r) const;

  Entries<MkFitSeedWrapper> wrappers_;
  Entries<std::vector<float>> floats_;
  Entries<std::vector<int>> ints_;
  Entries<TrajectorySeedCollection> seeds_;
  Entries<reco::TrackCollection> tracks_;
  // the seeder's products: the seeds (with their number) and the features
  Entries<mkseeder::MkSeederSeedsHostCollection> seedsSoA_;
  Entries<mkseeder::MkSeederDupCountersHostObject> seedCounters_;
  Entries<mkseeder::MkSeederSeedFeaturesHostCollection> features_;
  bool quiet_;
  unsigned int dumpEvents_;
  mutable std::ofstream dump_;
  mutable std::atomic<unsigned int> nDumped_{0};
  void dump(std::string const& ev, std::string const& product, MkFitSeedWrapper const& w) const;
  void dump(std::string const& ev, std::string const& product, std::vector<float> const& v) const;
  void dump(std::string const& ev, std::string const& product, std::vector<int> const& v) const;
  void dump(std::string const& ev,
            std::string const& product,
            mkseeder::MkSeederSeedFeaturesHostCollection const& features,
            uint32_t n) const;

  struct Total {
    uint64_t events = 0, n = 0, hash = 0;
  };
  mutable std::mutex mutex_;
  mutable std::map<std::string, Total> totals_;
};

MkFitSeedWrapperHasher::MkFitSeedWrapperHasher(edm::ParameterSet const& iConfig)
    : quiet_(iConfig.getParameter<bool>("quiet")), dumpEvents_(iConfig.getParameter<unsigned int>("dumpEvents")) {
  if (auto const& file = iConfig.getParameter<std::string>("dumpFile"); !file.empty())
    dump_.open(file);
  auto fill = [&](auto& entries, std::string const& par) {
    for (auto const& tag : iConfig.getParameter<std::vector<edm::InputTag>>(par))
      entries.emplace_back(name(tag), consumes(tag));
  };
  fill(wrappers_, "wrappers");
  fill(floats_, "floatVectors");
  fill(ints_, "intVectors");
  fill(seeds_, "trajectorySeeds");
  fill(tracks_, "tracks");
  fill(seedsSoA_, "seeder");
  fill(seedCounters_, "seeder");
  fill(features_, "seeder");
}

void MkFitSeedWrapperHasher::record(std::string const& product, std::string const& ev, Result const& r) const {
  if (!quiet_)
    edm::LogPrint("HASH") << "HASH " << ev << " " << product << " n=" << r.n << " " << std::hex << r.hash;
  std::lock_guard<std::mutex> lock(mutex_);
  auto& t = totals_[product];
  ++t.events;
  t.n += r.n;
  t.hash += r.hash;
}

template <typename T, typename F>
void MkFitSeedWrapperHasher::run(edm::Event const& iEvent,
                                 Entries<T> const& entries,
                                 std::string const& ev,
                                 F&& f) const {
  for (auto const& [product, token] : entries)
    record(product, ev, f(iEvent.get(token)));
}

void MkFitSeedWrapperHasher::analyze(edm::StreamID, edm::Event const& iEvent, edm::EventSetup const&) const {
  auto const id = iEvent.id();
  std::string const ev =
      std::to_string(id.run()) + ":" + std::to_string(id.luminosityBlock()) + ":" + std::to_string(id.event());
  run(iEvent, wrappers_, ev, [](auto const& x) { return hashWrapper(x); });
  run(iEvent, floats_, ev, [](auto const& x) { return hashVector(x); });
  run(iEvent, ints_, ev, [](auto const& x) { return hashVector(x); });
  run(iEvent, seeds_, ev, [](auto const& x) { return hashSeeds(x); });
  run(iEvent, tracks_, ev, [](auto const& x) { return hashTracks(x); });
  for (size_t k = 0; k < seedsSoA_.size(); ++k) {
    uint32_t const n = iEvent.get(seedCounters_[k].second)->nOutput;
    record(seedsSoA_[k].first + ":perigee", ev, hashPerigee(iEvent.get(seedsSoA_[k].second), n));
    record(features_[k].first + ":features", ev, hashFeatures(iEvent.get(features_[k].second), n));
  }
  if (dump_.is_open() && nDumped_++ < dumpEvents_) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto const& [product, token] : wrappers_)
      dump(ev, product, iEvent.get(token));
    for (auto const& [product, token] : floats_)
      dump(ev, product, iEvent.get(token));
    for (auto const& [product, token] : ints_)
      dump(ev, product, iEvent.get(token));
    for (size_t k = 0; k < features_.size(); ++k)
      dump(ev,
           features_[k].first + ":features",
           iEvent.get(features_[k].second),
           iEvent.get(seedCounters_[k].second)->nOutput);
  }
}

void MkFitSeedWrapperHasher::dump(std::string const& ev, std::string const& product, MkFitSeedWrapper const& w) const {
  auto const& seeds = w.seeds();
  auto const& q = w.quality();
  for (size_t i = 0; i < seeds.size(); ++i) {
    auto const& s = seeds[i];
    dump_ << ev << " " << product << " " << i << " label " << s.label() << " hits";
    for (int k = 0; k < s.nTotalHits(); ++k)
      dump_ << " " << s.getHitLyr(k) << ":" << s.getHitIdx(k);
    dump_ << std::hexfloat << " chg " << s.charge() << " chi2 " << s.chi2() << " score " << s.score() << " par";
    for (int k = 0; k < 6; ++k)
      dump_ << " " << s.state().parameters[k];
    dump_ << " err";
    for (int k = 0; k < 21; ++k)
      dump_ << " " << s.state().errors.Array()[k];
    dump_ << std::defaultfloat << "\n";
  }
  for (size_t i = 0; i < q.size(); ++i)
    dump_ << ev << " " << product << " quality " << i << std::hexfloat << " " << q[i].clean_score << " "
          << q[i].fake_score << std::defaultfloat << " " << q[i].n_amb << "\n";
}

void MkFitSeedWrapperHasher::dump(std::string const& ev,
                                  std::string const& product,
                                  std::vector<float> const& v) const {
  constexpr size_t kPerLine = 24;
  for (size_t i = 0; i < v.size(); i += kPerLine) {
    dump_ << ev << " " << product << " " << i << std::hexfloat;
    for (size_t k = i; k < std::min(v.size(), i + kPerLine); ++k)
      dump_ << " " << v[k];
    dump_ << std::defaultfloat << "\n";
  }
}

void MkFitSeedWrapperHasher::dump(std::string const& ev, std::string const& product, std::vector<int> const& v) const {
  constexpr size_t kPerLine = 24;
  for (size_t i = 0; i < v.size(); i += kPerLine) {
    dump_ << ev << " " << product << " " << i;
    for (size_t k = i; k < std::min(v.size(), i + kPerLine); ++k)
      dump_ << " " << v[k];
    dump_ << "\n";
  }
}

// per seed: isDropped, hasStateAtBeamline, the model's inputs
void MkFitSeedWrapperHasher::dump(std::string const& ev,
                                  std::string const& product,
                                  mkseeder::MkSeederSeedFeaturesHostCollection const& features,
                                  uint32_t n) const {
  auto const view = features.const_view();
  for (uint32_t i = 0; i < n; ++i) {
    dump_ << ev << " " << product << " " << i << " " << int(view.isDropped(i)) << " " << int(view.hasStateAtBeamline(i))
          << std::hexfloat;
    for (int k = 0; k < mkseeder::seedfeature::kNFeatures; ++k)
      dump_ << " " << view.inputs(i)(k);
    dump_ << std::defaultfloat << "\n";
  }
}

void MkFitSeedWrapperHasher::endJob() {
  for (auto const& [product, t] : totals_)
    edm::LogPrint("HASH") << "HASHTOTAL " << product << " events=" << t.events << " n=" << t.n << " " << std::hex
                          << t.hash;
}

void MkFitSeedWrapperHasher::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::vector<edm::InputTag>>("wrappers", {})->setComment("MkFitSeedWrapper products");
  desc.add<std::vector<edm::InputTag>>("floatVectors", {})->setComment("std::vector<float> products");
  desc.add<std::vector<edm::InputTag>>("intVectors", {})->setComment("std::vector<int> products");
  desc.add<std::vector<edm::InputTag>>("trajectorySeeds", {})->setComment("TrajectorySeedCollection products");
  desc.add<std::vector<edm::InputTag>>("tracks", {})->setComment("reco::TrackCollection products");
  desc.add<std::vector<edm::InputTag>>("seeder", {})
      ->setComment("MkSeederProducer modules: their seeds' perigees and the classifier's features");
  desc.add<bool>("quiet", false)->setComment("print only the totals at the end of the job");
  desc.add<std::string>("dumpFile", "")->setComment("non-empty: write the values of the first dumpEvents events there");
  desc.add<unsigned int>("dumpEvents", 1);
  descriptions.addWithDefaultLabel(desc);
}

DEFINE_FWK_MODULE(MkFitSeedWrapperHasher);

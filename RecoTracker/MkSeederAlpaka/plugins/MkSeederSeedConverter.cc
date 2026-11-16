// MkSeederSeedConverter: the final seeds of MkSeederProducer (its event products, on the host) as mkFit's seeds,
// mkFit's seeds of the seeder: the seeds with their quality, labelled by their position, and the same seeds with
// their state at the beam line ("beamline": the seed's own where the beam-line fit did not work). Not needed by the
// HLT menu (the seed classifier and the pixel tracks read the seeder's products); for validation and the mkFit seeds'
// consumers.

#include <algorithm>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "DataFormats/TrackReco/interface/TrackBase.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/EDPutToken.h"
#include "RecoTracker/MkFit/interface/MkFitSeedWrapper.h"
#include "RecoTracker/MkFitCore/interface/Track.h"
#include "RecoTracker/MkSeederAlpaka/interface/MkSeederSeedsHostCollection.h"

class MkSeederSeedConverter : public edm::global::EDProducer<> {
public:
  explicit MkSeederSeedConverter(edm::ParameterSet const &iConfig);

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

  void produce(edm::StreamID, edm::Event &iEvent, edm::EventSetup const &iSetup) const override;

private:
  static_assert(::mkseeder::kMaxSeedHits == mkfit::Track::Status::kMaxSeedHits);
  static constexpr int kMaxHits = ::mkseeder::kMaxSeedHits;

  // the seeds' columns as plain pointers, an Eigen column as its first component and the stride between its components
  struct SeedColumns {
    const uint8_t *nHits, *hasBeamlineState;
    const int32_t *charge, *beamlineCharge;
    const float *chi2;
    const int16_t *hitLayers;
    const int32_t *hitClusterIndices;
    const float *parameters, *covariance, *beamlineParameters, *beamlineCovariance;
    long layerStride, clusterIndexStride, parameterStride, covarianceStride;
    explicit SeedColumns(::mkseeder::MkSeederSeedsSoA::ConstView seeds);
  };
  // kSeedTile seeds of SeedColumns, read column by column (a seed's ~80 columns, each in its own memory page, are more
  // pages than the host's first-level TLB holds): the per-seed conversion reads them from here
  static constexpr uint32_t kSeedTile = 64;
  struct SeedTile {
    uint8_t nHits[kSeedTile], hasBeamlineState[kSeedTile];
    int32_t charge[kSeedTile], beamlineCharge[kSeedTile];
    float chi2[kSeedTile];
    int16_t hitLayers[kMaxHits][kSeedTile];
    int32_t hitClusterIndices[kMaxHits][kSeedTile];
    float parameters[6][kSeedTile], covariance[21][kSeedTile];
    float beamlineParameters[6][kSeedTile], beamlineCovariance[21][kSeedTile];
    // seeds [i0, i0 + m)
    void load(const SeedColumns &c, uint32_t i0, uint32_t m);
    mkfit::TrackState state(uint32_t j, bool beamline) const;
  };

  const edm::EDGetTokenT<::mkseeder::MkSeederSeedsHostCollection> seedsToken_;
  const edm::EDGetTokenT<::mkseeder::MkSeederDupCountersHostObject> seedCountersToken_;
  const edm::EDPutTokenT<MkFitSeedWrapper> seedsPutToken_;
  const edm::EDPutTokenT<MkFitSeedWrapper> beamlinePutToken_;
  const int algo_;
  const unsigned int maxNSeeds_;
};

MkSeederSeedConverter::SeedColumns::SeedColumns(::mkseeder::MkSeederSeedsSoA::ConstView seeds) {
  const auto hits = seeds.hits();
  const auto states = seeds.states();
  const auto clusterIndices = seeds.clusterIndices();
  nHits = hits.nHits().data();
  hasBeamlineState = states.hasBeamlineState().data(), charge = states.charge().data();
  beamlineCharge = states.beamlineCharge().data(), chi2 = states.chi2().data();
  hitLayers = hits.hitLayers(0).data(), layerStride = hits.hitLayers(0).innerStride();
  hitClusterIndices = clusterIndices.hitClusterIndex(0).data();
  clusterIndexStride = clusterIndices.hitClusterIndex(0).innerStride();
  parameters = states.parameters(0).data(), parameterStride = states.parameters(0).innerStride();
  covariance = states.covariance(0).data(), covarianceStride = states.covariance(0).innerStride();
  beamlineParameters = states.beamlineParameters(0).data();
  beamlineCovariance = states.beamlineCovariance(0).data();
}

void MkSeederSeedConverter::SeedTile::load(const SeedColumns &c, uint32_t i0, uint32_t m) {
  auto copy = [m, i0](auto *out, const auto *in) { std::copy(in + i0, in + i0 + m, out); };
  copy(nHits, c.nHits), copy(hasBeamlineState, c.hasBeamlineState);
  copy(charge, c.charge), copy(beamlineCharge, c.beamlineCharge), copy(chi2, c.chi2);
  // the hits up to the tile's largest number
  const int maxHits = *std::max_element(nHits, nHits + m);
  for (int k = 0; k < maxHits; ++k)
    copy(hitLayers[k], c.hitLayers + k * c.layerStride),
        copy(hitClusterIndices[k], c.hitClusterIndices + k * c.clusterIndexStride);
  for (int d = 0; d < 6; ++d)
    copy(parameters[d], c.parameters + d * c.parameterStride),
        copy(beamlineParameters[d], c.beamlineParameters + d * c.parameterStride);
  for (int d = 0; d < 21; ++d)
    copy(covariance[d], c.covariance + d * c.covarianceStride),
        copy(beamlineCovariance[d], c.beamlineCovariance + d * c.covarianceStride);
}

mkfit::TrackState MkSeederSeedConverter::SeedTile::state(uint32_t j, bool beamline) const {
  mkfit::TrackState ts;
  for (int d = 0; d < 6; ++d)
    ts.parameters[d] = beamline ? beamlineParameters[d][j] : parameters[d][j];
  for (int d = 0; d < 21; ++d)
    ts.errors.Array()[d] = beamline ? beamlineCovariance[d][j] : covariance[d][j];
  ts.charge = beamline ? beamlineCharge[j] : charge[j];
  return ts;
}

MkSeederSeedConverter::MkSeederSeedConverter(edm::ParameterSet const &iConfig)
    : seedsToken_{consumes(iConfig.getParameter<edm::InputTag>("src"))},
      seedCountersToken_{consumes(iConfig.getParameter<edm::InputTag>("src"))},
      seedsPutToken_{produces()},
      beamlinePutToken_{produces("beamline")},
      algo_{reco::TrackBase::algoByName(iConfig.getParameter<std::string>("algorithm"))},
      maxNSeeds_{iConfig.getParameter<unsigned int>("maxNSeeds")} {}

void MkSeederSeedConverter::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag{"mkSeederProducer"})
      ->setComment("The MkSeederProducer module (its final seeds and their number)");
  desc.add<std::string>("algorithm", "initialStep")->setComment("The seeds' track algorithm");
  desc.add("maxNSeeds", 500000U)->setComment("More seeds than this in an event: no seeds (with an error message)");
  descriptions.addWithDefaultLabel(desc);
}

void MkSeederSeedConverter::produce(edm::StreamID, edm::Event &iEvent, edm::EventSetup const &) const {
  const auto &seedsSoA = iEvent.get(seedsToken_);
  uint32_t n = iEvent.get(seedCountersToken_)->nOutput;
  mkfit::TrackVec seeds, blSeeds;
  std::vector<mkfit::SeedQuality> quality;
  if (n > maxNSeeds_) {
    edm::LogError("TooManySeeds") << "Exceeded maximum number of seeds! maxNSeeds=" << maxNSeeds_ << " nSeed=" << n;
    n = 0;
  }
  if (n > 0) {
    seeds.reserve(n), quality.reserve(n);
    // the beam-line states (the seeds whose beam-line fit worked); the beam-line copy is made after the seeds (its
    // hits' arrays then follow the seeds', as a copy of the vector lays them out)
    std::vector<std::pair<uint32_t, mkfit::TrackState>> blStates;
    blStates.reserve(n);
    const auto view = seedsSoA.const_view();
    const SeedColumns columns(view);
    const auto seedQuality = view.quality();
    auto tile = std::make_unique<SeedTile>();
    for (uint32_t i = 0; i < n; ++i) {
      const uint32_t j = i % kSeedTile;
      if (j == 0)
        tile->load(columns, i, std::min(kSeedTile, n - i));
      const int nHits = tile->nHits[j];
      mkfit::HitOnTrack hits[kMaxHits];
      for (int k = 0; k < nHits; ++k)
        hits[k] = {tile->hitClusterIndices[k][j], tile->hitLayers[k][j]};
      seeds.emplace_back(tile->state(j, false), tile->chi2[j], int(i), nHits, hits);
      seeds.back().setAlgorithm(mkfit::TrackBase::TrackAlgorithm(algo_));
      if (tile->hasBeamlineState[j])
        blStates.emplace_back(i, tile->state(j, true));
      quality.push_back({seedQuality.score(i), seedQuality.fakeScore(i), seedQuality.nAmbiguous(i)});
    }
    // the beam-line copy: the state at the beam line (if its fit worked, else the seed's)
    blSeeds = seeds;
    for (const auto &[i, state] : blStates)
      blSeeds[i].setState(state);
  }
  iEvent.emplace(beamlinePutToken_, std::move(blSeeds), std::vector<mkfit::SeedQuality>(quality));
  iEvent.emplace(seedsPutToken_, std::move(seeds), std::move(quality));
}

DEFINE_FWK_MODULE(MkSeederSeedConverter);

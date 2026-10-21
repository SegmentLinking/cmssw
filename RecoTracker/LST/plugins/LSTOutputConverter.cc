#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/TrackerRecHit2D/interface/Phase2TrackerRecHit1D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "DataFormats/TrackReco/interface/SeedStopInfo.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "Geometry/CommonTopologies/interface/GeomDet.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "RecoTracker/LST/interface/BLFSeedFromState.h"
#include "RecoTracker/LSTCore/interface/LSTOTHits.h"
#include "RecoTracker/LSTCore/interface/LSTPixelSeedRefs.h"
#include "RecoTracker/LSTCore/interface/LSTPixelTrackHits.h"
#include "RecoTracker/LSTCore/interface/PixelSeedStatesHostCollection.h"
#include "RecoTracker/LSTCore/interface/TCSeedStatesHostCollection.h"
#include "RecoTracker/LSTCore/interface/TrackCandidatesHostCollection.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedingHitSet.h"

#include "RecoTracker/TkSeedGenerator/interface/SeedCreator.h"
#include "RecoTracker/TkSeedGenerator/interface/SeedCreatorFactory.h"

#include "RecoTracker/TkTrackingRegions/interface/GlobalTrackingRegion.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"

#include <cmath>
#include <cstdint>
#include <optional>
#include <ranges>
#include <span>
#include <vector>

#include "RecoTracker/LST/plugins/LSTSeedHitList.h"

namespace {

  // The inner-tracker-only seed states of the LST input exist for tracks with exactly this many inner-tracker hits.
  constexpr int kITOnlyStateHits = 3;

  // The rechit accessors of the seed hit lists (LSTSeedHitList.h), as the makeSeed mode reads the rechits.
  class SeedRecHitAccess {
  public:
    SeedRecHitAccess(TrackerGeometry const& tracker, TrackerTopology const& topology)
        : tracker_(tracker), topology_(topology) {}

    lst::SeedHitOrderKey orderKey(TrackingRecHit const* hit) const {
      auto const* surface = hit->surface();
      return {hit->det()->subDetector(), surface->rSpan().first, std::abs(surface->zSpan().first)};
    }
    bool sharesInput(TrackingRecHit const* taken, TrackingRecHit const* other) const {
      return taken->sharesInput(other, TrackingRecHit::all);
    }
    bool inPixelSubdetector(TrackingRecHit const* hit) const {
      return hit->geographicalId().subdetId() <= PixelSubdetector::PixelEndcap;
    }
    bool isPSPixelSensor(TrackingRecHit const* hit) const {
      return tracker_.getDetectorType(hit->geographicalId()) == TrackerGeometry::ModuleType::Ph2PSP;
    }
    bool isPSStripSensor(TrackingRecHit const* hit) const {
      return tracker_.getDetectorType(hit->geographicalId()) == TrackerGeometry::ModuleType::Ph2PSS;
    }
    unsigned int layer(TrackingRecHit const* hit) const { return topology_.layer(hit->geographicalId()); }

  private:
    TrackerGeometry const& tracker_;
    TrackerTopology const& topology_;
  };

  lst::BLFStateRow stateRow(lst::SeedStateConst const& states, uint32_t i) {
    lst::BLFStateRow row{};
    row.x = states.x()[i];
    row.y = states.y()[i];
    row.z = states.z()[i];
    row.phi = states.phi()[i];
    row.cotTheta = states.cotTheta()[i];
    row.kappa = states.kappa()[i];
    for (int k = 0; k < 6; ++k)
      row.covCircle[k] = states.covCircle()[i][k];
    for (int k = 0; k < 3; ++k)
      row.covLine[k] = states.covLine()[i][k];
    row.charge = states.charge()[i];
    return row;
  }

}  // namespace

class LSTOutputConverter : public edm::stream::EDProducer<> {
public:
  explicit LSTOutputConverter(edm::ParameterSet const& iConfig);
  ~LSTOutputConverter() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;
  void produceSeedsFromStates(edm::Event& iEvent, const edm::EventSetup& iSetup) const;

  const bool seedsFromStates_;
  const edm::EDGetTokenT<lst::TrackCandidatesBaseHostCollection> lstOutputToken_;
  const edm::EDGetTokenT<lst::LSTOTHits> lstOTHitsToken_;
  const edm::EDGetTokenT<lst::LSTPixelSeedRefs> lstPixelSeedRefsToken_;
  const bool includeT5s_;
  const bool includeNonpLSTSs_;
  const bool dropOTHitsPurePLS_;
  const int maxITHitsToDropOTHitsPurePLS_;
  const bool produceSeeds_;
  const bool produceTrackCandidates_;
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> mfToken_;
  const edm::ESGetToken<Propagator, TrackingComponentsRecord> propagatorAlongToken_;
  const edm::ESGetToken<Propagator, TrackingComponentsRecord> propagatorOppositeToken_;
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> tGeomToken_;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> tTopoToken_;
  const edm::EDGetTokenT<lst::PixelSeedStatesHostCollection> pixelSeedStatesToken_;
  const edm::EDGetTokenT<lst::TCSeedStatesHostCollection> tcSeedStatesToken_;
  const edm::EDGetTokenT<lst::LSTPixelTrackHits> pixelTrackHitsToken_;
  const edm::ESGetToken<Propagator, TrackingComponentsRecord> propagatorToken_;
  std::unique_ptr<SeedCreator> seedCreator_;
  const edm::EDPutTokenT<TrajectorySeedCollection> trajectorySeedPutToken_;
  const edm::EDPutTokenT<TrajectorySeedCollection> trajectorySeedpLSPutToken_;
  const edm::EDPutTokenT<TrackCandidateCollection> trackCandidatePutToken_;
  const edm::EDPutTokenT<TrackCandidateCollection> trackCandidatepTCPutToken_;
  const edm::EDPutTokenT<TrackCandidateCollection> trackCandidateT4T5TCPutToken_;
  const edm::EDPutTokenT<TrackCandidateCollection> trackCandidateNopLSTCPutToken_;
  const edm::EDPutTokenT<TrackCandidateCollection> trackCandidatepTTCPutToken_;
  const edm::EDPutTokenT<TrackCandidateCollection> trackCandidatepLSTCPutToken_;
  const edm::EDPutTokenT<std::vector<SeedStopInfo>> seedStopInfoPutToken_;
  const edm::EDPutTokenT<std::vector<SeedStopInfo>> pTCsSeedStopInfoPutToken_;
  const edm::EDPutTokenT<std::vector<SeedStopInfo>> t4t5TCsSeedStopInfoPutToken_;
  const edm::EDPutTokenT<std::vector<SeedStopInfo>> pTTCsSeedStopInfoPutToken_;
};

LSTOutputConverter::LSTOutputConverter(edm::ParameterSet const& iConfig)
    : seedsFromStates_(iConfig.getParameter<bool>("seedsFromStates")),
      lstOutputToken_(consumes(iConfig.getParameter<edm::InputTag>("lstOutput"))),
      lstOTHitsToken_{consumes(iConfig.getParameter<edm::InputTag>("lstInput"))},
      lstPixelSeedRefsToken_{seedsFromStates_
                                 ? edm::EDGetTokenT<lst::LSTPixelSeedRefs>{}
                                 : consumes<lst::LSTPixelSeedRefs>(iConfig.getParameter<edm::InputTag>("lstInput"))},
      includeT5s_(iConfig.getParameter<bool>("includeT5s")),
      includeNonpLSTSs_(iConfig.getParameter<bool>("includeNonpLSTSs")),
      dropOTHitsPurePLS_(iConfig.getParameter<bool>("dropOTHitsPurePLS")),
      maxITHitsToDropOTHitsPurePLS_(iConfig.getParameter<int>("maxITHitsToDropOTHitsPurePLS")),
      produceSeeds_(iConfig.getParameter<bool>("produceSeeds")),
      produceTrackCandidates_(iConfig.getParameter<bool>("produceTrackCandidates")),
      mfToken_(esConsumes()),
      propagatorAlongToken_{seedsFromStates_ ? edm::ESGetToken<Propagator, TrackingComponentsRecord>{}
                                             : edm::ESGetToken<Propagator, TrackingComponentsRecord>{esConsumes(
                                                   iConfig.getParameter<edm::ESInputTag>("propagatorAlong"))}},
      propagatorOppositeToken_{seedsFromStates_ ? edm::ESGetToken<Propagator, TrackingComponentsRecord>{}
                                                : edm::ESGetToken<Propagator, TrackingComponentsRecord>{esConsumes(
                                                      iConfig.getParameter<edm::ESInputTag>("propagatorOpposite"))}},
      tGeomToken_(esConsumes()),
      tTopoToken_(esConsumes()),
      pixelSeedStatesToken_(seedsFromStates_ ? consumes<lst::PixelSeedStatesHostCollection>(
                                                   iConfig.getParameter<edm::InputTag>("pixelSeedStates"))
                                             : edm::EDGetTokenT<lst::PixelSeedStatesHostCollection>{}),
      tcSeedStatesToken_(seedsFromStates_ ? consumes<lst::TCSeedStatesHostCollection>(
                                                iConfig.getParameter<edm::InputTag>("tcSeedStates"))
                                          : edm::EDGetTokenT<lst::TCSeedStatesHostCollection>{}),
      pixelTrackHitsToken_(seedsFromStates_
                               ? consumes<lst::LSTPixelTrackHits>(iConfig.getParameter<edm::InputTag>("pixelTrackHits"))
                               : edm::EDGetTokenT<lst::LSTPixelTrackHits>{}),
      propagatorToken_(seedsFromStates_ ? edm::ESGetToken<Propagator, TrackingComponentsRecord>{esConsumes(
                                              iConfig.getParameter<edm::ESInputTag>("propagator"))}
                                        : edm::ESGetToken<Propagator, TrackingComponentsRecord>{}),
      seedCreator_(seedsFromStates_
                       ? std::unique_ptr<SeedCreator>{}
                       : SeedCreatorFactory::get()->create("SeedFromConsecutiveHitsCreator",
                                                           iConfig.getParameter<edm::ParameterSet>("SeedCreatorPSet"),
                                                           consumesCollector())),
      trajectorySeedPutToken_(produces<TrajectorySeedCollection>("")),
      trajectorySeedpLSPutToken_(produceSeeds_ ? produces<TrajectorySeedCollection>("pLSTSsLST")
                                               : edm::EDPutTokenT<TrajectorySeedCollection>{}),
      trackCandidatePutToken_(produceTrackCandidates_ ? produces<TrackCandidateCollection>("")
                                                      : edm::EDPutTokenT<TrackCandidateCollection>{}),
      trackCandidatepTCPutToken_(produceTrackCandidates_ ? produces<TrackCandidateCollection>("pTCsLST")
                                                         : edm::EDPutTokenT<TrackCandidateCollection>{}),
      trackCandidateT4T5TCPutToken_(produceTrackCandidates_ ? produces<TrackCandidateCollection>("t4t5TCsLST")
                                                            : edm::EDPutTokenT<TrackCandidateCollection>{}),
      trackCandidateNopLSTCPutToken_(produceTrackCandidates_ ? produces<TrackCandidateCollection>("nopLSTCsLST")
                                                             : edm::EDPutTokenT<TrackCandidateCollection>{}),
      trackCandidatepTTCPutToken_(produceTrackCandidates_ ? produces<TrackCandidateCollection>("pTTCsLST")
                                                          : edm::EDPutTokenT<TrackCandidateCollection>{}),
      trackCandidatepLSTCPutToken_(produceTrackCandidates_ ? produces<TrackCandidateCollection>("pLSTCsLST")
                                                           : edm::EDPutTokenT<TrackCandidateCollection>{}),
      seedStopInfoPutToken_(produceTrackCandidates_ ? produces<std::vector<SeedStopInfo>>("")
                                                    : edm::EDPutTokenT<std::vector<SeedStopInfo>>{}),
      pTCsSeedStopInfoPutToken_(produceTrackCandidates_ ? produces<std::vector<SeedStopInfo>>("pTCsLST")
                                                        : edm::EDPutTokenT<std::vector<SeedStopInfo>>{}),
      t4t5TCsSeedStopInfoPutToken_(produceTrackCandidates_ ? produces<std::vector<SeedStopInfo>>("t4t5TCsLST")
                                                           : edm::EDPutTokenT<std::vector<SeedStopInfo>>{}),
      pTTCsSeedStopInfoPutToken_(produceTrackCandidates_ ? produces<std::vector<SeedStopInfo>>("pTTCsLST")
                                                         : edm::EDPutTokenT<std::vector<SeedStopInfo>>{}) {
  if (seedsFromStates_ && produceTrackCandidates_)
    throw edm::Exception(edm::errors::Configuration)
        << "LSTOutputConverter: seedsFromStates makes seeds only and cannot be used with produceTrackCandidates";
  if (seedsFromStates_ && dropOTHitsPurePLS_ && maxITHitsToDropOTHitsPurePLS_ > kITOnlyStateHits)
    throw edm::Exception(edm::errors::Configuration)
        << "LSTOutputConverter: with seedsFromStates and dropOTHitsPurePLS, maxITHitsToDropOTHitsPurePLS = "
        << maxITHitsToDropOTHitsPurePLS_ << " is larger than the " << kITOnlyStateHits
        << " hits of the inner-tracker-only seed states of the pLS seeds rebuilt without their OT hits";
}

void LSTOutputConverter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("lstOutput", edm::InputTag("lstProducer"));
  desc.add<edm::InputTag>("lstInput", edm::InputTag("lstInputProducer"));
  desc.add<bool>("includeT5s", true);
  desc.add<bool>("includeNonpLSTSs", false);
  desc.add<bool>("dropOTHitsPurePLS", false);
  desc.add<int>("maxITHitsToDropOTHitsPurePLS", 3);
  desc.add<bool>("produceSeeds", true);
  desc.add<bool>("produceTrackCandidates", true);
  desc.add("propagatorAlong", edm::ESInputTag{"", "PropagatorWithMaterial"});
  desc.add("propagatorOpposite", edm::ESInputTag{"", "PropagatorWithMaterialOpposite"});
  desc.add<bool>("seedsFromStates", false)
      ->setComment(
          "make every seed from the stored Broken Line Fit states with no refit, each state stepped onto the surface "
          "of its seed's last hit; seeds only (produceTrackCandidates must be false), with dropOTHitsPurePLS "
          "maxITHitsToDropOTHitsPurePLS at most 3; the pixel seeds of lstInput, the two propagators above and "
          "SeedCreatorPSet are then not used");
  desc.add<edm::InputTag>("pixelSeedStates", edm::InputTag("lstInputProducer"))
      ->setComment(
          "with seedsFromStates: the seed states of the fitted pixel tracks of the LST input, in seed-index order");
  desc.add<edm::InputTag>("tcSeedStates", edm::InputTag("lstProducer"))
      ->setComment("with seedsFromStates: the seed states of the track candidates, in track-candidate order");
  desc.add<edm::InputTag>("pixelTrackHits", edm::InputTag("lstInputProducer"))
      ->setComment(
          "with seedsFromStates: the rechits of the fitted pixel tracks of the LST input, in seed-index order");
  desc.add<edm::ESInputTag>("propagator", edm::ESInputTag("", "AnyDirectionAnalyticalPropagator"))
      ->setComment("with seedsFromStates: the any-direction analytical propagator of the step onto the last hit");

  edm::ParameterSetDescription psd0;
  psd0.add<std::string>("ComponentName", std::string("SeedFromConsecutiveHitsCreator"));
  psd0.add<std::string>("propagator", std::string("PropagatorWithMaterial"));
  psd0.add<double>("SeedMomentumForBOFF", 5.0);
  psd0.add<double>("OriginTransverseErrorMultiplier", 1.0);
  psd0.add<double>("MinOneOverPtError", 1.0);
  psd0.add<std::string>("magneticField", std::string(""));
  psd0.add<std::string>("TTRHBuilder", std::string("WithTrackAngle"));
  psd0.add<bool>("forceKinematicWithRegionDirection", false);
  desc.add<edm::ParameterSetDescription>("SeedCreatorPSet", psd0);

  descriptions.addWithDefaultLabel(desc);
}

void LSTOutputConverter::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  if (seedsFromStates_) {
    produceSeedsFromStates(iEvent, iSetup);
    return;
  }

  // Setup
  auto const& lstOutput = iEvent.get(lstOutputToken_);
  // the input pixel seeds, indexed by the LST pixel seed index
  auto const& pixelSeeds = iEvent.get(lstPixelSeedRefsToken_).seeds;
  auto const& mf = iSetup.getData(mfToken_);
  auto const& propAlo = iSetup.getData(propagatorAlongToken_);
  auto const& propOppo = iSetup.getData(propagatorOppositeToken_);
  auto const& tracker = iSetup.getData(tGeomToken_);
  const TrackerTopology& tTopo = iSetup.getData(tTopoToken_);

  auto lstOutput_view = lstOutput.const_view();
  unsigned int nTrackCandidates = lstOutput_view.nTrackCandidates();

  auto const outputTSRP = iEvent.getRefBeforePut(trajectorySeedPutToken_);

  TrajectorySeedCollection outputTS, outputpLSTS;
  outputTS.reserve(nTrackCandidates);
  outputpLSTS.reserve(nTrackCandidates);
  TrackCandidateCollection outputTC, outputpTC, outputT4T5TC, outputNopLSTC, outputpTTC, outputpLSTC;
  outputTC.reserve(nTrackCandidates);
  outputpTC.reserve(nTrackCandidates);
  outputT4T5TC.reserve(nTrackCandidates);
  outputNopLSTC.reserve(nTrackCandidates);
  outputpTTC.reserve(nTrackCandidates);
  outputpLSTC.reserve(nTrackCandidates);

  auto const& otHits = iEvent.get(lstOTHitsToken_).hits;

  TrajectorySeedCollection seeds;
  using Hit = SeedingHitSet::ConstRecHitPointer;
  std::vector<Hit> hitsForSeed;

  LogDebug("LSTOutputConverter") << "nTrackCandidates " << nTrackCandidates;
  for (unsigned int i = 0; i < nTrackCandidates; i++) {
    auto iType = lstOutput_view.trackCandidateType()[i];
    bool const isT5orT4 = (iType == lst::LSTObjType::T5 || iType == lst::LSTObjType::T4);
    const auto iSeed = lstOutput_view.pixelSeedIndex()[i];
    if (!isT5orT4 && iSeed >= pixelSeeds.size())
      throw cms::Exception("LogicError") << "LST pixel seed index " << iSeed << " out of range: " << pixelSeeds.size()
                                         << " input pixel seeds";
    // T5 and T4 candidates have no pixel seed
    const auto pixelSeed = isT5orT4 ? edm::Ref<TrajectorySeedCollection>() : pixelSeeds[iSeed];
    const bool dropHitsOTpL =
        iType == lst::LSTObjType::pLS && dropOTHitsPurePLS_ &&
        std::prev(pixelSeed->recHits().end())->geographicalId().subdetId() > PixelSubdetector::PixelEndcap &&
        std::ranges::count_if(pixelSeed->recHits(), [](const auto& h) {
          return h.geographicalId().subdetId() <= PixelSubdetector::PixelEndcap;
        }) <= maxITHitsToDropOTHitsPurePLS_;
    LogDebug("LSTOutputConverter") << " cand " << i << " " << iType << " " << iSeed;
    TrajectorySeed seed;
    edm::RefToBase<TrajectorySeed> seedRef;
    edm::OwnVector<TrackingRecHit> recHits;
    if (!isT5orT4 && !dropHitsOTpL) {
      seed = *pixelSeed;
      seedRef = edm::RefToBase<TrajectorySeed>(pixelSeed);

      for (auto const& hit : seed.recHits())
        recHits.push_back(hit.clone());
    }

    if (iType != lst::LSTObjType::pLS) {
      // The pixel hits are packed into first kPixelLayerSlots layer slots.
      for (unsigned int layerSlot = lst::Params_TC::kPixelLayerSlots; layerSlot < lst::Params_TC::kLayers;
           ++layerSlot) {
        for (unsigned int hitSlot = 0; hitSlot < lst::Params_TC::kHitsPerLayer; ++hitSlot) {
          unsigned int hitIdx = lstOutput_view.hitIndices()[i][layerSlot][hitSlot];
          if (hitIdx == lst::kTCEmptyHitIdx)
            continue;
          bool hitOK = true;
          for (auto const& hit : recHits)
            if (hit.sharesInput(otHits[hitIdx], TrackingRecHit::all)) {
              hitOK = false;
              break;
            }
          if (hitOK)
            recHits.push_back(otHits[hitIdx]->clone());
        }
      }

      recHits.sort([](const auto& a, const auto& b) {
        const auto asub = a.det()->subDetector();
        const auto bsub = b.det()->subDetector();
        if (GeomDetEnumerators::isInnerTracker(asub) && GeomDetEnumerators::isOuterTracker(bsub)) {
          return true;
        } else if (GeomDetEnumerators::isOuterTracker(asub) && GeomDetEnumerators::isInnerTracker(bsub)) {
          return false;
        } else if (asub != bsub) {
          return asub < bsub;
        } else {
          const auto& apos = a.surface();
          const auto& bpos = b.surface();
          if (GeomDetEnumerators::isBarrel(asub)) {
            return apos->rSpan().first < bpos->rSpan().first;
          } else {
            return std::abs(apos->zSpan().first) < std::abs(bpos->zSpan().first);
          }
        }
      });

      // For T5/T4: makeSeed is needed whenever seeds or TCs are produced, since the resulting
      // seed is the only source of initial state for T5/T4 track candidates.
      // For other pT objects: makeSeed is only needed for seed output.
      if ((isT5orT4 && (produceSeeds_ || produceTrackCandidates_)) ||
          (!isT5orT4 && produceSeeds_ && includeNonpLSTSs_)) {
        hitsForSeed.clear();
        hitsForSeed.reserve(recHits.size());
        int n = 0;
        unsigned int firstLayer;
        for (auto const& hit : recHits) {
          if (iType == lst::LSTObjType::T5) {
            auto hType = tracker.getDetectorType(hit.geographicalId());
            if (hType != TrackerGeometry::ModuleType::Ph2PSP && n < 2)
              continue;  // the first two should be P
          }
          if (iType == lst::LSTObjType::T4) {
            unsigned int hitLayer = tTopo.layer(hit.geographicalId());
            auto hType = tracker.getDetectorType(hit.geographicalId());
            if (n == 0)
              firstLayer = hitLayer;
            else {
              if (hType == TrackerGeometry::ModuleType::Ph2PSS && hitLayer == firstLayer)
                continue;
            }
          }
          hitsForSeed.emplace_back(dynamic_cast<Hit>(&hit));
          n++;
        }
        GlobalTrackingRegion region;
        seedCreator_->init(region, iSetup, nullptr);
        seeds.clear();
        seedCreator_->makeSeed(seeds, hitsForSeed);
        if (seeds.empty()) {
          edm::LogInfo("LSTOutputConverter") << "failed to convert a LST object to a seed" << i << " " << iType << " "
                                             << lstOutput_view.pixelSeedIndex()[i];
          if (isT5orT4)
            continue;
        }
        if (isT5orT4) {
          seed = seeds[0];
          seedRef = edm::RefToBase<TrajectorySeed>(edm::Ref(outputTSRP, outputTS.size()));
        }

        auto trajectorySeed = (seeds.empty() ? seed : seeds[0]);
        outputTS.emplace_back(trajectorySeed);
        auto const& ss = trajectorySeed.startingState();
        LogDebug("LSTOutputConverter") << "Created a seed with " << trajectorySeed.nHits() << " " << ss.detId() << " "
                                       << ss.pt() << " " << ss.parameters().vector() << " " << ss.error(0);
      }
    } else if (produceSeeds_ || (produceTrackCandidates_ && dropHitsOTpL)) {  // pLS handling
      if (dropHitsOTpL) {                                                     // true if need to drop OT hits
        using Hit = SeedingHitSet::ConstRecHitPointer;
        hitsForSeed.clear();
        hitsForSeed.reserve(std::ranges::size(pixelSeed->recHits()));
        for (auto const& hit : pixelSeed->recHits()) {
          if (hit.geographicalId().subdetId() > PixelSubdetector::PixelEndcap)
            continue;
          hitsForSeed.emplace_back(dynamic_cast<Hit>(&hit));
          recHits.push_back(hit.clone());
        }
        GlobalTrackingRegion region;
        seedCreator_->init(region, iSetup, nullptr);
        seeds.clear();
        seedCreator_->makeSeed(seeds, hitsForSeed);
        if (seeds.empty())
          edm::LogInfo("LSTOutputConverter") << "failed to convert a pLS object to a seed" << i << iSeed;
        seed = seeds[0];
        seedRef = edm::RefToBase<TrajectorySeed>(edm::Ref(outputTSRP, outputTS.size()));
      }
      outputTS.emplace_back(seed);
      if (produceSeeds_)  // matches the logic for iEvent.emplace
        outputpLSTS.emplace_back(seed);
    }

    if (!produceTrackCandidates_)
      continue;

    TrajectoryStateOnSurface tsos =
        trajectoryStateTransform::transientState(seed.startingState(), (seed.recHits().end() - 1)->surface(), &mf);
    tsos.rescaleError(100.);
    auto tsosPair = propOppo.propagateWithPath(tsos, *recHits[0].surface());
    if (!tsosPair.first.isValid()) {
      LogDebug("LSTOutputConverter") << "Propagating to startingState opposite to momentum failed, trying along next";
      tsosPair = propAlo.propagateWithPath(tsos, *recHits[0].surface());
    }
    if (tsosPair.first.isValid()) {
      PTrajectoryStateOnDet st =
          trajectoryStateTransform::persistentState(tsosPair.first, recHits[0].det()->geographicalId().rawId());

      if (!includeT5s_ && isT5orT4)
        continue;

      auto tc = TrackCandidate(recHits, seed, st, seedRef);
      outputTC.emplace_back(tc);
      if (isT5orT4) {
        outputT4T5TC.emplace_back(tc);
        outputNopLSTC.emplace_back(tc);
      } else {
        outputpTC.emplace_back(tc);
        if (iType != lst::LSTObjType::pLS) {
          outputNopLSTC.emplace_back(tc);
          outputpTTC.emplace_back(tc);
        } else {
          outputpLSTC.emplace_back(tc);
        }
      }
    } else {
      edm::LogInfo("LSTOutputConverter") << "Failed to make a candidate initial state. Seed state is " << tsos
                                         << " TC cand " << i << " " << lstOutput_view.pixelSeedIndex()[i] << " "
                                         << lstOutput_view.pixelSeedIndex()[i] << " first hit "
                                         << recHits.front().globalPosition() << " last hit "
                                         << recHits.back().globalPosition();
    }
  }

  LogDebug("LSTOutputConverter") << "done with conversion: Track candidate output size = " << outputpTC.size()
                                 << " (p* objects) + " << outputT4T5TC.size() << " (T5 objects)";

  iEvent.emplace(trajectorySeedPutToken_, std::move(outputTS));
  if (produceSeeds_)
    iEvent.emplace(trajectorySeedpLSPutToken_, std::move(outputpLSTS));
  if (produceTrackCandidates_) {
    //dummy (for now) stop infos: one per used kind of candidates
    iEvent.emplace(seedStopInfoPutToken_, std::vector<SeedStopInfo>(pixelSeeds.size()));
    iEvent.emplace(pTCsSeedStopInfoPutToken_, std::vector<SeedStopInfo>(pixelSeeds.size()));
    iEvent.emplace(t4t5TCsSeedStopInfoPutToken_, std::vector<SeedStopInfo>(outputT4T5TC.size()));
    iEvent.emplace(pTTCsSeedStopInfoPutToken_, std::vector<SeedStopInfo>(pixelSeeds.size()));
    iEvent.emplace(trackCandidatePutToken_, std::move(outputTC));
    iEvent.emplace(trackCandidatepTCPutToken_, std::move(outputpTC));
    iEvent.emplace(trackCandidateT4T5TCPutToken_, std::move(outputT4T5TC));
    iEvent.emplace(trackCandidateNopLSTCPutToken_, std::move(outputNopLSTC));
    iEvent.emplace(trackCandidatepTTCPutToken_, std::move(outputpTTC));
    iEvent.emplace(trackCandidatepLSTCPutToken_, std::move(outputpLSTC));
  }
}

// The seeds from the stored states, in track-candidate order with the outputs of the makeSeed mode: a pLS from its
// pixel-track state; a T5 or a T4 from its track-candidate state, or no seed; a pT3 or a pT5 from its track-candidate
// state, or else the seed of its pixel track. A pixel-track state always gives a seed, attached to the surface of its
// last hit if the step fails: the LST input keeps only the states that pass the checks of the seed function.
void LSTOutputConverter::produceSeedsFromStates(edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  using Hit = TrackingRecHit const*;
  auto const tcs = iEvent.get(lstOutputToken_).const_view();
  const uint32_t nTrackCandidates = tcs.nTrackCandidates();
  auto const& otHits = iEvent.get(lstOTHitsToken_).hits;
  auto const pixelStates = iEvent.get(pixelSeedStatesToken_).const_view();
  const lst::SeedStateConst pixelState = pixelStates.state();
  const lst::SeedStateConst pixelITState = pixelStates.itState();
  const lst::PixelSeedStateExtraConst pixelExtra = pixelStates.extra();
  const lst::SeedStateConst tcState = iEvent.get(tcSeedStatesToken_).const_view().state();
  auto const& pixelTrackHits = iEvent.get(pixelTrackHitsToken_);
  auto const& magneticField = iSetup.getData(mfToken_);
  auto const& propagator = iSetup.getData(propagatorToken_);
  const SeedRecHitAccess access(iSetup.getData(tGeomToken_), iSetup.getData(tTopoToken_));

  const uint32_t nPixelRows = pixelState.metadata().size();
  if (static_cast<uint32_t>(tcState.metadata().size()) != nTrackCandidates)
    throw edm::Exception(edm::errors::LogicError)
        << "LSTOutputConverter: " << tcState.metadata().size() << " track-candidate seed states for "
        << nTrackCandidates << " track candidates";
  if (static_cast<uint32_t>(pixelITState.metadata().size()) != nPixelRows ||
      static_cast<uint32_t>(pixelExtra.metadata().size()) != nPixelRows ||
      pixelTrackHits.offsets.size() != nPixelRows + 1 || pixelTrackHits.offsets.back() != pixelTrackHits.hits.size())
    throw edm::Exception(edm::errors::LogicError)
        << "LSTOutputConverter: the pixel-track seed states (" << nPixelRows << ", " << pixelITState.metadata().size()
        << " and " << pixelExtra.metadata().size() << " rows) and the rechit offsets (" << pixelTrackHits.offsets.size()
        << " entries) of the pixel tracks do not match";

  auto const trackHits = [&pixelTrackHits](uint32_t seedIndex) {
    auto const& offsets = pixelTrackHits.offsets;
    return std::span<Hit const>(pixelTrackHits.hits.data() + offsets[seedIndex],
                                offsets[seedIndex + 1] - offsets[seedIndex]);
  };

  // the seed of pixel track seedIndex: its full state on all its hits, or for a pLS rebuilt without its OT hits the
  // inner-tracker-only state on its inner-tracker hits when that state exists
  auto const pixelTrackSeed = [&](uint32_t seedIndex, bool isPLS) {
    auto const hits = trackHits(seedIndex);
    const lst::DecodedStateHit full = lst::decodeStateHit(pixelState.stateHit()[seedIndex], 0, hits.size());
    if (full.kind != lst::StateHitKind::kPixel || full.index + 1 != hits.size())
      throw edm::Exception(edm::errors::LogicError)
          << "LSTOutputConverter: the seed state of pixel track " << seedIndex << " is not on its last hit";
    lst::PLSSeedHits choice{false, false, static_cast<uint32_t>(hits.size())};
    if (isPLS)
      choice =
          lst::plsSeedHits<Hit>(hits,
                                dropOTHitsPurePLS_,
                                maxITHitsToDropOTHitsPurePLS_,
                                pixelITState.status()[seedIndex] == static_cast<uint8_t>(lst::SeedStateStatus::kOk),
                                pixelITState.stateHit()[seedIndex],
                                pixelExtra.nITHits()[seedIndex],
                                access);
    if (choice.rebuildWithoutOT && !choice.itOnly)
      LogDebug("LSTOutputConverter") << "pLS of pixel track " << seedIndex
                                     << " kept with its OT hits: no usable inner-tracker-only state";
    const lst::SeedStateConst& states = choice.itOnly ? pixelITState : pixelState;
    auto result = lst::blfSeedFromState(stateRow(states, seedIndex),
                                        states.nFit()[seedIndex],
                                        states.B(),
                                        hits.first(choice.nHits),
                                        magneticField,
                                        propagator,
                                        /*attachIfStepFails=*/true);
    if (result.status != lst::BLFSeedStatus::kOk && result.status != lst::BLFSeedStatus::kAttachedAfterFailedStep)
      throw edm::Exception(edm::errors::LogicError)
          << "LSTOutputConverter: no seed for the " << (choice.itOnly ? "inner-tracker-only " : "")
          << "seed state of pixel track " << seedIndex << ", status " << static_cast<int>(result.status);
    return std::move(*result.seed);
  };

  std::vector<lst::OTSlotHit<Hit>> otSlotHits;
  std::vector<lst::SeedHitEntry<Hit>> entries;
  std::vector<Hit> seedHits;
  // the seed of track candidate i (pT3, pT5, T5, T4) from its own state, if it has one
  auto const trackCandidateSeed = [&](uint32_t i, lst::LSTObjType type, uint32_t seedIndex) {
    std::optional<TrajectorySeed> seed;
    if (tcState.status()[i] != static_cast<uint8_t>(lst::SeedStateStatus::kOk))
      return seed;
    const bool isT5orT4 = (type == lst::LSTObjType::T5 || type == lst::LSTObjType::T4);
    auto const pixelHits = isT5orT4 ? std::span<Hit const>() : trackHits(seedIndex);
    otSlotHits.clear();
    for (int layerSlot = lst::Params_TC::kPixelLayerSlots; layerSlot < lst::Params_TC::kLayers; ++layerSlot)
      for (int hitSlot = 0; hitSlot < lst::Params_TC::kHitsPerLayer; ++hitSlot) {
        const uint32_t hitIdx = tcs.hitIndices()[i][layerSlot][hitSlot];
        if (hitIdx == lst::kTCEmptyHitIdx)
          continue;
        if (hitIdx >= otHits.size())
          throw edm::Exception(edm::errors::LogicError)
              << "LSTOutputConverter: track candidate " << i << " has OT hit " << hitIdx << " of " << otHits.size();
        otSlotHits.push_back({otHits[hitIdx], hitIdx});
      }
    const uint32_t stateHit = tcState.stateHit()[i];
    const lst::DecodedStateHit decoded = lst::decodeStateHit(stateHit, otHits.size(), pixelHits.size());
    // an OT state hit is identified among the collected hits by the input it shares with its rechit, as the pixel
    // track's copy of an OT hit it carries is a distinct rechit
    const Hit stateOTHit = decoded.kind == lst::StateHitKind::kOT ? otHits[decoded.index] : nullptr;
    const auto list =
        (decoded.kind == lst::StateHitKind::kOT || decoded.kind == lst::StateHitKind::kPixel)
            ? lst::tcSeedHitList<Hit>(type, pixelHits, otSlotHits, stateHit, stateOTHit, access, entries, seedHits)
            : lst::SeedHitListResult{lst::SeedHitListStatus::kStateHitNotInList, false};
    if (list.status == lst::SeedHitListStatus::kStateHitNotInList)
      throw edm::Exception(edm::errors::LogicError)
          << "LSTOutputConverter: the seed state of track candidate " << i
          << " is on a hit that is not one of its hits, stateHit " << stateHit;
    if (list.status == lst::SeedHitListStatus::kStateHitDroppedBySubset) {
      LogDebug("LSTOutputConverter") << "track candidate " << i << " type " << static_cast<int>(type)
                                     << ": the hit subset of its seed drops the hit of its state, no seed";
      return seed;
    }
    auto result = lst::blfSeedFromState(stateRow(tcState, i),
                                        tcState.nFit()[i],
                                        tcState.B(),
                                        seedHits,
                                        magneticField,
                                        propagator,
                                        /*attachIfStepFails=*/false);
    LogDebug("LSTOutputConverter") << "track candidate " << i << " type " << static_cast<int>(type) << ": seed status "
                                   << static_cast<int>(result.status) << ", " << seedHits.size() << " hits"
                                   << (list.stateHitMoved ? ", the state's hit moved last" : "");
    seed = std::move(result.seed);
    return seed;
  };

  TrajectorySeedCollection outputTS, outputpLSTS;
  outputTS.reserve(nTrackCandidates);
  outputpLSTS.reserve(nTrackCandidates);
  LogDebug("LSTOutputConverter") << "nTrackCandidates " << nTrackCandidates << ", seeds from the stored states";
  for (uint32_t i = 0; i < nTrackCandidates; ++i) {
    const lst::LSTObjType type = tcs.trackCandidateType()[i];
    const bool isT5orT4 = (type == lst::LSTObjType::T5 || type == lst::LSTObjType::T4);
    const uint32_t seedIndex = tcs.pixelSeedIndex()[i];
    if (tcState.type()[i] != type || tcState.pixelSeedIndex()[i] != (isT5orT4 ? lst::kInvalidU32Idx : seedIndex))
      throw edm::Exception(edm::errors::LogicError)
          << "LSTOutputConverter: track candidate " << i << " (type " << static_cast<int>(type) << ", pixel seed "
          << seedIndex << ") and its seed state (type " << static_cast<int>(tcState.type()[i]) << ", pixel seed "
          << tcState.pixelSeedIndex()[i] << ") disagree";
    if (!isT5orT4 && seedIndex >= nPixelRows)
      throw edm::Exception(edm::errors::LogicError)
          << "LSTOutputConverter: track candidate " << i << " has pixel seed " << seedIndex << " of " << nPixelRows;
    // the makeSeed mode makes no seed without produceSeeds, and no pT3 or pT5 seed without includeNonpLSTSs
    if (!produceSeeds_ || (!isT5orT4 && type != lst::LSTObjType::pLS && !includeNonpLSTSs_))
      continue;
    if (type == lst::LSTObjType::pLS) {
      outputTS.push_back(pixelTrackSeed(seedIndex, /*isPLS=*/true));
      outputpLSTS.push_back(outputTS.back());
      continue;
    }
    auto seed = trackCandidateSeed(i, type, seedIndex);
    if (seed)
      outputTS.push_back(std::move(*seed));
    else if (!isT5orT4)
      outputTS.push_back(pixelTrackSeed(seedIndex, /*isPLS=*/false));
  }

  iEvent.emplace(trajectorySeedPutToken_, std::move(outputTS));
  if (produceSeeds_)
    iEvent.emplace(trajectorySeedpLSPutToken_, std::move(outputpLSTS));
}

DEFINE_FWK_MODULE(LSTOutputConverter);

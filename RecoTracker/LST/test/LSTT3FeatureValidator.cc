// Recomputes the LST T3 transformer features on the host, following the standalone LST ntuple code
// (write_lst_ntuple.cc / AccessHelper.cc / lst_math.h), and compares them with LSTT3FeatureProducer.
// Optionally appends the producer's features to a CSV file for comparison with transformer-oc graphs.

#include <cmath>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "RecoTracker/LSTCore/interface/LSTInputHostCollection.h"
#include "RecoTracker/LSTCore/interface/MiniDoubletsHostCollection.h"
#include "RecoTracker/LSTCore/interface/ObjectRangesHostCollection.h"
#include "RecoTracker/LSTCore/interface/SegmentsHostCollection.h"
#include "RecoTracker/LSTCore/interface/T3FeaturesHostCollection.h"
#include "RecoTracker/LSTCore/interface/TripletsHostCollection.h"

namespace {
  // lst_math::Hit
  float hitPhi(float x, float y) {
    float phi = M_PI + std::atan2(-y, -x);
    while (phi >= M_PI)
      phi -= 2 * M_PI;
    while (phi < -M_PI)
      phi += 2 * M_PI;
    return phi;
  }
  float hitEta(float x, float y, float z) {
    float rt = std::sqrt(x * x + y * y);
    float r3 = std::sqrt(x * x + y * y + z * z);
    return ((z > 0) - (z < 0)) * std::acosh(r3 / rt);
  }
  float reducePhi(float dphi) {
    while (dphi >= M_PI)
      dphi -= 2 * M_PI;
    while (dphi < -M_PI)
      dphi += 2 * M_PI;
    return dphi;
  }
  constexpr float k2Rinv1GeVf = (2.99792458e-3 * 3.8) / 2;  // as in write_lst_ntuple.cc
}  // namespace

class LSTT3FeatureValidator : public edm::global::EDAnalyzer<> {
public:
  explicit LSTT3FeatureValidator(edm::ParameterSet const& config)
      : lstInputToken_{consumes(config.getParameter<edm::InputTag>("lstInput"))},
        rangesToken_{consumes(config.getParameter<edm::InputTag>("lst"))},
        miniDoubletsToken_{consumes(config.getParameter<edm::InputTag>("lst"))},
        segmentsToken_{consumes(config.getParameter<edm::InputTag>("lst"))},
        tripletsToken_{consumes(config.getParameter<edm::InputTag>("lst"))},
        featuresToken_{consumes(config.getParameter<edm::InputTag>("features"))},
        maxT3Pt_(config.getParameter<double>("maxT3Pt")),
        tolerance_(config.getParameter<double>("tolerance")),
        csvFile_(config.getParameter<std::string>("csvFile")) {
    if (!csvFile_.empty()) {
      csv_.open(csvFile_);
      csv_ << "run,lumi,event,row,t3Index";
      for (unsigned int i = 0; i < lst::t3features::kN; ++i)
        csv_ << ",f" << i;
      csv_ << "\n";
    }
  }

  void analyze(edm::StreamID, edm::Event const& iEvent, edm::EventSetup const&) const override {
    auto const hits = iEvent.get(lstInputToken_).const_view().hits();
    auto const ranges = iEvent.get(rangesToken_).const_view();
    auto const mds = iEvent.get(miniDoubletsToken_).const_view().miniDoublets();
    auto const segments = iEvent.get(segmentsToken_).const_view().segments();
    auto const& tripletsHC = iEvent.get(tripletsToken_);
    auto const triplets = tripletsHC.const_view().triplets();
    auto const occupancy = tripletsHC.const_view().tripletsOccupancy();
    auto const features = iEvent.get(featuresToken_).const_view();

    // Reference: loop like setTripletBranches, apply the pt mask like t3_processing.py
    std::vector<unsigned int> refIndex;
    std::vector<std::array<float, lst::t3features::kN>> ref;
    for (int mod = 0; mod < occupancy.metadata().size(); ++mod) {
      for (unsigned int i = 0; i < occupancy.nTriplets()[mod]; ++i) {
        unsigned int t3 = ranges.tripletModuleIndices()[mod] + i;
        std::array<float, lst::t3features::kN> f{};
        // getLSsFromT3 / getMDsFromLS / getHitsFromT3
        unsigned int ls[2] = {triplets.segmentIndices()[t3][0], triplets.segmentIndices()[t3][1]};
        unsigned int md[2][2];
        for (int k = 0; k < 2; ++k)
          for (int j = 0; j < 2; ++j)
            md[k][j] = segments.mdIndices()[ls[k]][j];
        unsigned int t3Hits[6] = {mds.anchorHitIndices()[md[0][0]],
                                  mds.outerHitIndices()[md[0][0]],
                                  mds.anchorHitIndices()[md[0][1]],
                                  mds.outerHitIndices()[md[0][1]],
                                  mds.anchorHitIndices()[md[1][1]],
                                  mds.outerHitIndices()[md[1][1]]};
        f[0] = triplets.radius()[t3] * k2Rinv1GeVf * 2;
        if (!(f[0] < maxT3Pt_))
          continue;
        f[1] = hitEta(hits.xs()[t3Hits[4]], hits.ys()[t3Hits[4]], hits.zs()[t3Hits[4]]);
        f[2] = hitPhi(hits.xs()[t3Hits[0]], hits.ys()[t3Hits[0]]);
        for (int k = 0; k < 2; ++k) {
          // what CUT_VALUE_DEBUG stores in Segment.h
          float dPhiChange = segments.dPhiChanges()[ls[k]];
          float inAlpha = mds.dphichanges()[md[k][0]], outAlpha = mds.dphichanges()[md[k][1]];
          float* fls = &f[lst::t3features::kLS + 5 * k];
          fls[0] = reducePhi(mds.anchorPhi()[md[k][1]] - mds.anchorPhi()[md[k][0]]);
          fls[1] = dPhiChange;
          fls[2] = inAlpha - dPhiChange;
          fls[3] = outAlpha - dPhiChange;
          fls[4] = inAlpha - outAlpha;
        }
        for (int m = 0; m < 4; ++m) {
          unsigned int iMD = md[m / 2][m % 2];
          unsigned int a = mds.anchorHitIndices()[iMD], o = mds.outerHitIndices()[iMD];
          float* fmd = &f[lst::t3features::kMD + 9 * m];
          float v[9] = {hits.xs()[a],
                        hits.ys()[a],
                        hits.zs()[a],
                        hits.xs()[o],
                        hits.ys()[o],
                        hits.zs()[o],
                        mds.dphis()[iMD],
                        mds.dphichanges()[iMD],
                        mds.dzs()[iMD]};
          std::copy(v, v + 9, fmd);
        }
        refIndex.push_back(t3);
        ref.push_back(f);
      }
    }

    const unsigned int n = features.nT3s();
    unsigned int nBadIndex = 0, nBadValue = 0;
    float maxDiff[3] = {0, 0, 0};  // T3, LS, MD groups (relative to max(1, |ref|))
    unsigned int worstFeature = 0;
    float worstDiff = 0, worstRef = 0, worstVal = 0;
    if (n != ref.size())
      throw cms::Exception("LSTT3FeatureValidator") << "nT3s " << n << " != reference " << ref.size();
    for (unsigned int r = 0; r < n; ++r) {
      nBadIndex += features.tripletIndex()[r] != refIndex[r];
      bool bad = false;
      for (unsigned int i = 0; i < lst::t3features::kN; ++i) {
        float d = std::abs(features.features()[r][i] - ref[r][i]) / std::max(1.f, std::abs(ref[r][i]));
        if (!(d <= tolerance_))  // also catches NaN
          bad = true;
        int g = i < lst::t3features::kLS ? 0 : (i < lst::t3features::kMD ? 1 : 2);
        maxDiff[g] = std::max(maxDiff[g], d);
        if (d > worstDiff) {
          worstDiff = d;
          worstFeature = i;
          worstRef = ref[r][i];
          worstVal = features.features()[r][i];
        }
      }
      nBadValue += bad;
    }
    edm::LogPrint("LSTT3FeatureValidator")
        << "event " << iEvent.id().event() << ": nT3s=" << n << " (slots " << triplets.metadata().size()
        << ") badIndex=" << nBadIndex << " badRows=" << nBadValue << " maxRelDiff T3/LS/MD=" << maxDiff[0] << "/"
        << maxDiff[1] << "/" << maxDiff[2] << " worst f" << worstFeature << " (" << worstVal << " vs " << worstRef << ")"
        << (nBadIndex + nBadValue == 0 ? " OK" : " FAIL");

    if (csv_.is_open()) {
      std::lock_guard<std::mutex> guard(csvMutex_);
      for (unsigned int r = 0; r < n; ++r) {
        csv_ << iEvent.id().run() << ',' << iEvent.id().luminosityBlock() << ',' << iEvent.id().event() << ',' << r << ','
             << features.tripletIndex()[r];
        for (unsigned int i = 0; i < lst::t3features::kN; ++i)
          csv_ << ',' << features.features()[r][i];
        csv_ << '\n';
      }
    }
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("lstInput", edm::InputTag{"lstInputProducer"});
    desc.add<edm::InputTag>("lst", edm::InputTag{"lstProducer"});
    desc.add<edm::InputTag>("features", edm::InputTag{"lstT3FeatureProducer"});
    desc.add<double>("maxT3Pt", 2000.);
    desc.add<double>("tolerance", 1e-5)->setComment("max |diff| / max(1, |ref|) per feature");
    desc.add<std::string>("csvFile", "")->setComment("if set, write the producer's features to this CSV");
    descriptions.addWithDefaultLabel(desc);
  }

private:
  const edm::EDGetTokenT<lst::LSTInputHostCollection> lstInputToken_;
  const edm::EDGetTokenT<lst::ObjectRangesHostCollection> rangesToken_;
  const edm::EDGetTokenT<lst::MiniDoubletsHostCollection> miniDoubletsToken_;
  const edm::EDGetTokenT<lst::SegmentsHostCollection> segmentsToken_;
  const edm::EDGetTokenT<lst::TripletsHostCollection> tripletsToken_;
  const edm::EDGetTokenT<lst::T3FeaturesHostCollection> featuresToken_;
  const double maxT3Pt_;
  const double tolerance_;
  const std::string csvFile_;
  mutable std::ofstream csv_;
  mutable std::mutex csvMutex_;
};

DEFINE_FWK_MODULE(LSTT3FeatureValidator);

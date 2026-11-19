#ifndef RecoTracker_MkSeederAlpaka_plugins_alpaka_MkSeederClean_h
#define RecoTracker_MkSeederAlpaka_plugins_alpaka_MkSeederClean_h

// The cleaning of the chain finder's quads and triplet seeds: the kept seeds (with their hit rows, the fit's input)
// and the triplet-MLP probe items. The algorithm is described at cleanQuadsAndTriplets() (MkSeederClean.dev.cc); the
// kernels are in MkSeederCleanKernels.h.

#include <cstdint>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

#include "MkSeederDeviceTypes.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkseeder {

  struct CleanParameters {
    int nPatterns;             // the configured patterns (in pattern order); a combination not among them: nPatterns
    int nPatternTableEntries;  // entries of the pattern table
    int flagFailedTriplets;  // trip_dnn_ext and trip_dnn: a triplet that failed the triplet MLP (tripletFailedMlp) is flagged
    int dedup, dedupInner, nSharedHitWalk, tripCleanMin;
    int outerTrackerQuadsVetoTriplets;  // 0 (min_pix_ot): the kept quads with >= 2 outer-tracker hits veto no triplet
    int tripletVetoBySingleQuad;        // trip_veto_quad: one kept quad must hold tripCleanMin of the triplet's hits
    float tripletVetoMaxQuadFakeScore;  // > 0: only kept quads with a lower fake score veto (trip_veto_fk)
    float tripletVetoMaxTripletScore;   // > 0: only kept better triplets with a lower score veto (trip_veto_tsc)
    int probesEnabled;                  // the flagged kept triplets get probe items
    int nQuadRounds, nTripletRounds;    // the rounds launched (then the completion)
  };

  // (the views are mutable: the kernels' helpers take the view structs by const reference and still write through
  // them, as through plain pointers)
  struct CleanInputs {
    CleanParameters parameters;
    mutable QuadsConstView quads;     // the chain finder's (in the sorted order: quadOfSortedIndex)
    const uint32_t *count;            // the quads (the chain finder's counter)
    uint32_t cap;                     // the quads' capacity, also the kept seeds'
    PatternsView patterns;            // the patterns' order (sorted by their packed layers)
    LayerGeometryView layerGeometry;  // per mkFit layer: isPixel (the HitVec), isPixelEnvelope
    EventLayersView layers;           // per mkFit layer: firstHit (its first cluster-order position)
    // cluster index -> cluster-order position (hitPosition, MkSeederBinKernels.h), per HitVec
    PixelClustersView pixelClusters;
    OTClustersView otClusters;
    ClusterOrderView clusterOrder;  // cluster-order position -> bin-order row (binOrderPosition)
    HitsView hits;                  // bin order: x, y, z
    uint32_t nTot;
    MkSeederCleanCounters *counters;
    // out
    mutable KeptSeedsView kept;  // per label
  };

  // ---- the duplicate cleanings with their merges (MkSeederDupKernels.h). Call 1 on the seeds of the kept seeds' fits, giving the post step's input; call 2
  // (pix_fill_dedup) on the post step's output, giving the final seeds with their beam-line states and quality
  struct DupParameters {
    float maxSigma2, dphi, dz, highPtThreshold, highPtMaxDPhi0, highPtMaxDPhi1, highPtMaxDz, deta, noShareEta,
        noShareMaxSigma2, noShareDphi, noShareDz, noShareRes, phiBinWidth;
    int shared, merge, noShareMerge, noShare, nPhiBins, rounds;
    int noShareTriplets;  // 0: a seed of 3 hits is not removed by a no-share neighbour (dup_ns_trip)
    int sharedTriplet;  // the shared hits for a seed of 3 hits to be removed by a sharer (dup_shared_trip, else shared)
    int firstEtaBin, nEtaBins;  // the no-share eta bins' range; the seeds outside it share one overflow bin (the last)
    int pass;                   // 1 or 2
    int mergedNoShare;          // a strong merged-cluster candidate hit is not a shared hit (mrg_share)
    // the merge's OT hits (dup_merge_ot bits for this pass, 0: all; MkSeederConfig.h)
    int mergeOT;
    float mergeOTSigma2, mergeOTMaxScore, psCurvatureSigma, psCurvatureResolution;
  };
  struct DupInputs {
    DupParameters parameters;
    uint32_t nK;  // the capacity: call 1 the kept items (the cleaning's labels before the probe), call 2 the post seeds
    // call 1 inputs: the kept seeds, their fits, the probe items
    mutable KeptSeedsConstView kept;
    mutable SeedStatesConstView fits;
    mutable SeedHelicesConstView helices;  // call 1 per kept seed, call 2 per post seed
    mutable SeedHitsConstView probe;
    const MkSeederCleanCounters *cleanCounters;  // the probe items (nProbesWritten)
    uint32_t probeCap;
    // call 2 inputs: the post step's output (its seeds' labels, their quality)
    mutable SeedHitsConstView postHits;
    mutable SeedStatesConstView postStates;
    mutable SeedQualityConstView postQuality;
    const uint32_t *postCount;        // the post seeds (call 1's MkSeederDupCounters::nOutput)
    LayerGeometryView layerGeometry;  // per mkFit layer: isPixel (the seeds' tiers)
    HitsView hits;                    // bin order: x, y, z, detIdInLayer
    EventLayersView layers;           // per mkFit layer: firstHit, firstOTHit (dup_merge_ot 2)
    OTHitsView otHits;                // the OT hits' PS marks (dup_merge_ot 2)
    uint32_t nTot;
    uint32_t nBins;  // the no-share bins, the overflow bin included
    MkSeederDupCounters *counters;
    // out (capacity nK): call 1 the post step's input (with its cot), call 2 the final seeds (states with their
    // beam-line states), in the kept seeds' order, with their labels and quality (the kept seed's of the label)
    mutable SeedHitsView outHits;
    mutable SeedStatesView outStates;
    mutable SeedQualityView outQuality;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkseeder

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkseeder {

  // a duplicate cleaning; v's inputs and outputs set, its state allocated here. The refit of the merged seeds with
  // fit (call 1: fitSeeds' kernel; call 2: refitSeedsWithBeamline's)
  void cleanDuplicateSeeds(Queue &queue, mkseeder::DupInputs v, mkseeder::FitInputs const &fit);

  // the probe items of the nKept kept seeds (the kept triplets that failed the triplet MLP, counted by the cleaning
  // in counters->nProbes: probe has that many rows)
  void buildProbeSeeds(Queue &queue,
                       mkseeder::KeptSeedsConstView kept,
                       mkseeder::HitsView hits,
                       ::mkseeder::MkSeederCleanCounters *counters,
                       mkseeder::SeedHitsView probe,
                       uint32_t nKept);

  // the final seeds' perigees and states at the beam line (MkSeederPcaKernels.h): the classifier's state features
  void computeSeedPerigees(Queue &queue, mkseeder::PcaInputs const &v);
  // the final seeds' states at their last pixel hit (seeds whose last hit is an outer-tracker hit; MkSeederPcaKernels.h)
  void computeSeedPixelStates(Queue &queue, mkseeder::PcaInputs const &v, mkseeder::SeedPixelStatesView out);
  // the final seeds' hits' cluster indices (the hits' clusterIndex of their hitRows): the first *count of cap seeds
  void setSeedClusterIndices(Queue &queue,
                             mkseeder::SeedHitsConstView seeds,
                             mkseeder::HitsView hits,
                             const uint32_t *count,
                             uint32_t cap,
                             mkseeder::SeedClusterIndicesView out);
  // their other classifier features (MkSeederPcaKernels.h); the sharing's lists allocated here
  void computeClassifierFeatures(Queue &queue, mkseeder::ClassifierFeatureInputs v);

  // per fitted seed (fitOk; the post step's states: all) of the first *count (nullptr: cap) its helix (device trig,
  // equal to the host libm's up to float rounding)
  void computeSeedHelices(Queue &queue,
                          mkseeder::SeedStatesConstView states,
                          bool checkFitOk,
                          uint32_t cap,
                          const uint32_t *count,
                          mkseeder::SeedHelicesView helices,
                          float bField);

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkseeder

#endif
